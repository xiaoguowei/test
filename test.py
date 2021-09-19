import os
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pprint import pprint
from IPython.display import clear_output
import tensorflow_datasets as tfds
import tensorflow as tf

output_dir = "nmt"
en_vocab_file = os.path.join(output_dir, "en_vocab") #英文字典的路径 test/nmt/en_vocab
zh_vocab_file = os.path.join(output_dir, "zh_vocab") ##中文字典的路径 test/nmt/zh_vocab
checkpoint_path = os.path.join(output_dir, "checkpoints") #存儲模型的路徑
log_dir = os.path.join(output_dir, 'logs')
download_dir = "tensorflow-datasets/downloads" #下載數據的路徑 - (下載完下次訓練就不用加載數據了,直接導入加快執行速度)

if not os.path.exists(output_dir):
  os.makedirs(output_dir)

list_tfds = tfds.list_builders()
tfds_zh_en=tfds.builder("wmt19_translate/zh-en")
config = tfds.translate.wmt.WmtConfig(
  version=tfds.core.Version('0.0.3', experiments=None),
  language_pair=("zh", "en"),
  subsets={
    tfds.Split.TRAIN: ["newscommentary_v14"]
  }
)

builder = tfds.builder("wmt_translate", config=config)
builder.download_and_prepare(download_dir=download_dir)
clear_output()

train_examples = builder.as_dataset(split=tfds.Split.TRAIN, as_supervised=True)
print("數據集總共有：",len(train_examples))

#中英字典
try:
  tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.load_from_file(en_vocab_file)
  print(f"載入已建立的英文字典： {en_vocab_file}")
except:
  print("沒有已建立的英文字典，從頭建立。")
  tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
      (en.numpy() for en, _ in train_examples),
      target_vocab_size=2**13)

tokenizer_en.save_to_file(en_vocab_file)
print(f"字典大小：{tokenizer_en.vocab_size}")
print(f"前 10 個 subwords：{tokenizer_en.subwords[:10]}")
print()

try:
  tokenizer_zh = tfds.deprecated.text.SubwordTextEncoder.load_from_file(zh_vocab_file)
  print(f"載入已建立的中文字典： {zh_vocab_file}")
except:
  print("沒有已建立的中文字典，從頭建立。")
  tokenizer_zh = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
      (zh.numpy() for _, zh in train_examples),
      target_vocab_size=2**13, # 有需要可以調整字典大小
      max_subword_length=1) # 每一個中文字就是字典裡的一個單位

tokenizer_zh.save_to_file(zh_vocab_file)
print(f"字典大小：{tokenizer_zh.vocab_size}")
print(f"前 10 個 subwords：{tokenizer_zh.subwords[:10]}")
print()

# 用字典將原文轉換成相對應的索引序列，並且序列添加開頭(BOS)和結束(EOS)Token
# 英文 BOS 的 index： 8177
# 英文 EOS 的 index： 8178
# 中文 BOS 的 index： 4891
# 中文 EOS 的 index： 4892
def encode(en_token,zh_token):
  # 因為字典的索引從 0 開始，
  # 我們可以使用 tokenizer_en.vocab_size 這個值作為 BOS 的索引值
  # 用 tokenizer_en.vocab_size + 1 作為 EOS 的索引值
  en_indices = [tokenizer_en.vocab_size] + tokenizer_en.encode(en_token.numpy()) + [tokenizer_en.vocab_size + 1]
  # 同理，不過是使用中文字典的最後一個索引 + 1
  zh_indices = [tokenizer_zh.vocab_size] + tokenizer_zh.encode(zh_token.numpy()) + [tokenizer_zh.vocab_size + 1]
  return en_indices, zh_indices


# 因為 tf.data.Dataset 裡頭都是在操作 Tensors（而非 Python 字串），
# 所以這個 encode 函式預期的輸入也是 TensorFlow 裡的 Eager Tensors。
# 但只要我們使用 numpy() 將 Tensor 裡的實際字串取出以後，做的事情就跟上一節完全相同
# 使用 tf.py_function 將我們剛剛定義的 encode 函式包成一個以 eager 模式執行的 TensorFlow Op
def tf_encode(en_t, zh_t):
  # 在 `tf_encode` 函式裡頭的 `en_t` 與 `zh_t` 都不是 Eager Tensors
  # 要到 `tf.py_funtion` 裡頭才是
  # 另外因為索引都是整數，所以使用 `tf.int64`
  return tf.py_function(encode, [en_t, zh_t], [tf.int64, tf.int64])


# 去掉長度超過40的數據,剩餘217967筆
MAX_LENGTH = 40
def filter_max_length(en, zh, max_length=MAX_LENGTH):
  # en, zh 分別代表英文與中文的索引序列
  return tf.logical_and(tf.size(en) <= max_length,tf.size(zh) <= max_length)

#################################建立輸入管道##################################################
MAX_LENGTH = 40
BATCH_SIZE = 64
BUFFER_SIZE = 20000
# 訓練集(因為有去掉超過40長度的數據,所以剩217967筆)
train_dataset = (train_examples  # 輸出：(英文句子, 中文句子)
                 .map(tf_encode) # 輸出：(英文索引序列, 中文索引序列)
                 .filter(filter_max_length) # 同上，且序列長度都不超過 40
                 .cache() # 加快讀取數據
                 .shuffle(BUFFER_SIZE) # 將例子洗牌確保隨機性
                 .padded_batch(BATCH_SIZE, # 將 batch 裡的序列都 pad 到batch中最長的一句的長度
                               padded_shapes=([-1], [-1]))
                 .prefetch(tf.data.experimental.AUTOTUNE)) # 加速

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates
def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # 将 sin 应用于数组中的偶数索引（indices）；2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # 将 cos 应用于数组中的奇数索引；2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


#用來識別序列實際的內容到哪裡。此遮罩負責的就是將序列中被補 0 的地方（也就是 <pad>）的位置蓋住，讓 Transformer 可以避免「關注」到這些位置
def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # 添加额外的维度来将填充加到
  # 注意力对数（logits）。
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


#用來確保 Decoder 在進行自注意力機制時輸出序列裡頭的每個子詞只會關注到自己之前（左邊）的字詞，
#不會不小心關注到未來（右邊）理論上還沒被 Decoder 生成的子詞
def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)


#定義一個簡單函式來產生所有的遮罩
def create_masks(inp, tar):
  # 英文句子的 padding mask，要交給 Encoder layer 自注意力機制用的
  enc_padding_mask = create_padding_mask(inp)

  # 同樣也是英文句子的 padding mask，但是是要交給 Decoder layer 的 MHA 2
  # 關注 Encoder 輸出序列用的
  dec_padding_mask = create_padding_mask(inp)

  # Decoder layer 的 MHA1 在做自注意力機制用的
  # `combined_mask` 是中文句子的 padding mask 跟 look ahead mask 的疊加
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

  return enc_padding_mask, combined_mask, dec_padding_mask



# 建立 Transformer 裡 Encoder / Decoder layer 都有使用到的 Feed Forward 元件
def point_wise_feed_forward_network(d_model, dff):
  # 此 FFN 對輸入做兩個線性轉換，中間加了一個 ReLU activation func
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


def scaled_dot_product_attention(q, k, v, mask):
  """计算注意力权重。
  q, k, v 必须具有匹配的前置维度。
  k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
  虽然 mask 根据其类型（填充或前瞻）有不同的形状，
  但是 mask 必须能进行广播转换以便求和。

  参数:
    q: 请求的形状 == (..., seq_len_q, depth)
    k: 主键的形状 == (..., seq_len_k, depth)
    v: 数值的形状 == (..., seq_len_v, depth_v)
    mask: Float 张量，其形状能转换成
          (..., seq_len_q, seq_len_k)。默认为None。

  返回值:
    输出，注意力权重
  """
  ## 將 `q`、 `k` 做點積再 scale
  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # 缩放 matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32) #取得輸入序列(seq_k)的長度
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # 將遮罩「加」到被丟入 softmax 前的 logits
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)   #極大負值的位置變得無關緊要，在經過 softmax 以後的值趨近於 0

  # softmax 在最后一个轴（seq_len_k）上归一化，因此分数相加等于1。
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  # 以注意權重對 v 做加權平均（weighted average）
  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

# 在初始的時候指定輸出維度 `d_model` & `num_heads，
# 在呼叫的時候輸入 `v`, `k`, `q` 以及 `mask`
# 輸出跟 scaled_dot_product_attention 函式一樣有兩個：
# output.shape      == (batch_size, seq_len_q, d_model)
# attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads # 指定要將 `d_model` 拆成幾個 heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0 # 要確保可以平分

    self.depth = d_model // self.num_heads # 每個 head 裡子詞的新的 repr. 維度

    self.wq = tf.keras.layers.Dense(d_model) # 分別給 q, k, v 的 3 個線性轉換
    self.wk = tf.keras.layers.Dense(d_model) # 注意我們並沒有指定 activation func
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model) # # 多 heads 串接後通過的線性轉換

  def split_heads(self, x, batch_size):
    # (batch_size, seq_len, d_model)
    # 把d_model分拆為(num_heads, depth).
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth)) #這邊形狀為(batch_size, seq_len, num_heads, depth)

    # 將 head 的維度拉前使得最後兩個維度為子詞以及其對應的 depth 向量
    return tf.transpose(x, perm=[0, 2, 1, 3]) #转置结果使得形状为 (batch_size, num_heads, seq_len, depth)

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    # 將輸入的 q, k, v 都各自做一次線性轉換到 `d_model` 維空間
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    # 將最後一個 `d_model` 維度分成 `num_heads` 個 `depth` 維度
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)


    #讓每個句子的每個 head 的 qi, ki, vi 都各自進行注意力機制
    # 輸出會多一個 head 維度
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights


# Encoder 裡頭會有 N 個 EncoderLayers，而每個 EncoderLayer 裡又有兩個 sub-layers: MHA & FFN
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        ## Transformer 論文內預設 dropout rate 為 0.1
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        # layer norm 很常在 RNN-based 的模型被使用。一個 sub-layer 一個 layer norm
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # 一個 sub-layer 一個 dropout layer
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # 除了 `attn`，其他張量的 shape 皆為 (batch_size, input_seq_len, d_model)
        # attn.shape == (batch_size, num_heads, input_seq_len, input_seq_len)

        # sub-layer 1: MHA
        # Encoder 利用注意機制關注自己當前的序列，因此 v, k, q 全部都是自己
        # 另外別忘了我們還需要 padding mask 來遮住輸入序列中的 <pad> token
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        # sub-layer 2: FFN
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


# Decoder 裡頭會有 N 個 DecoderLayer，
# 而 DecoderLayer 又有三個 sub-layers: 自注意的 MHA, 關注 Encoder 輸出的 MHA & FFN
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        # 3 個 sub-layers
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        # 定義每個 sub-layer 用的 LayerNorm
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # 定義每個 sub-layer 用的 Dropout
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # 所有 sub-layers 的主要輸出皆為 (batch_size, target_seq_len, d_model)
        # enc_output 為 Encoder 輸出序列，shape 為 (batch_size, input_seq_len, d_model)
        # attn_weights_block_1 則為 (batch_size, num_heads, target_seq_len, target_seq_len)
        # attn_weights_block_2 則為 (batch_size, num_heads, target_seq_len, input_seq_len)

        # sub-layer 1: Decoder layer 自己對輸出序列做注意力。
        # 我們同時需要 look ahead mask 以及輸出序列的 padding mask
        # 來避免前面已生成的子詞關注到未來的子詞以及 <pad>
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # sub-layer 2: Decoder layer 關注 Encoder 的最後輸出
        # 記得我們一樣需要對 Encoder 的輸出套用 padding mask 避免關注到 <pad>
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        # sub-layer 3: FFN
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)  # 記得 training
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        # 除了主要輸出 `out3` 以外，輸出 multi-head 注意權重方便之後理解模型內部狀況
        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 rate=0.1):
        # Encoder 的初始參數除了本來就要給 EncoderLayer 的參數還多了：
        # - num_layers: 決定要有幾個 EncoderLayers, 前面影片中的 `N`
        # - input_vocab_size: 用來把索引轉成詞嵌入向量
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)

        # 建立 `num_layers` 個 EncoderLayers
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # 輸入的 x.shape == (batch_size, input_seq_len)
        # 以下各 layer 的輸出皆為 (batch_size, input_seq_len, d_model)
        seq_len = tf.shape(x)[1]

        # 將 2 維的索引序列轉成 3 維的詞嵌入張量，並依照論文乘上 sqrt(d_model)
        # 再加上對應長度的位置編碼
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        # 對 embedding 跟位置編碼的總合做 regularization
        # 這在 Decoder 也會做
        x = self.dropout(x, training=training)

        # 通過 N 個 EncoderLayer 做編碼
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    # 初始參數跟 Encoder 只差在用 `target_vocab_size` 而非 `inp_vocab_size`
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # 為中文（目標語言）建立詞嵌入層
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(target_vocab_size, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}  # 用來存放每個 Decoder layer 的注意權重

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            # 將從每個 Decoder layer 取得的注意權重全部存下來回傳，方便我們觀察
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    # 初始參數包含 Encoder & Decoder 都需要超參數以及中英字典數目
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, rate)

        # 這個 FFN 輸出跟中文字典一樣大的 logits 數，等通過 softmax 就代表每個中文字的出現機率
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    # enc_padding_mask 跟 dec_padding_mask 都是英文序列的 padding mask，
    # 只是一個給 Encoder layer 的 MHA 用，一個是給 Decoder layer 的 MHA 2 使用
    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        # 將 Decoder 輸出通過最後一個 linear layer
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none') #reduction 參數設為 none 請 loss_object 不要把每個位置的 error 加總

def loss_function(real, pred):
  # 這次的 mask 將序列中不等於 0 的位置視為 1，其餘為 0
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  # 照樣計算所有位置的 cross entropy 但不加總
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask # 只計算非 <pad> 位置的損失

  return tf.reduce_mean(loss_)

#定義兩個tf.keras.metrics，方便之後使用 TensorBoard 來追蹤模型 performance：
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')


# num_layers 決定 Transfomer 裡頭要有幾個 Encoder / Decoder layers
# d_model 決定我們子詞的 representation space 維度
# num_heads 要做幾頭的自注意力運算
# dff 決定 FFN 的中間維度
# dropout_rate 預設 0.1，一般用預設值即可
# input_vocab_size：輸入語言（英文）的字典大小
# target_vocab_size：輸出語言（中文）的字典大小

num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = tokenizer_en.vocab_size + 2
target_vocab_size = tokenizer_zh.vocab_size + 2
dropout_rate = 0.1
print("input_vocab_size: ",input_vocab_size )
print("target_vocab_size: ",target_vocab_size )

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  # 論文預設 `warmup_steps` = 4000
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# 將客製化 learning rate schdeule 丟入 Adam opt.
# Adam opt. 的參數都跟論文相同
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

transformer = Transformer(num_layers,
              d_model,
              num_heads,
              dff,
              input_vocab_size,
              target_vocab_size,
              rate=dropout_rate)

# 方便比較不同實驗/ 不同超參數設定的結果
run_id = f"{num_layers}layers_{d_model}d_{num_heads}heads_{dff}dff"
checkpoint_path = os.path.join(checkpoint_path, run_id)
log_dir = os.path.join(log_dir, run_id)

# tf.train.Checkpoint 可以幫我們把想要存下來的東西整合起來，方便儲存與讀取
# 一般來說你會想存下模型以及 optimizer 的狀態
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

# ckpt_manager 會去 checkpoint_path 看有沒有符合 ckpt 裡頭定義的東西
# 存檔的時候只保留最近 5 次 checkpoints，其他自動刪除
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# 如果检查点存在，则恢复最新的检查点。
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)

    # 用來確認之前訓練多少 epochs 了
    last_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])
    print(f'已讀取最新的 checkpoint，模型已訓練 {last_epoch} epochs。')
else:
    last_epoch = 0
    print("沒找到 checkpoint，從頭訓練。")

# 该 @tf.function 将追踪-编译 train_step 到 TF 图中，以便更快地
# 执行。该函数专用于参数张量的精确形状。为了避免由于可变序列长度或可变
# 批次大小（最后一批次较小）导致的再追踪，使用 input_signature 指定
# 更多的通用形状。
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)  # 讓 TensorFlow 幫我們將 eager code 優化並加快運算
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    # 紀錄 Transformer 的所有運算過程以方便之後做梯度下降
    with tf.GradientTape() as tape:
        # 注意是丟入 `tar_inp` 而非 `tar`。記得將 `training` 參數設定為 True
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        # 計算左移一個字的序列跟模型預測分佈之間的差異，當作 loss
        loss = loss_function(tar_real, predictions)

    # 取出梯度並呼叫前面定義的 Adam optimizer 幫我們更新 Transformer 裡頭可訓練的參數
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    # 將 loss 以及訓練 acc 記錄到 TensorBoard 上，非必要
    train_loss(loss)
    train_accuracy(tar_real, predictions)


# 用來寫資訊到 TensorBoard，非必要但十分推薦
summary_writer = tf.summary.create_file_writer(log_dir)
EPOCHS = 3
plt_loss = []  # 用來儲存每次EPOCH的loss用來畫圖
plt_accuracy = []  # 用來儲存每次EPOCH的accuracy用來畫圖
for epoch in range(EPOCHS):
    start = time.time()

    # 重置紀錄 TensorBoard 的 metrics
    train_loss.reset_states()
    train_accuracy.reset_states()

    # 一個 epoch 就是把我們定義的訓練資料集一個一個 batch 拿出來處理，直到看完整個數據集
    for (batch, (inp, tar)) in enumerate(train_dataset):
        # 每次 step 就是將數據丟入 Transformer，讓它生預測結果並計算梯度最小化 loss
        train_step(inp, tar)

        # 每50batch打印一次 loss,accuracy
        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    # 每5個 epoch 完成就存1次檔
    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

        # 將 loss 以及 accuracy 寫到 TensorBoard 上
    with summary_writer.as_default():
        tf.summary.scalar("train_loss", train_loss.result(), step=epoch + 1)
        tf.summary.scalar("train_acc", train_accuracy.result(), step=epoch + 1)
    plt_loss.append(train_loss.result())
    plt_accuracy.append(train_accuracy.result())
    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))



EPOCH=np.arange(1, EPOCHS+1, 1)
print(plt_loss)
print(plt_accuracy)
plt.plot(EPOCH,plt_loss) # (x, y)
plt.xlabel("epochs",fontsize=20)
plt.ylabel("loss",fontsize=20)
plt.savefig("loss.png")

plt.plot(EPOCH,plt_accuracy) # (x, y)
plt.xlabel("epochs",fontsize=20)
plt.ylabel("acurracy",fontsize=20)
plt.savefig("acurracy.png")

## 給定一個英文句子，輸出預測的中文索引數字序列以及注意權重 dict
def evaluate(inp_sentence):
  # 準備英文句子前後會加上的 <start>, <end>
  start_token = [tokenizer_en.vocab_size]
  end_token = [tokenizer_en.vocab_size + 1]

  # inp_sentence 是字串，我們用 Subword Tokenizer 將其變成子詞的索引序列
  # 並在前後加上 BOS / EOS
  inp_sentence = start_token + tokenizer_en.encode(inp_sentence) + end_token
  encoder_input = tf.expand_dims(inp_sentence, 0)

  # Decoder 在第一個時間點吃進去的輸入
  # 是一個只包含一個中文 <start> token 的序列
  decoder_input = [tokenizer_zh.vocab_size]
  output = tf.expand_dims(decoder_input, 0) # 增加 batch 維度

  # auto-regressive，一次生成一個中文字並將預測加到輸入再度餵進 Transformer
  for i in range(MAX_LENGTH):
    # 每多一個生成的字就得產生新的遮罩
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = transformer(encoder_input,
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)

    # 將序列中最後一個 distribution 取出，並將裡頭值最大的當作模型最新的預測字
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # 如果 predicted_id 等于结束标记，就返回结果
    if predicted_id == tokenizer_zh.vocab_size+1:
      return tf.squeeze(output, axis=0), attention_weights

    # 將 Transformer 新預測的中文索引加到輸出序列中，讓 Decoder 可以在產生
    # 下個中文字的時候關注到最新的 `predicted_id`
    output = tf.concat([output, predicted_id], axis=-1)
  # 將 batch 的維度去掉後回傳預測的中文索引序列
  return tf.squeeze(output, axis=0), attention_weights

layer_name = f"decoder_layer{num_layers}_block2"

print("sentence:", sentence)
print("-" * 20)
print("predicted_seq:", predicted_seq)
print("-" * 20)
print("attention_weights.keys():")
for layer_name, attn in attention_weights.items():
  print(f"{layer_name}.shape: {attn.shape}")
print("-" * 20)
print("layer_name:", layer_name)

#view attention
# 這個函式將英 -> 中翻譯的注意權重視覺化（注意：我們將注意權重 transpose 以最佳化渲染結果
zhfont = mpl.font_manager.FontProperties(fname='chinese.simhei.ttf')
plt.style.use("seaborn-whitegrid")
def plot_attention_weights(attention_weights, sentence, predicted_seq, layer_name,max_len_tar=None):
  fig = plt.figure(figsize=(17, 7))

  sentence = tokenizer_en.encode(sentence)
  # 只顯示中文序列前 `max_len_tar` 個字以避免畫面太過壅擠
  if max_len_tar:
    predicted_seq = predicted_seq[:max_len_tar]
  else:
    max_len_tar = len(predicted_seq)

  # 將某一個特定 Decoder layer 裡頭的 MHA 1 或 MHA2 的注意權重拿出來並去掉 batch 維度
  attention_weights = tf.squeeze(attention_weights[layer_name], axis=0)

  # 將每個 head 的注意權重畫出
  for head in range(attention_weights.shape[0]):
    ax = fig.add_subplot(2, 4, head+1)

    # 画出注意力权重
    # [注意]我為了將長度不短的英文子詞顯示在 y 軸，將注意權重做了 transpose
    attn_map = np.transpose(attention_weights[head][:max_len_tar, :])
    ax.matshow(attn_map, cmap='viridis')  # (inp_seq_len, tar_seq_len)
    # ax.matshow(attention[head][:-1, :], cmap='viridis')

    fontdict = {"fontproperties": zhfont}

    ax.set_xticks(range(max(max_len_tar, len(predicted_seq))))
    ax.set_xlim(-0.5, max_len_tar -1.5)


    ax.set_yticks(range(len(sentence) + 2))
    ax.set_xticklabels([tokenizer_zh.decode([i]) for i in predicted_seq
                        if i < tokenizer_zh.vocab_size],
                       fontdict=fontdict, fontsize=18)
    ax.set_yticklabels(
        ['<start>']+[tokenizer_en.decode([i]) for i in sentence]+['<end>'],
                       fontdict=fontdict)


    ax.set_xlabel('Head {}'.format(head+1))
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

  plt.tight_layout()
  plt.savefig("Multi_Head_Attention_8")

