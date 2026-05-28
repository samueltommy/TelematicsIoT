# =========================================================
# transformers.py
# Modul Arsitektur Kombinasi LSTM + Multi-Head Attention (MindSpore)
# =========================================================
import numpy as np
import mindspore as ms
from mindspore import nn, ops

class MultiHeadAttention(nn.Cell):
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model harus habis dibagi oleh num_heads!"

        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = nn.Dense(d_model, d_model)
        self.wk = nn.Dense(d_model, d_model)
        self.wv = nn.Dense(d_model, d_model)

        self.dense = nn.Dense(d_model, d_model)
        
        # PERBAIKAN: Menggunakan keep_prob (1 - dropout_rate) di MindSpore
        self.dropout = nn.Dropout(keep_prob=1.0 - dropout_rate)

        self.softmax = ops.Softmax(axis=-1)
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.matmul = ops.BatchMatMul()

    def split_heads(self, x):
        batch = x.shape[0]
        x = self.reshape(x, (batch, -1, self.num_heads, self.depth))
        return self.transpose(x, (0, 2, 1, 3))

    def construct(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        score = self.matmul(q, self.transpose(k, (0, 1, 3, 2)))
        score = score / np.sqrt(self.depth)

        attn = self.softmax(score)
        out = self.matmul(attn, v)

        out = self.transpose(out, (0, 2, 1, 3))
        batch = out.shape[0]
        out = self.reshape(out, (batch, -1, self.num_heads * self.depth))

        out = self.dense(out)
        return self.dropout(out)


class LSTMTransformer(nn.Cell):
    def __init__(self, input_dim, lstm_units=64, num_heads=4, ff_dim=128, output_dim=3, dropout_rate=0.1):
        super().__init__()

        self.lstm = nn.LSTM(input_dim, lstm_units, batch_first=True)
        self.attn = MultiHeadAttention(lstm_units, num_heads, dropout_rate)

        self.add = ops.Add()
        self.norm1 = nn.LayerNorm([lstm_units])
        self.norm2 = nn.LayerNorm([lstm_units])

        self.ff1 = nn.Dense(lstm_units, ff_dim)
        self.ff2 = nn.Dense(ff_dim, lstm_units)
        
        # PERBAIKAN: Menggunakan keep_prob (1 - dropout_rate) di MindSpore
        self.dropout_ff = nn.Dropout(keep_prob=1.0 - dropout_rate)

        self.relu = ops.ReLU()
        self.fc = nn.Dense(lstm_units, output_dim)

    def construct(self, x):
        x, _ = self.lstm(x)

        attn_out = self.attn(x)
        x = self.add(x, attn_out)
        x = self.norm1(x)

        ff = self.relu(self.ff1(x))
        ff = self.dropout_ff(self.ff2(ff))
        x = self.add(x, ff)
        x = self.norm2(x)

        x = x[:, -1, :]
        return self.fc(x)