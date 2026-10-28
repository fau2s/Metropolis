import math

from keras.backend import softmax
from numpy import random
from tensorflow import math, matmul, cast, float32, reshape, shape, transpose
from tensorflow.python.keras.layers import Layer, Dense


class MultiHeadAttention(Layer):
    def __init__(self, h, d_k, d_v, d_model, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.attention = DotProductAttention()
        self.heads = h
        self.d_k = d_k
        self.d_v = d_v
        self.W_q = Dense(d_k)
        self.W_k = Dense(d_k)
        self.W_v = Dense(d_k)
        self.W_o = Dense(d_model)

    def reshapeTensor(self, x, heads, flag):
        if flag:
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            x = transpose(x, perm=(0, 2, 1, 3))
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], -1))
        return x

    def call(self, queries, keys, values, mask=None):
        q_reshaped = self.reshapeTensor(self.W_q(queries), self.heads, True)
        k_reshaped = self.reshapeTensor(self.W_k(keys), self.heads, True)
        v_reshaped = self.reshapeTensor(self.W_v(values), self.heads, True)
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)

        output = self.reshapeTensor(o_reshaped, self.heads, False)

        return self.W_o(output)


class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)

    def call(self, queries, keys, values, d_k, mask=None):
        scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))

        if mask is not None:
            scores += -1e9 * mask

        weights = softmax(scores)

        return matmul(weights, values)


if __name__ == "__main__":
    h = 8
    d_k = 64
    d_v = 64
    d_model = 512
    batch_size = 64

    inputSeqLength = 5

    queries = random.random((batch_size, inputSeqLength, d_k))
    keys = random.random((batch_size, inputSeqLength, d_k))
    values = random.random((batch_size, inputSeqLength, d_v))

    multiHeadAttention = MultiHeadAttention(h, d_k, d_v, d_model)
    print(multiHeadAttention(queries, keys, values))