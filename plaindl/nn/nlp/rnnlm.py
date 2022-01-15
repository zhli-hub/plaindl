import plaindl.nn as nn
from plaindl.nn.layers import *
from plaindl import Tensor


class RnnLM(nn.Module):
    def forward(self, *input):
        xs, ts = input
        N, T = xs.shape

        emb = self.embedding(xs)
        rnn = self.rnn(emb)

        H = rnn.value.shape[2]

        rnn.value = rnn.value.reshape(-1, H)
        predict = self.linear(rnn)
        predict.value = predict.value.reshape(N, T, -1)

        loss = self.loss(predict, ts)

        return loss


    def __init__(self, vocab_size, vec_size, hidden_size):
        super(RnnLM, self).__init__()
        V, D, H = vocab_size, vec_size, hidden_size

        self.embedding = Embedding(num_embeddings=V, word_embeddings=D, is_time=True)
        self.rnn = RNN(H)
        self.linear = Linear(V)

        self.loss = TimeSoftmaxWithCrossLoss()



