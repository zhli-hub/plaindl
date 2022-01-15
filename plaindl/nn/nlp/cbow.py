import plaindl as pdl
import plaindl.nn as nn
from plaindl.nn.initializer import *
from plaindl.nn.layers import *


class CBOW(nn.Module):
    def __init__(self, vocab_size, hidden_size, window_size, corpus, init_fn=RandomInitializer(0.01)):
        super(CBOW, self).__init__()
        V, H = vocab_size, hidden_size

        # w_in = init_fn([V, H], bias=False, trainable=True)
        w_out = init_fn([V, H], bias=False, trainable=True)
        self.embeding = Embedding(num_embeddings=V, word_embeddings=H, w_init_fn=init_fn)

        self.in_layers = []
        for i in range(2 * window_size):
            self.in_layers.append(self.embeding)

        self.ns_loss = NegativeSamplingLoss(num_embeddings=V, word_embeddings=H, corpus=corpus,w_init_fn=init_fn)

        self.word_vecs = self.embeding.get_weight()

    def forward(self, *input):
        input, target = input
        _h = self.in_layers[0](input[:, 0])
        for i in range(len(self.in_layers) - 1):
            _h = _h + self.in_layers[i+1](input[:, i+1])

        _h = _h * (1 / len(self.in_layers))

        loss = self.ns_loss(_h, target)
        return loss


if __name__ == '__main__':
    pass
