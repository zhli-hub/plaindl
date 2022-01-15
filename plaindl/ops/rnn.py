import numpy as np
from plaindl.tensor import Tensor
from plaindl.ops.ops import Op


class TimeRNNOp(Op):

    class RNNCell(object):
        def __init__(self, Wh, Wx, b):
            self.Wh = Wh
            self.Wx = Wx
            self.b = b
            self.cache = []

        def forward(self, x, h_prev):
            t = np.dot(h_prev, self.Wh) + np.dot(x, self.Wx) + self.b
            h_next = np.tanh(t)

            self.cache = [h_next, h_prev, x]
            return h_next

        def backward(self, dh_next):
            h_next, h_prev, x = self.cache

            dt = dh_next * (1 - h_next ** 2)
            db = dt

            dh_prev = np.dot(dt, self.Wh.T)
            dWh = np.dot(h_prev.T, dt)

            dx = np.dot(dt, self.Wx.T)
            dWx = np.dot(x.T, dt)

            return dx, dWx, dh_prev, dWh, db



    @staticmethod
    def name():
        return "RNN"

    @staticmethod
    def forward(ctx, inputs):
        xs, Wx, Wh, b = inputs
        N, T, D = xs.shape
        _, H = Wx.shape
        # stateful,h = ctx['stateful'], ctx['h']
        grad = {'dWx': np.zeros_like(Wx),
                     'dWh': np.zeros_like(Wh),
                     'db': np.zeros((N, H))}
        rnn_cell = []

        hs = np.empty((N, T, H), dtype='f')

        if not ctx['stateful'] or ctx['h'] is None:
            ctx['h'] = np.zeros((N, H), dtype='f')

        for t in range(T):
            cell = TimeRNNOp.RNNCell(Wh=Wh, Wx=Wx, b=b)
            ctx['h'] = cell.forward(xs[:, t, :], ctx['h'])
            hs[:, t, :] = ctx['h']
            rnn_cell.append(cell)

        ctx['grad'] = grad
        ctx['rnn_cell'] = rnn_cell

        return hs

    @staticmethod
    def backward(ctx, inputs, dout):
        xs, Wx, Wh, b = inputs
        N, T, D = xs.shape
        _, H = Wx.shape

        dh = 0
        dxs = np.zeros_like(xs, dtype='f')
        dout = dout.reshape(N, T, -1)



        for t in reversed(range(T)):
            dt = dout[:, t, :] + dh

            dx, dWx, dh_prev, dWh, db = ctx['rnn_cell'][t].backward(dt)

            ctx['grad']['dWx'] += dWx
            ctx['grad']['dWh'] += dWh
            ctx['grad']['db'] += db

            dxs[:, t, :] = dx

            dh = dh_prev

        return [dxs, ctx['grad']['dWx'], ctx['grad']['dWh'], ctx['grad']['db']]



    def __call__(self, xs, Wx, Wh, b, stateful):
        ctx = {'stateful':stateful, 'h':None}
        return Tensor([xs, Wx, Wh, b], self,ctx=ctx)

    # def reset_state(self):
    #     self.h = None
    #
    # def set_state(self, h):
    #     self.h = h


class TimeEmbed(Op):
    @staticmethod
    def name():
        return "TimeEmbedding"

    @staticmethod
    def forward(ctx, inputs):
        idxs, W = inputs
        N, T = idxs.shape
        V, D = W.shape

        out = np.empty((N, T, D), dtype='f')

        for t in range(T):
            idx = idxs[:, t]
            out[:, t, :] = W[idx]

        return out
    @staticmethod
    def backward(ctx, inputs, dout):
        idxs, W = inputs
        N, T = idxs.shape
        # V, D = W.shape
        dW = np.zeros_like(W)

        for t in range(T):
            idx = idxs[:, t]
            for i, word_id in enumerate(idx):
                dW[word_id] += dout[i, t, :]

        return [0.0, dW]

    def __call__(self, *args):
        idxs, W = args
        return Tensor([idxs, W], self)
