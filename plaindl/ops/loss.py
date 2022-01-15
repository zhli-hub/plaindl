from plaindl.ops.ops import *
import plaindl.utils.functional as F


class BaseLoss(Op):

    @staticmethod
    def forward(ctx, inputs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, inputs, dout):
        raise NotImplementedError

    @staticmethod
    def name():
        raise NotImplementedError

    def __call__(self, predict, label):

        Tensor = predict.__class__
        return Tensor([predict, label], self)


class SoftmaxWithCrossLoss(BaseLoss):
    @staticmethod
    def name():
        return "SoftmaxWithCrossLoss"

    @staticmethod
    def forward(ctx, inputs):
        x, label = inputs
        act = F.softmax(x)
        loss = F.cross_entropy_error(act, label)

        return loss

    @staticmethod
    def backward(ctx, inputs, dout=1):
        x, label = inputs

        if label.size == x.size:
            label = label.argmax(axis=1)

        act = F.softmax(x)
        batch_size = label.shape[0]

        dx = act.copy()
        dx[np.arange(batch_size), label] -= 1
        dx *= dout
        dx = dx / batch_size

        return [dx]


class SigmoidWithCrossLoss(BaseLoss):

    def __init__(self):
        # self.label = None
        self.act = None

    @staticmethod
    def name():
        return "SigmoidWithCrossLoss"

    @staticmethod
    def forward(ctx, inputs):
        x, label = inputs
        act = F.sigmoid(x)
        loss = F.cross_entropy_error(np.c_[1 - act, act], label)

        ctx['act'] = act
        return loss

    @staticmethod
    def backward(ctx, inputs, dout):
        x, label = inputs
        batch_size = label.shape[0]

        dx = (ctx['act'] - label) * dout / batch_size

        return [dx]


class TimeSoftmaxWithCrossLoss(BaseLoss):

    @staticmethod
    def forward(ctx, inputs):
        xs, label = inputs

        N, T, V = xs.shape
        xs = xs.reshape(N * T, -1)

        if label.ndim == 3:
            label = label.argmax(axis=2)
        ts = label.reshape(N * T)

        ys = F.softmax(xs)
        # loss = F.cross_entropy_error(ys, ts)

        loss = -np.sum(np.log(ys[np.arange(N * T), ts])) / ys.shape[0]

        cache = (ts, ys, (N, T, V))
        ctx['cache'] = cache

        return loss

    @staticmethod
    def backward(ctx, inputs, dout):
        ts, ys, (N, T, V) = ctx['cache']

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= N*T

        # dx = dx.reshape((N, T, V))

        return [dx]

    @staticmethod
    def name():
        return "TimeSoftmaxWithCrossLoss"
