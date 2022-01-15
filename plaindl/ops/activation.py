from plaindl.ops.ops import *
import plaindl.utils.functional as F
from plaindl.tensor import Tensor


class Sigmoid(Op):

    @staticmethod
    def name():
        return "Sigmoid"

    def __call__(self, input):
        # Tensor = input.__class__
        return Tensor([input], self)

    @staticmethod
    def forward(ctx, inputs):
        x, = inputs
        return F.sigmoid(x)

    @staticmethod
    def backward(ctx, inputs, dout):
        x, = inputs
        y = 1 / (1 + np.exp(-x))
        return [dout * y * (1.0 - y)]


class Relu(Op):
    def __init__(self):
        self.mask = None

    @staticmethod
    def name():
        return "Relu"

    @staticmethod
    def forward(ctx, inputs):
        x, = inputs
        out = x.copy()
        mask = (x <= 0)
        out[mask] = 0
        ctx['mask'] = mask
        return out

    @staticmethod
    def backward(ctx, inputs, dout):
        dout[ctx['mask']] = 0
        return [dout]

    def __call__(self, *args):
        x, = args
        return Tensor([x], self)