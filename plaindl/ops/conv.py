import numpy as np
import plaindl.utils.functional as F
from plaindl.tensor import Tensor
from plaindl.ops.ops import Op
from plaindl.optim.optimizer import SGD


class Conv2dOp(Op):
    # def __init__(self):
    #     self.in_channels = None
    #     self.out_channels = None
    #     self.stride = None
    #     self.padding = None
    #     self.kernel_size = None
    #     self.cache = None
    #
    #     self.bias = False

    @staticmethod
    def name():
        return "conv2d"

    @staticmethod
    def forward(ctx, inputs):
        bias = ctx['bias']
        if bias:
            x, w, b = inputs
        else:
            x, w = inputs
        N, C, H, W = x.shape

        out_h = int((H + 2 * ctx['padding'] - ctx['kernel_size']) // ctx['stride']) + 1
        out_w = int((W + 2 * ctx['padding'] - ctx['kernel_size']) // ctx['stride']) + 1

        col = F.im2col(x, ctx['kernel_size'], ctx['kernel_size'], ctx['stride'], ctx['padding'])
        col_w = w.reshape((ctx['out_channels'], -1)).T

        cache = [col, col_w]
        ctx['cache'] = cache

        out = np.dot(col, col_w)
        if bias:
            out = out + b
        out = out.reshape((N, out_h, out_w, -1)).transpose(0, 3, 1, 2)

        return out

    @staticmethod
    def backward(ctx, inputs, dout):
        cache = ctx['cache']
        bias = ctx['bias']
        if bias:
            x, w, b = inputs
        else:
            x, w = inputs
        N, C, H, W = x.shape

        dout = dout.transpose(0, 2, 3, 1)

        dout = dout.reshape((-1, ctx['out_channels']))
        db = dout
        dcol_w = np.dot(cache[0].T, dout)
        dcol = np.dot(dout, cache[1].T)

        dx = F.col2im(dcol, x.shape, ctx['kernel_size'], ctx['kernel_size'], ctx['stride'], ctx['padding'])
        dw = dcol_w.transpose(1, 0).reshape(ctx['out_channels'], C, ctx['kernel_size'], ctx['kernel_size'])

        if bias:
            return [dx, dw, db]
        return [dx, dw]

    def __call__(self, *args):
        x, w, b, in_channels, out_channels, kernel_size, stride, padding = args
        ctx = {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': kernel_size,
               'stride': stride, 'padding': padding}
        if isinstance(b, Tensor):
            ctx['bias'] = True
            return Tensor([x, w, b], self, ctx=ctx)
        ctx['bias'] = False
        return Tensor([x, w], self, ctx=ctx)


class MaxPoolingOp(Op):
    # def __init__(self):
    #     self.kernel_size = None
    #     self.padding = None
    #     self.stride = None
    #     self.out_h = None
    #     self.out_w = None
    #     self.argmax = None

    @staticmethod
    def name():
        return "MaxPooling"

    @staticmethod
    def forward(ctx, inputs):
        x, = inputs
        N, C, H, W = x.shape

        kernel_size, stride, padding = ctx['kernel_size'], ctx['stride'], ctx['padding']

        out_h = int((H + 2 * padding - kernel_size) // stride) + 1
        out_w = int((W + 2 * padding - kernel_size) // stride) + 1

        col = F.im2col(x, kernel_size, kernel_size, stride, padding)
        col = col.reshape((N * out_h * out_w * C, -1))

        out = np.max(col, axis=1)
        argmax = np.argmax(col, axis=1)

        out = out.reshape((N, out_h, out_w, C)).transpose(0, 3, 1, 2)

        cache = [argmax, out_h, out_w]

        ctx['cache'] = cache

        return out

    @staticmethod
    def backward(ctx, inputs, dout):
        x, = inputs
        N, C, H, W = x.shape

        argmax, out_h, out_w = ctx['cache']
        kernel_size, stride, padding = ctx['kernel_size'], ctx['stride'], ctx['padding']

        dout = dout.transpose(0, 2, 3, 1)

        dmax = np.zeros((N * out_h * out_w * C, kernel_size * kernel_size))
        dmax[np.arange(argmax.size), argmax.flatten()] = dout.flatten()

        dcol = dmax.reshape((N * out_h * out_w, -1))
        dx = F.col2im(dcol, x.shape, kernel_size, kernel_size, stride, padding)

        return [dx]

    def __call__(self, *args):
        x, kernel_size, stride, padding = args
        ctx = {'kernel_size': kernel_size, 'stride': stride, 'padding': padding}

        return Tensor([x], self, ctx=ctx)


if __name__ == '__main__':
    input = np.random.randn(1, 3, 6, 6)
    w = np.random.randn(10, 3, 3, 3)
    w = Tensor(w, trainable=True)
    conv = Conv2dOp()
    pool = MaxPoolingOp()
    conv_out = conv(input, w, 3, 10, 3, 1, 0)
    print(conv_out.value.shape)
    out = pool(conv_out, 2, 2, 0)
    print(out.shape)

    dout = np.random.randn(1, 10, 3, 3)
    optim = SGD(0.01)
    optim.gradient(out)
