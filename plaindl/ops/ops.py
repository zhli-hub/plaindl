import numpy as np


class Op(object):

    @staticmethod
    def name():
        raise NotImplementedError

    @staticmethod
    def forward(ctx, inputs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, inputs, dout):
        raise NotImplementedError

    def __call__(self, *args):
        raise NotImplementedError

    # def apply(self, *inputs):
    #     self.__call__(*inputs)



class MatMul(Op):

    @staticmethod
    def name():
        return "MatMul"

    @staticmethod
    def forward(ctx, inputs):
        x, W = inputs
        out = np.dot(x, W)
        return out

    def __call__(self, *args):
        x, W = args
        Tensor = W.__class__

        return Tensor([x, W], self)

    @staticmethod
    def backward(ctx, inputs, dout):
        x, W = inputs
        dx = np.dot(dout, W.T)

        dW = np.dot(x.T, dout)
        return [dx, dW]


class Mul(Op):
    """乘法运算"""

    @staticmethod
    def name():
        return "Mul"

    @staticmethod
    def forward(ctx, inputs):
        return inputs[0] * inputs[1]

    @staticmethod
    def backward(ctx, inputs, dout):
        a, b = inputs
        da = dout * b
        db = dout * a
        return [da, db]

    def __call__(self, *args):
        a, b = args
        Tensor = a.__class__
        return Tensor([a, b], self)


class Add(Op):
    """加法运算"""

    @staticmethod
    def name():
        return "Add"

    def __call__(self, *args):
        a, b = args
        Tensor = b.__class__
        return Tensor([a, b], self)

    @staticmethod
    def forward(ctx, inputs):
        return inputs[0] + inputs[1]

    @staticmethod
    def backward(ctx, inputs, dout):
        return [dout, dout]  # gradient of a and b


class Identity(Op):
    """输入输出一样"""

    @staticmethod
    def name():
        return "Identity"

    def __call__(self, a, ):
        return

    @staticmethod
    def forward(ctx, inputs):
        return inputs[0]

    @staticmethod
    def backward(ctx, inputs, dout):
        return [dout]


class Embed(Op):
    @staticmethod
    def name():
        return "Embedding"

    @staticmethod
    def forward(ctx, inputs):
        idx, W = inputs

        return W[idx]

        # res = np.zeros([idx.shape[0], idx.shape[1], W.shape[1]])
        # # print(res.shape)
        #
        # for i in range(idx.shape[0]):
        #     res[i] = W[idx[i]]
        # return res
    @staticmethod
    def backward(ctx, inputs, dout):
        idx, W = inputs
        dW = np.zeros_like(W)

        for i, word_id in enumerate(idx):
            dW[word_id] += dout[i]

        return [0.0, dW]

    def __call__(self, *args):
        idx, W = args
        Tensor = W.__class__
        return Tensor([idx, W], self)




class Dot(Op):
    """
    计算两个向量或矩阵的内积,输入的向量形式必须为(1,n)
    """
    @staticmethod
    def name():
        return "Dot"

    @staticmethod
    def forward(ctx, inputs):
        a, b = inputs
        assert a.ndim == b.ndim == 2

        return np.sum(a * b, axis=1)

    @staticmethod
    def backward(ctx, inputs, dout):
        a, b = inputs
        dout = dout.reshape(dout.shape[0], 1)
        da = dout * b
        db = dout * a
        return [da, db]

    def __call__(self, *args):
        a, b = args
        Tensor = b.__class__
        return Tensor([a, b], self)



class Affine(Op):

    @staticmethod
    def name():
        return "Affine"

    @staticmethod
    def forward(ctx, inputs):
        x, W = inputs
        ctx['origin_x_shape'] = x.shape
        x = x.reshape((x.shape[0], -1))
        out = np.dot(x, W)

        return out

    def __call__(self, *args):
        x, W = args
        Tensor = W.__class__
        return Tensor([x, W], self)

    @staticmethod
    def backward(ctx, inputs, dout):
        x, W = inputs
        x = x.reshape((x.shape[0], -1))

        dx = np.dot(dout, W.T)
        dx = dx.reshape(ctx['origin_x_shape'])
        dW = np.dot(x.T, dout)
        return [dx, dW]


class LSTM(Op):

    @staticmethod
    def name():
        pass

    @staticmethod
    def forward(ctx, inputs):
        pass

    @staticmethod
    def backward(ctx, inputs, dout):
        pass

    def __call__(self, *args):
        pass