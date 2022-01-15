from plaindl import Tensor
import numpy as np


class Initializer(object):
    def __call__(self, shape, bias, trainable):
        value = self._init(shape)
        return Tensor(value, trainable=trainable, bias=bias)

    def _init(self, shape):
        raise NotImplementedError


class RandomInitializer(Initializer):
    def __init__(self, ratio):
        self.ratio = ratio

    def _init(self, shape):
        return self.ratio*np.random.randn(*shape).astype('f')


class ConstantInitializer(Initializer):
    def __init__(self, _constant):
        self.constant = _constant

    def _init(self, shape):
        return np.full(shape=shape,fill_value=self.constant)


if __name__ == '__main__':
    init = RandomInitializer(0.01)
    print(init([10, 10], bias=False, trainable=True).value)
    constant = ConstantInitializer(0.0)
    print(constant([10], bias=True, trainable=True).value)
