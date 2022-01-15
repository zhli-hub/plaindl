from plaindl.ops.ops import *
from plaindl.utils.functional import *


class Tensor(object):
    global_id = -1
    matmul = MatMul()
    add = Add()
    identity = Identity()
    mul = Mul()

    def __init__(self, inputs, op=None, trainable=False, bias=False, ctx=None):
        self.inputs = inputs
        self.op = op
        self.trainable = trainable
        self.value = None
        self.id = Tensor.global_id
        self.dout = 0.0
        self.bias = bias

        if ctx is None:
            self.ctx = {}
        else:
            self.ctx = ctx


        if self.op is None:
            self._type = 1  # 叶子节点
            self.op = Tensor.identity
        else:
            self._type = 0  # 操作节点

        Tensor.global_id += 1

        self.__evaluate()

        # print("eager exec: %s" % self)

    def input2values(self):
        """ 将输入统一转换成数值，因为具体的计算只能发生在数值上 """
        new_inputs = []
        for i in self.inputs:
            if isinstance(i, Tensor):
                i = i.value
            new_inputs.append(i)
        return new_inputs

    def __evaluate(self):
        if self._type is 1:
            self.value = self.inputs
            return

        inputs = self.input2values()
        self.value = self.op.forward(ctx=self.ctx, inputs=inputs)

    def size(self, i):

        return self.value.shape[i]

    @property
    def shape(self):
        return self.value.shape

    @property
    def op_type(self):
        return self.op.name()

    @property
    def is_leaf(self):
        return self._type

    def __del__(self):
        # print('Node id :%d has been deleted'% self.id)
        Tensor.global_id -= 1

    def __add__(self, other):
        return Tensor.add(self, other)

    def __matmul__(self, other):
        return Tensor.matmul(self, other)

    def __mul__(self, other):
        return Tensor.mul(self, other)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if type(self.value) is not np.ndarray:
            return "Tensor(id: %d,op: %s,value : %s)" % (self.id, self.op.name(), self.value)
        return "Tensor(id: %d,op: %s,shape: %s,value: %s)" % (self.id, self.op.name(),
                                                          tensor_shape_to_string(self.value.shape), self.value)
