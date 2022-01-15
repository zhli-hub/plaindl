import numpy as np
from plaindl.tensor import Tensor


class BaseOptimizer(object):
    def __init__(self):
        self.topo_list = None
        self.reverse_topo = None

    def dfs(self, topo_list, node):
        if node.is_leaf:
            topo_list.append(node)
            return
        for n in node.inputs:
            if isinstance(n, Tensor):
                self.dfs(topo_list, n)
        topo_list.append(node)  # 同一个节点可以添加多次，他们的梯度会累加

    def topological_sorting(self, root):
        """拓扑排序：采用DFS方式"""
        lst = []
        self.dfs(lst, root)
        # lst.append(root)
        return lst

    @staticmethod
    def zero_grad(reverse_topo):
        for n in reverse_topo:
            if n.is_leaf:
                n.dout = 0.0

    def gradient(self, loss):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class SGD(BaseOptimizer):

    def __init__(self, lr=0.01):
        super(SGD).__init__()
        self.lr = lr
        self._loss = None

    def __call__(self, loss, lr=0.01):
        self._loss = loss
        self.topo_list = self.topological_sorting(self._loss)
        self.lr = lr

    @property
    def loss(self):
        return self._loss.value


    def gradient(self, loss):
        """calculate the gradient of each node in the network"""
        self._loss = loss
        self.topo_list = self.topological_sorting(self._loss)

        self.reverse_topo = list(reversed(self.topo_list))  # 按照拓扑排序的反向开始微分
        self.reverse_topo[0].dout = np.ones_like(self._loss.value)  # 输出节点梯度是1.0

        self.zero_grad(self.reverse_topo)

        for n in self.reverse_topo:
            if n.is_leaf:
                continue
            grad = n.op.backward(n.ctx, n.input2values(), n.dout)
            # 将梯度累加到每一个输入变量的梯度上
            for i, g in zip(n.inputs, grad):
                if isinstance(i, Tensor):
                    i.dout += g

            n.dout = 0


    def update(self):
        for n in self.topo_list:

            if n.trainable:
                if n.bias:
                    n.value -= self.lr * np.sum(n.dout, axis=0)
                else:
                    n.value -= self.lr * n.dout
