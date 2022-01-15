from plaindl.nn.module import Module
from collections import OrderedDict


class Sequential(Module):
    def forward(self, *input):
        input, = input
        for i in range(len(self.layers)):
            input = self.layers[i](input)

        return input

    def __init__(self, *input):
        super(Sequential, self).__init__()
        input_len = len(input)
        input, = input
        self.layers = []
        if input_len is 1 and isinstance(input, dict):

            for name, module in input.items():
                self.add_module(name, module)
                self.layers.append(module)

        else:
            for idx, module in enumerate(input):
                self.add_module(str(idx), module)
                self.layers.append(module)

