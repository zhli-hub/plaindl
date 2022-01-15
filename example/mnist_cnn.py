import plaindl as pdl
import plaindl.nn as nn
from plaindl.ops.activation import *
import plaindl.data as Data
import matplotlib.pyplot as plt

batch_size = 64
trainDataLoader = Data.DataLoader(Data.MNIST('mnist_data', train=True, one_hot=True),
                                  batch_size=batch_size,
                                  shuffle=True)

max_iters = trainDataLoader.size // batch_size
max_epoch = 30

optim = pdl.optim.SGD(lr=0.01)
loss_list = []


class SimpleConvnet(nn.Module):
    def __init__(self):
        super(SimpleConvnet, self).__init__()

        self.layers = [
            nn.Conv2d(30, 3, 1, 1),
            Relu(),
            nn.MaxPooling(2, 2),

            nn.Conv2d(30, 3, 1, 1),
            Relu(),
            nn.MaxPooling(2, 2),

            nn.Linear(100),
            Sigmoid(),

            nn.Linear(10)
        ]

        self.loss = nn.SoftmaxWithCrossLoss()

    def forward(self, *input):
        x, label = input

        for i in range(len(self.layers)):
            x = self.layers[i](x)

        loss = self.loss(x, label)

        return loss


model = SimpleConvnet()

for epoch in range(max_epoch):
    iters = 0
    total_loss = 0
    loss_count = 0

    for data in trainDataLoader:
        inputs, labels = data
        inputs = inputs.reshape((-1, 28, 28))
        x = Tensor(inputs)
        loss = model(x, labels)
        optim.gradient(loss)
        optim.update()

        total_loss += loss.value
        loss_count += 1

        if (iters + 1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print('| epoch %d |  iter %d / %d | loss %.2f'
                  % (epoch + 1, iters + 1, max_iters, avg_loss))
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0
        iters += 1
# 绘制学习结果

import numpy as np
plt.plot(np.arange(len(loss_list)), loss_list, label='train')
plt.xlabel('iterations (x10)')
plt.ylabel('loss')
plt.show()
