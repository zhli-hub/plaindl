import plaindl as pdl
from plaindl.tensor import Tensor
import plaindl.data as Data
import plaindl.nn as nn
import matplotlib.pyplot as plt

batch_size = 64
trainDataLoader = Data.DataLoader(Data.MNIST('mnist_data', train=True, one_hot=True),
                                  batch_size=batch_size,
                                  shuffle=True)

max_iters = trainDataLoader.size // batch_size
max_epoch = 40

optim = pdl.optim.SGD(lr=0.01)
loss_list = []


class mlp(nn.Module):
    def forward(self, _inputs, _labels):

        predict = self.layers(_inputs)

        return self.loss_fn(predict, _labels)

    def __init__(self):
        super(mlp, self).__init__()
        self.loss_fn = pdl.ops.SoftmaxWithCrossLoss()
        self.sigmoid = pdl.ops.Sigmoid()

        self.layers = nn.Sequential({
            "linear1": nn.Linear(100),
            "sigmoid1": self.sigmoid,
            "linear2": nn.Linear(100),
            "sigmoid2": self.sigmoid,
            "linear3": nn.Linear(10)}
        )


model = mlp()
# print(model.state_dict())
for epoch in range(max_epoch):
    iters = 0
    total_loss = 0
    loss_count = 0

    for data in trainDataLoader:
        inputs, labels = data
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

# mlp.save('.')
