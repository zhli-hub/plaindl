import plaindl as pdl
import numpy as np
import plaindl.nn as nn
import matplotlib.pyplot as plt
sigmoid = pdl.ops.Sigmoid()


def load_data(seed=1984):
    np.random.seed(seed)
    N = 100  # 各类的样本数
    DIM = 2  # 数据的元素个数
    CLS_NUM = 3  # 类别数

    x = np.zeros((N*CLS_NUM, DIM))
    t = np.zeros((N*CLS_NUM, CLS_NUM), dtype=np.int)

    for j in range(CLS_NUM):
        for i in range(N):#N*j, N*(j+1)):
            rate = i / N
            radius = 1.0*rate
            theta = j*4.0 + 4.0*rate + np.random.randn()*0.2

            ix = N*j + i
            x[ix] = np.array([radius*np.sin(theta),
                              radius*np.cos(theta)]).flatten()
            t[ix, j] = 1

    return x, t


class mlp(nn.Module):
    def forward(self, input, labels):
        h1 = input @ self.params['w1'] + self.params['b1']
        z1 = sigmoid(h1)

        h2 = z1 @ self.params['w2'] + self.params['b2']

        predict = pdl.ops.SoftmaxWithCrossLoss()
        loss = predict(h2, labels)
        return loss

    def predict(self, x):
        h1 = x @ self.w1 + self.b1
        z1 = sigmoid(h1)

        h2 = z1 @ self.w2 + self.b2
        return pdl.utils.softmax(h2.value)

    def __init__(self):
        super(mlp, self).__init__()
        self.x = None
        self.grad = []

        self.w1 = pdl.Tensor(np.random.randn(2, 10)*0.01, trainable=True)
        self.b1 = pdl.Tensor(np.zeros(10), trainable=True, bias=True)

        self.w2 = pdl.Tensor(np.random.randn(10, 3)*0.01, trainable=True)
        self.b2 = pdl.Tensor(np.zeros(3), trainable=True, bias=True)

        self.params = {"w1":self.w1,"b1":self.b1,"w2":self.w2,"b2":self.b2}
        self.set_parameters_dict(self.params)



model = mlp()
batch_size = 30
x, t = load_data()
optim = pdl.optim.SGD(lr=1)
data_size = len(x)
max_iters = data_size // batch_size
total_loss=0
loss_count = 0
loss_list = []




for epoch in range(0):
    # np.random.seed(1024)
    idx = np.random.permutation(data_size)

    x = x[idx]
    t = t[idx]

    for iters in range(max_iters):
        batch_x = x[iters*batch_size:(iters+1)*batch_size]
        batch_t = t[iters*batch_size:(iters+1)*batch_size]

        input = pdl.Tensor(batch_x)
        # 计算梯度，更新参数
        loss = model(input, batch_t)

        optim.gradient(loss)
        optim.update()

        total_loss += loss.value
        loss_count += 1

        # 定期输出学习过程
        if (iters+1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print('| epoch %d |  iter %d / %d | loss %.2f'
                  % (epoch + 1, iters + 1, max_iters, avg_loss))
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0

# print(model.state_dict())
# model.save_state_dict("spiral_mlp_model.pkl")


model.load_state_dict("spiral_mlp_model.pkl")
# 绘制学习结果
# plt.plot(np.arange(len(loss_list)), loss_list, label='train')
# plt.xlabel('iterations (x10)')
# plt.ylabel('loss')
# plt.show()

# 绘制决策边界
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
score = model.predict(pdl.Tensor(X))
predict_cls = np.argmax(score, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)
plt.axis('off')

# 绘制数据点
x, t = load_data()
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()