# PlainDL

#### 介绍
PlainDL是一种基于numpy的轻量级深度学习训练框架。
目前已经支持自动微分、各种常见的算子（如dense、cnn、rnn、embedding等）、基于数值微分的梯度检测、模型的序列化保存和加载等功能。

#### 特性
使用纯python实现，结构清晰，逻辑简单，有助于熟悉深度学习训练的内部实现细节。接口与pytorch类似，上手容易。


#### 安装

使用pip下载安装
```javascript
pip install plaindl
```

#### 代码结构
```
├─example 包括了多个实际的任务例子，包括图像识别、词向量训练、语言模型等
├─plaindl
│  ├─data 数据集加载与读取
│  ├─nn
│  │  ├─cv 封装了用于计算机视觉任务的模型
│  │  ├─nlp 封装用于自然语言处理任务的模型
│  │  └─各种用于构建神经网络的组件，包括模型的基类module和各种网络层
│  ├─ops 各种算子的定义  
│  ├─optim 优化器
│  ├─utils 工具文件
```


#### 例子

1.  使用numpy生成一个tensor
```python
import numpy as np
import plaindl as pdl

x = pdl.Tensor(np.random.randn(2,3))
```
2.  定义一个简单的神经网络
```python
import plaindl as pdl
import plaindl.nn as nn

class mlp(nn.Module):
    def forward(self, inputs, labels):
        predict = self.layers(inputs)
        return self.loss_fn(predict, labels)

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
```

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request
