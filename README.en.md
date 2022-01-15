# PlainDL

#### Introduction
Plaindl is a lightweight deep learning training framework based on numpy.
At present, it has supported the functions of automatic differentiation, various common operators (such as dense, CNN, RNN, embedding, etc.), gradient detection based on numerical differentiation, serialization and storage and loading of models.

#### Characteristics
The implementation of pure Python is clear in structure and simple in logic, which helps to familiarize with the details of the internal implementation of deep learning training. The interface is similar to python, and easy to start.



#### Installation

1.  download and install with PIP
```
pip install plaindl
```
#### Code structure
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


#### Examples

1.  Use numpy to generate a tensor
```python
import numpy as np
import plaindl as pdl

x = pdl.Tensor(np.random.randn(2,3))
```
2.  Define a simple neural network
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

#### Participation and contribution

1. Fork warehouse
2. New feat_ XXX branch
3. Submission code
4. New pull request
