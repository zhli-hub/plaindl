from plaindl.nn.initializer import *
from plaindl.ops.loss import *
from plaindl.utils.sampler import *
from plaindl.nn.module import Module
from plaindl.ops.rnn import *
from plaindl.ops.conv import *


class Linear(Module):
    def name(self):
        return "LinearLayer"

    def forward(self, *inputs):
        x, = inputs

        in_features = x.value.reshape((x.value.shape[0], -1)).shape[1]
        if not self.init:
            self._init_parameter(in_features)
            self.init = True

        if self.bias:
            return self.affine(x, self.params['w']) + self.params['b']
            # x @ self.params['w'] + self.params['b']
        else:
            return self.affine(x, self.params['w'])
            # x @ self.params['w']

    def __init__(self, out_features, bias=True, w_init_fn=RandomInitializer(0.1), b_init_fn=ConstantInitializer(0.0)):
        """

        :param out_features: 隐藏层神经元的数量
        :param bias:是否添加偏置
        :param w_init:参数w的初始化方法，默认为符合正态分布的随机初始化
        :param b_init:参数b的初始化方法，默认为0
        """
        super(Linear, self).__init__()
        self.in_features = None
        self.out_features = out_features
        self.bias = bias
        self.params = {"w": None, "b": None}
        self.shape = {'w': [None, out_features], 'b': [out_features]}
        self.init_fn = {'w': w_init_fn, 'b': b_init_fn}

        self.affine = Affine()
        self.init = False

        self.set_parameters_dict(self.params)

    def _init_parameter(self, in_features):
        self.shape['w'][0] = in_features
        self.in_features = in_features
        self.params['w'] = self.init_fn['w'](self.shape['w'], bias=False, trainable=True)
        if self.bias:
            self.params['b'] = self.init_fn['b'](self.shape['b'], bias=True, trainable=True)

        self.set_parameters_dict(self.params)


class Embedding(Module):

    def __init__(self, num_embeddings, word_embeddings, w_init_fn=RandomInitializer(0.1), is_time=False):
        super(Embedding, self).__init__()

        self.params = {'w': w_init_fn((num_embeddings, word_embeddings), bias=False, trainable=True)}
        self.is_time = is_time

    def name(self):
        return "EmbeddingLayer"

    def get_weight(self):
        return self.params['w'].value

    def forward(self, *inputs):
        _idx, = inputs
        # if not isinstance(_idx, Tensor):
        #     _idx = Tensor(_idx)
        if not self.is_time:
            embed = Embed()
        else:
            from plaindl.ops.rnn import TimeEmbed
            embed = TimeEmbed()
        return embed(_idx, self.params['w'])


class EmbeddingDot(Module):

    def __init__(self, num_embeddings, word_embedding, w_init_fn=RandomInitializer(0.1)):
        super(EmbeddingDot, self).__init__()

        # self.params = {'w': w_init_fn()}
        self.embed = Embedding(num_embeddings, word_embedding, w_init_fn=w_init_fn)
        self.target_w = None

    def name(self):
        pass

    def forward(self, *inputs):
        _h, _idx = inputs
        self.target_w = self.embed(_idx)
        # print(self.target_w.value)
        dot = Dot()
        return dot(_h, self.target_w)
        # assert type(_idx) == Tensor


class NegativeSamplingLoss(Module):

    def __init__(self, num_embeddings, word_embeddings, corpus, power=0.75, sample_size=2,
                 w_init_fn=RandomInitializer(0.1)):
        """
        :param weight: embedding层的初始权重
        :param corpus: 语料库，以单词ID列表的形式提供
        :param power: 用于计算概率分布
        :param sample_size: 负采样的数量
        """
        super(NegativeSamplingLoss, self).__init__()
        # assert isinstance(weight, Tensor)
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithCrossLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(num_embeddings,
                                              word_embeddings,
                                              w_init_fn=w_init_fn) for _ in range(sample_size + 1)]

    def forward(self, *inputs):
        _h, _target = inputs
        assert _target.ndim == 1
        batch_size = _target.size

        negative_sample = self.sampler.get_negative_sample(_target)

        predict = self.embed_dot_layers[0](_h, _target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0](predict, correct_label)

        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            predict = self.embed_dot_layers[i + 1](_h, negative_target)
            loss = loss + self.loss_layers[i + 1](predict, negative_label)

        return loss

    def name(self):
        return "NegativeSamplingLoss"


class RNN(Module):
    def forward(self, *input):
        xs, = input  # xs的维度为[N,T,D]
        if xs.value.ndim == 2:
            xs.value = xs.value.reshape(xs.size, 1)  # 对应xs的词向量维度是1的情况
        if not self.init:
            self._init_parameter(xs.value.shape[-1])
            self.init = True

        hs = self.time_rnn_op(xs, self.params["Wx"], self.params["Wh"], self.params["b"], stateful=True)

        return hs

    def __init__(self, hidden_size, wx_init_fn=RandomInitializer(0.1),
                 wh_init_fn=RandomInitializer(0.1),
                 b_init_fn=ConstantInitializer(0.0)):
        super(RNN, self).__init__()

        self.time_rnn_op = TimeRNNOp()
        self.shape = {"Wx": [None, hidden_size], "Wh": [hidden_size, hidden_size], "b": [hidden_size]}
        self.init_fn = {"Wx": wx_init_fn, "Wh": wh_init_fn, "b": b_init_fn}
        self.params = {"Wx": None, "Wh": None, "b": None}

        self.init = False

    def _init_parameter(self, in_features):
        self.shape['Wx'][0] = in_features
        self.in_features = in_features

        self.params['Wx'] = self.init_fn['Wx'](self.shape['Wx'], bias=False, trainable=True)
        self.params['Wh'] = self.init_fn['Wh'](self.shape['Wh'], bias=False, trainable=True)
        self.params['b'] = self.init_fn['b'](self.shape['b'], bias=True, trainable=True)

    # def reset_state(self):
    #     self.time_rnn_op.reset_state()


class Conv2d(Module):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0,
                 w_init_fn=RandomInitializer(0.1),
                 bias=True,
                 b_init_fn=ConstantInitializer(0.0)):
        super(Conv2d, self).__init__()
        self.in_channels = None
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.params = {'w': None, 'b': None}
        self.shape = {'w': [out_channels, None, kernel_size, kernel_size], 'b': [out_channels]}
        self.init_fn = {'w': w_init_fn, 'b': b_init_fn}

        self.conv = Conv2dOp()

        self.init = False

    def forward(self, *input):
        x, = input
        x = x.value

        if x.ndim == 3:
            x = x[:, np.newaxis, :, :]

        N, self.in_channels, H, W = x.shape

        if not self.init:
            self.shape['w'][1] = self.in_channels
            self.params['w'] = self.init_fn['w'](self.shape['w'], bias=False, trainable=True)
            self.params['b'] = self.init_fn['b'](self.shape['b'], bias=True, trainable=True)

        if self.bias:
            out = self.conv(x, self.params['w'], self.params['b'], self.in_channels,
                            self.out_channels, self.kernel_size, self.stride, self.padding)
        else:
            out = self.conv(x, self.params['w'], 0, self.in_channels, self.out_channels,
                            self.kernel_size, self.stride, self.padding)

        # out_h, out_w = out.shape[2], out.shape[3]
        # if self.bias:
        #     out.value = out.value.reshape((N, self.out_channels, -1))
        #     out = out + self.params['b']
        #     out.value = out.value.reshape((N, self.out_channels, out_h, out_w))

        return out


class MaxPooling(Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(MaxPooling, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.max_pooling = MaxPoolingOp()

    def forward(self, *input):
        x, = input
        out = self.max_pooling(x, self.kernel_size, self.stride, self.padding)
        return out
