# encoding: utf-8
"""
@author: monitor1379
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: layers.py
@time: 2016/9/20 9:36


"""

from abc import ABCMeta, abstractmethod

from . import initializations
from . import activations
from .gates import AddGate, MulGate


class Layer(object):
    """
    神经网络中所有layer的顶层抽象类，所有子类需要实现
    前向计算以及后向求导两个方法。
    """

    __metaclass__ = ABCMeta

    layer_type = 'Layer'

    def __init__(self):
        # 输入输出数据及其形状
        self.input = None
        self.output = None
        self.d_input = None
        self.d_output = None
        self.input_shape = None
        self.output_shape = None

        # 模型基本成员
        self.trainable = False
        self.mid = {}
        self.config = {}
        self.trainable_params = []
        self.grads = []

        self.previous_layer = None
        self.latter_layer = None

    @abstractmethod
    def build(self):
        """
        因为某些层的某些属性在构造方法中还不能得到，
        需要通过前一层的某些属性才能求得，所以通过
        提供build方法对模型进行二次构建。
        """
        pass

    def summary(self):
        s = 'type:%15s,\t' % self.layer_type
        s += 'in:{},\t out:{}'.format(self.input_shape, self.output_shape)
        return s

    @abstractmethod
    def forward(self, _input):
        """前向计算过程"""
        pass

    @abstractmethod
    def backward(self, _output):
        """后向传播过程。该过程仅仅计算本层传播误差，不负责更新模型参数"""
        pass


class Dense(Layer):
    """
    全连接层。
    """

    layer_type = 'Dense'

    def __init__(self, output_dim, input_dim=None, init='uniform', **kwargs):
        super(Dense, self).__init__()

        # 构造方法传入参数
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init = initializations.get(init)

        # 输入输出数据及其形状
        self.input = None
        self.output = None
        self.d_input = None
        self.d_output = None
        self.input_shape = (None, input_dim)
        self.output_shape = (None, output_dim)

        # 中间计算结果
        self.mid = {}

        # 模型参数
        self.w = None
        self.b = None

        # 模型基本成员
        self.trainable = True
        self.config = {}
        self.trainable_params = []
        self.grads = []
        self.previous_layer = None
        self.latter_layer = None

    def build(self):
        # 如果input_dim为None，则通过前一层获得
        if not self.input_dim:
            # 如果本层是网络的第一层
            if not self.previous_layer:
                raise Exception('BuildError : 全连接层为第一层时必须在构造方法中'
                                '提供input_dim参数!')
            else:
                self.input_dim = self.previous_layer.output_shape[-1]

        # 设置层的输入输出形状
        self.input_shape = (None, self.input_dim)

        # 初始化参数
        self.w = self.init((self.input_dim, self.output_dim))
        self.b = initializations.zeros((1, self.output_dim))
        self.trainable_params = [self.w, self.b]

    def forward(self, _input):
        self.input = _input
        mul = MulGate.forward(self.w, self.input)
        add = AddGate.forward(mul, self.b)
        self.mid['mul'] = mul
        self.output = add
        return self.output

    def backward(self, _d_output):
        self.d_output = _d_output
        d_add = self.d_output
        d_mul, d_b = AddGate.backward(self.mid['mul'], self.b, d_add)
        d_w, self.d_input = MulGate.backward(self.w, self.input, d_mul)
        self.grads = [d_w, d_b]
        return self.d_input


class Activation(Layer):
    """
    激活层。
    """

    layer_type = 'Activation'

    def __init__(self, activation, **kwargs):
        super(Activation, self).__init__()

        # 构造方法传入参数
        self.activation = activations.get(activation)

        # 输入输出数据及其形状
        self.input = None
        self.output = None
        self.d_input = None
        self.d_output = None
        self.input_shape = (None, None)
        self.output_shape = (None, None)

        # 模型基本成员
        self.trainable = False
        self.config = {}
        self.trainable_params = []
        self.grads = []
        self.previous_layer = None
        self.latter_layer = None

    def build(self):
        if not self.previous_layer:
            raise Exception('BuildError: 暂时不支持激活层为第一层!')
        self.input_shape = self.previous_layer.output_shape
        self.output_shape = self.input_shape

    def forward(self, _input):
        self.input = _input
        self.output = self.activation.forward(self.input)
        return self.output

    def backward(self, _d_output):
        self.d_output = _d_output
        self.d_input = self.activation.backward(self.input, self.d_output)
        return self.d_input

