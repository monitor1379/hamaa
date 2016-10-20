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

import numpy as np
from abc import ABCMeta, abstractmethod

from . import initializations
from . import activations
from .gates import AddGate, MulGate
from .utils import conv_utils

class Layer(object):
    """
    神经网络中所有layer的顶层抽象类，所有子类需要实现
    前向计算以及后向求导两个方法。
    """

    __metaclass__ = ABCMeta

    layer_type = 'Layer'

    def __init__(self, **kwargs):
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
                raise Exception('BuildError : 全连接层为第一层'
                                '时必须在构造方法中提供input_dim'
                                '参数!')
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
        mul = MulGate.forward(self.input, self.w)
        add = AddGate.forward(mul, self.b)
        self.mid['mul'] = mul
        self.output = add
        return self.output

    def backward(self, _d_output):
        self.d_output = _d_output
        d_add = self.d_output
        d_mul, d_b = AddGate.backward(self.mid['mul'], self.b, d_add)
        self.d_input, d_w = MulGate.backward(self.input, self.w, d_mul)
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
        self.d_input = self.activation.backward(self.input,
                                                self.d_output)
        return self.d_input


class Convolution2D(Layer):
    """
    卷积层。
    """

    layer_type = 'Convolution2D'

    def __init__(self, nb_kernel, kernel_height, kernel_width, input_shape=None,
                 init='glorot_normal', activation='linear', stride=1, **kwargs):
        super(Convolution2D, self).__init__()

        # 构造方法传入参数
        self.nb_kernel = nb_kernel
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.CHW_shape = input_shape
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.stride = stride

        # 输入输出数据及其形状
        self.input = None
        self.output = None
        self.d_input = None
        self.d_output = None

        self.N = None
        self.C = None
        self.H = None
        self.W = None

        self.KN = nb_kernel
        self.KC = self.C
        self.KH = kernel_height
        self.KW = kernel_width

        self.CH = None
        self.CW = None

        self.OH = None
        self.OW = None

        self.input_shape = (self.N, self.C, self.H, self.W)
        self.kernel_shape = (self.KN, self.KC, self.KH, self.KW)
        self.output_shape = (self.N, self.KN, self.CH, self.CW)

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
        # 如果不是第一层
        if self.previous_layer:
            self.input_shape = self.previous_layer.output_shape
        # 如果是第一层
        else:
            # 如果没有指明输入数据形状
            if not self.CHW_shape:
                # 报错
                raise Exception('BuildError : 卷积层为第一层时必'
                                '须在构造方法中提供input_shape参数!')
            # 如果指明了输入数据形状
            else:
                self.input_shape = (None, self.CHW_shape[0],
                                    self.CHW_shape[1], self.CHW_shape[2])
        self.kernel_shape = (self.kernel_shape[0], self.input_shape[1],
                             self.kernel_shape[2], self.kernel_shape[3])

        CH, CW = conv_utils.get_conv_shape(self.input_shape[2],
                                           self.input_shape[3])
        self.output_shape = (None, self.kernel_shape[0], )
        # 初始化参数
        self.w = self.init(shape=self.kernel_shape)
        self.b = initializations.zeros(shape=(self.kernel_shape[0], 1, 1, 1))
        self.trainable_params = [self.w, self.b]

        # # 输入数据x的形状
        # self.mid['N'] = self.input_shape[0]
        # self.mid['C'] = self.input_shape[1]
        # self.mid['H'] = self.input_shape[2]
        # self.mid['W'] = self.input_shape[3]
        #
        # # 卷积核w的形状
        # self.mid['KN'] = self.kernel_shape[0]
        # self.mid['KC'] = self.kernel_shape[1]
        # self.mid['KH'] = self.kernel_shape[2]
        # self.mid['KW'] = self.kernel_shape[3]
        #
        # # 卷积结果conv_x的形状
        # self.mid['CH'] = (self.mid['H'] - self.mid['KH']) / self.stride + 1
        # self.mid['CW'] = (self.mid['W'] - self.mid['KW']) / self.stride + 1
        #
        # # im2col的output结果columnize_x的形状
        # self.mid['OH'] = -1
        # self.mid['OW'] = -1


    def forward(self, _input):
        self.input = _input
        self.mid['columnize_x'] = conv_utils.im2col_NCHW(self.input,
                                                         self.kernel_shape[2],
                                                         self.kernel_shape[3],
                                                         self.stride)

        self.mid['rowing_w'] = self.w.reshape(self.kernel_shape[0],
                                              np.product(self.kernel_shape[1:]))

        self.mid['mul'] = MulGate.forward(self.mid['columnize_x'],
                                          self.mid['rowing_w'])

        self.mid['add'] = None
        return None

    def backward(self, _d_output):
        self.d_output = _d_output
        d_add = self.d_output
        d_mul, d_b = AddGate.backward(self.mid['mul'], self.b, d_add)
        self.d_input, d_w = MulGate.backward(self.input, self.w, d_mul)
        self.grads = [d_w, d_b]
        return self.d_input
