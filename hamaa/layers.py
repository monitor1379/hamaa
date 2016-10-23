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
import sys


from . import initializations
from . import activations
from .gates import AddGate, MulGate
from .utils.conv_utils import *

def print_size(name, var):
    t = sys.getsizeof(var) * 1.0 / (1024 * 1024)
    print "{} 's size:{}MB".format(name, t)

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
        self.trainable_params = []
        self.grads = []

        self.mid = {}
        self.config = {}

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
        self.input_shape = [None, input_dim]
        self.output_shape = [None, output_dim]

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
        self.input_shape[1] = self.input_dim

        # 初始化参数
        self.w = self.init((self.input_dim, self.output_dim))
        self.b = initializations.zeros((1, self.output_dim))
        self.trainable_params = [self.w, self.b]

    def forward(self, _input):
        self.input = _input
        self.output_shape[0] = self.input_shape[0] = self.input.shape[0]
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
        self.input_shape = [None, None]
        self.output_shape = [None, None]

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
        self.output_shape[0] = self.input_shape[0] = self.input.shape[0]
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
        self.input_shape = [None, None, None, None]
        self.output_shape = [None, None, None, None]
        self.w_shape = [None, None, None, None]

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
        # ==================================
        # 计算input_shape
        # 如果有前一层
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
                self.input_shape[1:] = self.CHW_shape
        # ==================================
        # 计算w_shape以及output_shape
        N, C, H, W = self.input_shape
        KN = self.nb_kernel
        KC = self.input_shape[1]
        KH = self.kernel_height
        KW = self.kernel_width
        CH, CW = get_conv_shape(H, W, KH, KW, self.stride)

        self.w_shape = [KN, KC, KH, KW]
        self.output_shape = [N, KN, CH, CW]

        # 初始化参数
        self.w = self.init(shape=self.w_shape)
        self.b = initializations.zeros(shape=(1, KN, 1, 1))
        self.trainable_params = [self.w, self.b]

    def forward(self, _input):
        self.input = _input
        self.output_shape[0] = self.input_shape[0] = self.input.shape[0]

        # 计算形状
        N, C, H, W = self.input_shape
        KN, KC, KH, KW = self.w_shape
        CH, CW = self.output_shape[2:]

        # 准备工作
        x = self.input
        rowing_w = self.w.reshape(KN, KH * KW * C)
        # columnize_x = np.empty((N, KH * KW * C, CH * CW), dtype=np.double)
        mul = np.empty(self.output_shape, dtype=self.input.dtype)

        # 前向计算
        for n in xrange(N):
            # 将输入变形
            x_n = np.array(x[n]).reshape(1, C, H, W)

            # 计算
            columnize_x_n = im2col_NCHW(x_n, KH, KW, self.stride)
            rowing_mul_n = MulGate.forward(rowing_w, columnize_x_n)

            # 将输出变形
            mul_n = rowing_mul_n.reshape(1, KN, CH, CW)

            # 保存中间计算结果
            mul[n] = mul_n
            # columnize_x[n] = columnize_x_n

        add = AddGate.forward(mul, self.b)
        act = self.activation.forward(add)

        # 保存中间计算变量
        # self.mid['columnize_x'] = columnize_x
        self.mid['rowing_w'] = rowing_w
        self.mid['mul'] = mul
        self.mid['add'] = add
        self.mid['act'] = act

        print '======================================'
        if not self.previous_layer:
            print '第一层：前向'
        else:
            print '第二层：前向'
        print 'input shape: ', self.input_shape
        print_size('rowing_w', rowing_w)
        # print_size('columnize_x', columnize_x)
        print_size('mul', mul)
        print_size('add', add)
        print_size('act', act)


        self.output = act
        return self.output

    def backward(self, _d_output):
        self.d_output = _d_output

        # 形状变量
        N, C, H, W = self.input_shape
        KN, KC, KH, KW = self.w_shape
        CH, CW = self.output_shape[2:]

        # 提取中间计算变量
        # columnize_x = self.mid['columnize_x']
        rowing_w = self.mid['rowing_w']
        mul = self.mid['mul']
        add = self.mid['add']
        act = self.mid['act']

        # 准备工作
        d_x = np.empty_like(self.input, dtype=self.input.dtype)
        d_w = np.zeros_like(self.w, dtype=np.double)

        print '======================================'
        if not self.previous_layer:
            print '第一层：后向'
        else:
            print '第二层：后向'
        print 'input shape: ', self.input_shape
        print_size('d_x', d_x)
        print_size('d_w', d_w)

        # 反向求导
        x = self.input
        d_act = self.d_output
        d_add = self.activation.backward(add, d_act)
        d_mul, d_b = AddGate.backward(mul, self.b, d_add)

        for n in xrange(N):
            x_n = np.array(x[n]).reshape(1, C, H, W)

            # 计算
            columnize_x_n = im2col_NCHW(x_n, KH, KW, self.stride)

            # columnize_x_n = columnize_x[n]
            d_mul_n = d_mul[n]
            d_rowing_mul_n = d_mul_n.reshape(KN, CH*CW)
            d_rowing_w, d_columnize_x_n = MulGate.backward(rowing_w, columnize_x_n, d_rowing_mul_n)
            d_x_n = col2im_NCHW(d_columnize_x_n, KH, KW, CH, CW, self.stride)
            d_x_n = d_x_n.reshape(C, H, W)

            d_x[n] = d_x_n

            d_w_n = d_rowing_w.reshape(KN, KC, KH, KW)
            d_w += d_w_n

        # 保存梯度
        self.grads = [d_w, d_b]
        self.d_input = d_x
        return self.d_input


class Flatten(Layer):
    """
    平铺层，将4维输入(NCHW)转化为2维输出(NM)。
    """

    layer_type = 'Flatten'

    def __init__(self, input_shape=None, **kwargs):
        super(Flatten, self).__init__()

        self.CHW_shape = input_shape  # (C, H, W)格式

        # 输入输出数据及其形状
        self.input = None
        self.output = None
        self.d_input = None
        self.d_output = None
        self.input_shape = [None, None, None, None]
        self.output_shape = [None, None]

        # 中间计算结果
        self.mid = {}

        # 模型基本成员
        self.trainable = True
        self.config = {}
        self.trainable_params = []
        self.grads = []
        self.previous_layer = None
        self.latter_layer = None

    def build(self):
        # 如果具有前一层
        if self.previous_layer:
            self.input_shape = self.previous_layer.output_shape
        # 如果是第一层
        else:
            if not self.CHW_shape:
                raise Exception('BuildError : Flatten为第一层时必'
                                '须在构造方法中提供input_shape参数!')
            else:
                self.input_shape[1:] = self.CHW_shape
        self.output_shape[-1] = np.product(self.input_shape[1:])

    def forward(self, _input):
        self.input = _input
        self.output_shape[0] = self.input_shape[0] = self.input.shape[0]
        self.output = self.input.reshape(self.output_shape)
        return self.output

    def backward(self, _d_output):
        self.d_output = _d_output
        self.d_input = self.d_output.reshape(self.input_shape)
        return self.d_input


class MeanPooling2D(Layer):
    """
    均值池化层
    """

    layer_type = 'MeanPooling2D'

    def __init__(self, pooling_size, input_shape=None, **kwargs):
        super(MeanPooling2D, self).__init__()

        self.pooling_size = pooling_size
        self.CHW_shape = input_shape  # (C, H, W)格式

        # 输入输出数据及其形状
        self.input = None
        self.output = None
        self.d_input = None
        self.d_output = None
        self.input_shape = [None, None, None, None]
        self.output_shape = [None, None, None, None]

        # 中间计算结果
        self.mid = {}

        # 模型基本成员
        self.trainable = True
        self.config = {}
        self.trainable_params = []
        self.grads = []
        self.previous_layer = None
        self.latter_layer = None

    def build(self):
        # ==================================
        # 计算input_shape
        # 如果具有前一层
        if self.previous_layer:
            self.input_shape = self.previous_layer.output_shape
        # 如果是第一层
        else:
            if not self.CHW_shape:
                raise Exception('BuildError : MeanPoolnig2D层为第一层'
                                '时必须在构造方法中提供input_shape参数!')
            self.input_shape[1:] = self.CHW_shape

        if self.pooling_size[0] != self.pooling_size[1]:
            raise Exception('BuildError : 目前MeanPooling2D层只支持方形采样！')
        # ==================================
        N, C, H, W = self.input_shape
        self.output_shape = [N, C, H/self.pooling_size[0], W/self.pooling_size[1]]

    def forward(self, _input):
        self.input = _input
        self.output_shape[0] = self.input_shape[0] = self.input.shape[0]

        N, C, H, W = self.input_shape
        KN, KC, KH, KW = 1, 1, self.pooling_size[0], self.pooling_size[1]
        CH, CW = self.output_shape[2:]
        stride = KH

        rowing_w = np.ones(shape=(KN, KC*KH*KW), dtype=self.input.dtype) / (KH*KW)

        columnize_x = np.empty(shape=(KH*KW, N*C*CH*CW), dtype=self.input.dtype)

        for n in xrange(N):
            for c in xrange(C):
                ocol = (n*C+c)*CH*CW
                columnize_x[0:KH*KW, ocol: ocol+CH*CW] = im2col_HW(self.input[n][c], KH, KW, stride)
        rowing_output = MulGate.forward(rowing_w, columnize_x)
        self.output = rowing_output.reshape(self.output_shape)

        self.mid['rowing_w'] = rowing_w
        self.mid['columnize_x'] = columnize_x

        return self.output

    def backward(self, _d_output):
        self.d_output = _d_output

        N, C, H, W = self.input_shape
        KN, KC, KH, KW = 1, 1, self.pooling_size[0], self.pooling_size[1]
        CH, CW = self.output_shape[2:]
        stride = KH

        rowing_w = self.mid['rowing_w']
        columnize_x = self.mid['columnize_x']
        d_input = np.empty(shape=self.input_shape, dtype=self.input.dtype)

        d_rowing_output = self.d_output.reshape(1, N*C*CH*CW)
        _, d_columnize_x = MulGate.backward(rowing_w, columnize_x, d_rowing_output)
        for n in xrange(N):
            for c in xrange(C):
                ocol = (n*C+c)*CH*CW
                d_input[n][c] = col2im_HW(d_columnize_x[0:KH*KW, ocol: ocol+CH*CW], KH, KW, CH, CW, stride)
        self.d_input = d_input
        return self.d_input



