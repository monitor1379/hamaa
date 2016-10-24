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
import sys


from . import initializations
from . import activations
from .gates import AddGate, MulGate
from .utils.conv_utils import *
from .utils.time_utils import tic, toc


def print_size(name, var):
    t = sys.getsizeof(var) * 1.0 / (1024 * 1024)
    print "{} 's size:{}MB".format(name, t)


class Layer(object):
    """
    神经网络中所有layer的顶层抽象类，所有子类需要实现
    前向计算以及后向求导两个方法。
    """

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
        self.trainable_params = []
        self.grads = []

        self.previous_layer = None
        self.latter_layer = None

        self.mode = "test"
        self.mid = {}
        self.config = {}

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

    def forward(self, _input):
        """前向计算方法，根据当前模式来选择调用forward_train或者forward_test"""
        # 更新当前输入输出的数据个数
        self.output_shape[0] = self.input_shape[0] = _input.shape[0]

        if self.mode == 'train':
            return self.forward_train(_input)
        elif self.mode == 'test':
            return self.forward_test(_input)
        else:
            raise Exception('Error: Unknown forward mode: {} !'.format(self.mode))

    def forward_train(self, _input):
        pass

    def forward_test(self, _input):
        pass

    def backward(self, _output):
        """后向传播过程。该过程仅仅计算本层传播误差，不负责更新模型参数。"""
        pass


class Dense(Layer):
    """
    全连接层。
    """

    layer_type = 'Dense'

    def __init__(self, output_dim, input_dim=None, init='uniform', **kwargs):
        super(Dense, self).__init__()
        # Layer.__init__(self)
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
        self.trainable_params = []
        self.grads = []

        self.previous_layer = None
        self.latter_layer = None

        self.mode = "test"
        self.mid = {}
        self.config = {}

    def build(self):
        # 如果是第一层
        if not self.previous_layer:
            # 如果没有给定输入数据的维数，则报错
            if not self.input_dim:
                raise Exception('BuildError : 全连接层为第一层'
                                '时必须在构造方法中提供input_dim'
                                '参数!')
        # 如果不是第一层
        else:
            self.input_dim = self.previous_layer.output_shape[-1]

        # 设置层的输入输出形状
        self.input_shape[1] = self.input_dim

        # 初始化参数
        self.w = self.init((self.input_dim, self.output_dim))
        self.b = initializations.zeros((1, self.output_dim))
        self.trainable_params = [self.w, self.b]

    def forward_train(self, _input):
        self.input = _input
        # 前向计算
        mul = MulGate.forward(self.input, self.w)
        add = AddGate.forward(mul, self.b)
        self.output = add
        # 保留中间计算结果
        self.mid['mul'] = mul
        return self.output

    def forward_test(self, _input):
        self.input = _input
        # 前向计算
        mul = MulGate.forward(self.input, self.w)
        add = AddGate.forward(mul, self.b)
        self.output = add
        # 删除中间计算结果
        del mul
        return self.output

    def backward(self, _d_output):
        self.d_output = _d_output
        d_add = self.d_output
        mul = self.mid.get('mul')
        # 反向传播
        d_mul, d_b = AddGate.backward(mul, self.b, d_add)
        self.d_input, d_w = MulGate.backward(self.input, self.w, d_mul)
        # 将需要更新的参数的梯度放在grads列表中
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
        self.trainable_params = []
        self.grads = []

        self.previous_layer = None
        self.latter_layer = None

        self.mode = "test"
        self.mid = {}
        self.config = {}

    def build(self):
        # 如果是第一层
        if not self.previous_layer:
            raise Exception('BuildError: 暂时不支持激活层为第一层!')
        self.input_shape = self.previous_layer.output_shape
        self.output_shape = self.input_shape

    def forward_train(self, _input):
        self.input = _input
        self.output = self.activation.forward(self.input)
        return self.output

    def forward_test(self, _input):
        self.input = _input
        self.output = self.activation.forward(self.input)
        return self.output

    def backward(self, _d_output):
        self.d_output = _d_output
        self.d_input = self.activation.backward(self.input, self.output, self.d_output)
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

        # 模型参数
        self.w = None
        self.b = None

        # 模型基本成员
        self.trainable = True
        self.trainable_params = []
        self.grads = []

        self.previous_layer = None
        self.latter_layer = None

        self.mode = "test"
        self.mid = {}
        self.config = {}

    def build(self):
        # ==================================
        # 如果是第一层
        if not self.previous_layer:
            # 如果没有指明输入数据形状，则报错
            if not self.CHW_shape:
                raise Exception('BuildError : 卷积层为第一层时必'
                                '须在构造方法中提供input_shape参数!')
            # 如果指明了输入数据形状
            else:
                self.input_shape[1:] = self.CHW_shape
        # 如果不是第一层
        else:
            self.input_shape = self.previous_layer.output_shape
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

    def forward_train(self, _input):
        self.input = _input
        self.output_shape[0] = self.input_shape[0] = self.input.shape[0]

        # 计算形状
        N, C, H, W = self.input_shape
        KN, KC, KH, KW = self.w_shape
        CH, CW = self.output_shape[2:]

        # 准备工作
        x = self.input
        rowing_w = self.w.reshape(KN, KC * KH * KW)

        # 前向计算
        columnize_x = im2col_NCHW(x, KH, KW, self.stride)
        rowing_mul = MulGate.forward(rowing_w, columnize_x)
        # rowing_w:     (KN, C*KH*KW)
        # columnize_x:  (C*KH*KW, N*CH*CW)
        # rowing_mul:   (KN, N*CH*CW)
        # mul:          (N, KN, CH, CW)
        mul = rowing_mul.reshape(KN, N, CH, CW).swapaxes(0, 1)
        add = AddGate.forward(mul, self.b)
        act = self.activation.forward(add)
        self.output = act

        # 保存中间计算结果
        self.mid['rowing_w'] = rowing_w
        self.mid['columnize_x'] = columnize_x
        self.mid['mul'] = mul
        self.mid['add'] = add

        return self.output

    def forward_test(self, _input):
        self.input = _input

        # 计算形状
        N, C, H, W = self.input_shape
        KN, KC, KH, KW = self.w_shape
        CH, CW = self.output_shape[2:]

        # 准备工作
        x = self.input
        rowing_w = self.w.reshape(KN, KH * KW * C)
        # mul: (N, KN, CH, CW)
        mul = np.empty(self.output_shape, dtype=self.input.dtype)

        # 前向计算
        for n in xrange(N):
            # 将输入变形
            x_n = np.array(x[n]).reshape(1, C, H, W)
            # 计算
            # columnize_x_n: (C*KH*KW, 1*CH*CW)
            columnize_x_n = im2col_NCHW(x_n, KH, KW, self.stride)
            # rowing_mul_n: (KN, 1*CH*CW)
            rowing_mul_n = MulGate.forward(rowing_w, columnize_x_n)
            # 将输出变形
            mul_n = rowing_mul_n.reshape(1, KN, CH, CW)
            # 保存
            mul[n] = mul_n

        add = AddGate.forward(mul, self.b)
        del mul

        act = self.activation.forward(add)
        del add

        self.output = act
        return self.output

    def backward(self, _d_output):
        self.d_output = _d_output

        # 形状变量
        N, C, H, W = self.input_shape
        KN, KC, KH, KW = self.w_shape
        CH, CW = self.output_shape[2:]

        # 准备工作
        d_x = np.empty_like(self.input, dtype=self.input.dtype)
        d_w = np.zeros_like(self.w, dtype=np.double)

        # 提取中间计算变量
        rowing_w = self.mid['rowing_w']
        columnize_x = self.mid['columnize_x']
        mul = self.mid['mul']
        add = self.mid['add']
        act = self.output

        # 反向求导
        d_act = self.d_output
        d_add = self.activation.backward(add, act, d_act)
        d_mul, d_b = AddGate.backward(mul, self.b, d_add)
        d_rowing_mul = d_mul.swapaxes(0, 1).reshape(KN, N * CH * CW)
        d_rowing_w, d_columnize_x = MulGate.backward(rowing_w, columnize_x, d_rowing_mul)
        d_x = col2im_NCHW(d_columnize_x, KH, KW, CH, CW, self.stride)

        self.d_input = d_x
        d_w = d_rowing_w.reshape(KN, KC, KH, KW)

        # 保存梯度
        self.grads = [d_w, d_b]
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

        # 模型基本成员
        self.trainable = False
        self.trainable_params = []
        self.grads = []

        self.previous_layer = None
        self.latter_layer = None

        self.mode = "test"
        self.mid = {}
        self.config = {}

    def build(self):
        # 如果是第一层
        if not self.previous_layer:
            if not self.CHW_shape:
                raise Exception('BuildError : Flatten为第一层时必'
                                '须在构造方法中提供input_shape参数!')
            else:
                self.input_shape[1:] = self.CHW_shape
        # 如果不是第一层
        else:
            self.input_shape = self.previous_layer.output_shape
        self.output_shape[-1] = np.product(self.input_shape[1:])

    def forward_train(self, _input):
        self.input = _input
        self.output = self.input.reshape(self.output_shape)
        return self.output

    def forward_test(self, _input):
        self.input = _input
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

        # 模型基本成员
        self.trainable = False
        self.trainable_params = []
        self.grads = []

        self.previous_layer = None
        self.latter_layer = None

        self.mode = "test"
        self.mid = {}
        self.config = {}

    def build(self):
        #  如果是第一层
        if not self.previous_layer:
            if not self.CHW_shape:
                raise Exception('BuildError : MeanPoolnig2D层为第一层'
                                '时必须在构造方法中提供input_shape参数!')
            self.input_shape[1:] = self.CHW_shape
        # 如果具有前一层
        else:
            self.input_shape = self.previous_layer.output_shape

        if self.pooling_size[0] != self.pooling_size[1]:
            raise Exception('BuildError : 目前MeanPooling2D层只支持方形采样！')

        N, C, H, W = self.input_shape
        self.output_shape = [N, C, H/self.pooling_size[0], W/self.pooling_size[1]]

    def forward_train(self, _input):
        self.input = _input

        # 计算形状
        N, C, H, W = self.input_shape
        KN, KC, KH, KW = 1, 1, self.pooling_size[0], self.pooling_size[1]
        CH, CW = self.output_shape[2:]
        stride = KH

        # 准备工作
        rowing_w = np.ones(shape=(KN, KC * KH * KW), dtype=self.input.dtype) / (KH * KW)
        columnize_x = np.empty(shape=(KH * KW, N * C * CH * CW), dtype=self.input.dtype)
        # columnize_x: (KH*KW, N*C*CH*CW)

        # 前向计算
        for n in xrange(N):
            for c in xrange(C):
                # 依次计算input[n][c]的im2col_HW结果，并存放在columnize_x中，每个im2col_HW结果依次挨着横着放
                ocol = (n * C + c) * CH * CW
                columnize_x[0:KH*KW, ocol: ocol+CH*CW] = im2col_HW(self.input[n][c], KH, KW, stride)

        # rowing_output: (KN, N*C*CH*CW)
        # output: (N, C, CH, CW) = (N, C, H/KH, W/KW)
        rowing_output = MulGate.forward(rowing_w, columnize_x)
        self.output = rowing_output.reshape(self.output_shape)

        self.mid['rowing_w'] = rowing_w
        self.mid['columnize_x'] = columnize_x

        return self.output

    def forward_test(self, _input):
        self.input = _input

        N, C, H, W = self.input_shape
        KN, KC, KH, KW = 1, 1, self.pooling_size[0], self.pooling_size[1]
        CH, CW = self.output_shape[2:]
        stride = KH

        rowing_w = np.ones(shape=(KN, KC * KH * KW), dtype=self.input.dtype) / (KH * KW)
        columnize_x = np.empty(shape=(KH * KW, N * C * CH * CW), dtype=self.input.dtype)

        for n in xrange(N):
            for c in xrange(C):
                ocol = (n * C + c) * CH * CW
                columnize_x[0:KH * KW, ocol: ocol + CH * CW] = im2col_HW(self.input[n][c], KH, KW, stride)

        rowing_output = MulGate.forward(rowing_w, columnize_x)
        del columnize_x

        self.output = rowing_output.reshape(self.output_shape)
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

        d_rowing_output = self.d_output.reshape(1, N * C * CH * CW)
        _, d_columnize_x = MulGate.backward(rowing_w, columnize_x, d_rowing_output)

        for n in xrange(N):
            for c in xrange(C):
                ocol = (n * C + c) * CH * CW
                d_input[n][c] = col2im_HW(d_columnize_x[0:KH*KW, ocol: ocol+CH*CW], KH, KW, CH, CW, stride)

        self.d_input = d_input
        return self.d_input



