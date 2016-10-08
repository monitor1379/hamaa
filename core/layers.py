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

from gates import *
from initializer import InitializerManager
from utils.time_utils import tic, toc
from utils.np_utils import eval_numerical_gradient_array, sum_abs_err

class Dense:
    """全连接层"""

    layer_type = 'Dense'

    def __init__(self, input_dim, output_dim, init, layer_name=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init = init

        self.layer_name = layer_name

        self.w = InitializerManager.get(init).initialize(shape=[input_dim, output_dim])
        self.b = InitializerManager.get('zero').initialize(shape=[1, output_dim])

        # 表示该层的参数可训练
        self.trainable = True

        self.order_number = -1

        # 保存前向计算的中间结果，加快计算速度
        self.input = 0
        self.mul = 0
        self.add = 0
        self.output = 0

        # 保存后向传播的中间结果，加快计算速度
        self.d_input = 0
        self.d_w = 0
        self.d_b = 0
        self.d_output = 0

    def forward(self, x):
        self.input = x
        self.mul = MulGate.forward(self.w, self.input)
        self.add = AddGate.forward(self.mul, self.b)
        self.output = self.add
        return self.add

    def backward(self, d_output):
        self.d_output = d_output
        d_mul, self.d_b = AddGate.backward(self.mul, self.b, self.d_output)
        self.d_w, self.d_input = MulGate.backward(self.w, self.input, d_mul)
        return self.d_input

    def update(self, lr):
        # print 'dense :', np.mean(self.d_w)
        self.w -= lr * self.d_w
        self.b -= lr * self.d_b


class Activation:
    """激活层"""

    layer_type = 'Activation'

    def __init__(self, activation, layer_name=None):
        self.act_type = activation
        self.layer_name = layer_name

        self.activation = ActivationManager.get(activation)
        self.trainable = False
        self.order_number = -1

        self.input = 0
        self.output = 0

        self.d_input = 0
        self.d_output = 0



    def forward(self, x):
        self.input = x
        self.output = self.activation.forward(x)
        return self.output

    def backward(self, d_output):
        self.d_output = d_output
        self.d_input = self.activation.backward(self.input, d_output)
        return self.d_input


class Convolution2D:
    """卷积层"""

    layer_type = 'Convolution2D'

    def __init__(self, input_shape, nb_kernel, kernel_height, kernel_width, activation='linear', stride=1, padding_size=(0, 0), layer_name=None):
        """input_shape指输入形如（channels, rows，cols）的3D张量"""
        self.input_shape = input_shape
        self.nb_kernel = nb_kernel
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.activation = ActivationManager.get(activation)
        self.stride = stride
        self.padding_size = padding_size
        self.layer_name = layer_name

        # 单个输入数据的形状
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

        # 单个输出数据的形状
        self.output_channels = nb_kernel
        input_size = (self.input_height, self.input_width)
        kernel_size = (kernel_height, kernel_width)
        self.output_height, self.output_width = Conv2DGate.get_output_shape(input_size, kernel_size, stride, padding_size)
        self.output_shape = (self.output_channels, self.output_height, self.output_width)

        self.trainable = True
        self.order_number = -1

        # 权重
        self.w = np.random.randn(nb_kernel, self.input_channels, kernel_height, kernel_width) / \
                 np.sqrt(self.input_channels * kernel_height * kernel_width)

        # self.w = np.random.randn(nb_kernel, self.input_channels, kernel_height, kernel_width)
        self.b = InitializerManager.get('zero').initialize(shape=[nb_kernel, 1, 1])

        self.n = -1
        self.input = 0
        self.conv_sum = 0
        self.add = 0
        self.act = 0
        self.output = 0

        self.d_input = 0
        self.d_w = np.empty(self.w.shape)
        self.d_b = np.empty(self.b.shape)
        self.d_conv_sum = 0
        self.d_add = 0
        self.d_act = 0
        self.d_output = 0

    def forward(self, x):
        self.input = x
        self.conv_sum = np.zeros((x.shape[0], self.output_shape[0], self.output_shape[1], self.output_shape[2]))
        if self.n == -1 or self.n != x.shape[0]:
            self.n = x.shape[0]
            self.add = np.zeros((self.n, self.output_shape[0], self.output_shape[1], self.output_shape[2]))
            self.d_conv_sum = np.zeros(self.conv_sum.shape)
            self.d_input = np.zeros(self.input.shape)
        # tic('conv2d ' + str(self.n * self.nb_kernel * self.input_channels) + '---->> ')

        for i in xrange(self.n):  # 对于每个输入数据
            for j in xrange(self.nb_kernel):  # 对于每个kernel
                for channel in xrange(self.input_channels):  # 对于每个feature map
                    self.conv_sum[i][j] += Conv2DGate.forward(self.input[i][channel], self.w[j][channel], stride=self.stride, padding_size=self.padding_size)
                # 加上偏置
                self.add[i][j] = AddGate.forward(self.conv_sum[i][j], self.b[j])
        # 激活函数
        self.act = self.activation.forward(self.add)
        self.output = self.act
        # toc()
        return self.output

    def backward(self, d_output):
        self.d_output = d_output
        self.d_act = self.d_output

        # 对激活过程求导
        self.d_add = self.activation.backward(self.add, self.d_act)
        # tic('conv2d ' + str(self.n * self.nb_kernel * self.input_channels) + '<<---- ')
        for i in xrange(self.n):  # 对于每个输入数据
            for j in xrange(self.nb_kernel):  # 对于每个kernel
                # 对偏置求导
                self.d_conv_sum[i][j], self.d_b[j] = AddGate.backward(self.conv_sum[i][j], self.b[j], self.d_add[i][j])
                for channel in xrange(self.input_channels):  # 对于每个feature map
                    # 如果该卷积层是输入层，则可以只对卷积核求导，以减少耗时
                    self.d_input[i][channel], self.d_w[j][channel] = Conv2DGate.backward(self.input[i][channel],
                                                                                         self.w[j][channel],
                                                                                         self.d_conv_sum[i][j],
                                                                                         grad_type=self.order_number)
        # toc()
        return self.d_input

    def update(self, lr):
        # print 'conv2d :', np.mean(self.d_w)
        self.w -= lr * self.d_w
        self.b -= lr * self.d_b


class MaxPooling2D:
    """最大池化层"""

    layer_type = 'MaxPooling2D'

    def __init__(self, input_shape, pool_size, layer_name=None):
        """input_shape指输入形如（channels, rows，cols）的3D张量"""
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.layer_name = layer_name

        self.output_shape = (input_shape[0], input_shape[1] / pool_size[0], input_shape[2] / pool_size[1])

        self.trainable = False

        self.n = -1
        self.input = 0
        self.output = 0

        self.d_input = 0
        self.d_output = 0

    def forward(self, x):
        self.input = x
        if self.n == -1 or self.n != x.shape[0]:
            self.n = x.shape[0]
            self.output = np.empty((x.shape[0], self.output_shape[0], self.output_shape[1], self.output_shape[2]))
            self.d_input = np.empty((self.n, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        # tic('maxp2d ' + str(self.n * self.output_shape[0]) + '---->> ')
        for i in xrange(self.n):
            for c in xrange(self.output_shape[0]):
                self.output[i][c] = MaxPooling2DGate.forward(self.input[i][c], self.pool_size)
        # toc()
        return self.output

    def backward(self, d_output):
        self.d_output = d_output
        # tic('maxp2d ' + str(self.n * self.output_shape[0]) + '<<---- ')
        for i in xrange(self.n):
            for c in xrange(self.output_shape[0]):
                self.d_input[i][c] = MaxPooling2DGate.backward(self.input[i][c], self.pool_size, self.d_output[i][c])
        # toc()
        return self.d_input



class MeanPooling2D:
    """平均池化层"""

    layer_type = 'MeanPooling2D'

    def __init__(self, input_shape, pool_size, layer_name=None):
        """input_shape指输入形如（channels, rows，cols）的3D张量"""
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.layer_name = layer_name

        self.output_shape = (input_shape[0], input_shape[1] / pool_size[0], input_shape[2] / pool_size[1])

        self.trainable = False

        self.n = -1
        self.input = 0
        self.output = 0

        self.d_input = 0
        self.d_output = 0

    def forward(self, x):
        self.input = x
        if self.n == -1 or self.n != x.shape[0]:
            self.n = x.shape[0]
            self.output = np.empty((x.shape[0], self.output_shape[0], self.output_shape[1], self.output_shape[2]))
            self.d_input = np.empty((self.n, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        # tic('mean2d ' + str(self.n * self.output_shape[0]) + '---->> ')
        for i in xrange(self.n):
            for c in xrange(self.output_shape[0]):
                self.output[i][c] = MeanPooling2DGate.forward(self.input[i][c], self.pool_size)
        # toc()
        return self.output

    def backward(self, d_output):
        self.d_output = d_output
        # tic('mean2d ' + str(self.n * self.output_shape[0]) + '<<---- ')
        for i in xrange(self.n):
            for c in xrange(self.output_shape[0]):
                self.d_input[i][c] = MeanPooling2DGate.backward(self.pool_size, self.d_output[i][c])
        # toc()
        return self.d_input


class Flatten:
    """拉伸层，将二维变成一维，且神经元数目不变"""

    layer_type = 'Flatten'

    def __init__(self, layer_name=None):
        self.layer_name = layer_name

        self.input_shape = 0
        self.output_shape = 0

        self.trainable = False

        self.input = 0
        self.output = 0

        self.d_input = 0
        self.d_output = 0

    def forward(self, x):
        self.input = x
        self.input_shape = x.shape[1:]
        self.output_shape = np.product(self.input_shape)
        n = x.shape[0]
        self.output = self.input.reshape((n, self.output_shape))
        return self.output

    def backward(self, d_output):
        self.d_output = d_output
        n = d_output.shape[0]
        self.d_input = self.d_output.reshape((n, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        return self.d_input


class ActivationManager:
    """激活函数管理类"""

    activations = {
            'linear': LinearGate,
            'sigmoid': SigmoidGate,
            'tanh': TanhGate,
            'relu': ReLUGate,
    }

    @staticmethod
    def get(name):
        return ActivationManager.activations[name]
