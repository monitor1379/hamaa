# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: gates.py
@time: 2016/9/11 9:01

计算单元
"""


import numpy as np
from utils.time_utils import tic, toc
import core.conv
# np.seterr(all='raise')

class MulGate:
    """乘法单元"""

    @staticmethod
    def forward(w, x):
        return np.dot(x, w)

    @staticmethod
    def backward(w, x, d_z):
        d_w = np.dot(np.transpose(x), d_z)
        d_x = np.dot(d_z, np.transpose(w))
        # print d_w
        return d_w, d_x


class AddGate:
    """专用于n*m维与1*m维的加法单元"""

    @staticmethod
    def forward(x, b):
        return x + b

    @staticmethod
    def backward(x, b, d_z):
        d_x = np.array(d_z)
        d_b = np.array(d_z)
        if b.shape[0] == 1:
            d_b = np.sum(d_b, axis=0, keepdims=True)
        if b.shape[1] == 1:
            d_b = np.sum(d_b, axis=1, keepdims=True)
        return d_x, d_b


class LinearGate:
    """线性计算单元"""

    @staticmethod
    def forward(x):
        return np.array(x)

    @staticmethod
    def backward(x, d_z):
        return np.ones_like(x) * d_z

class SigmoidGate:
    """sigmoid单元"""

    @staticmethod
    def forward(x):
        z = 1.0 / (1.0 + np.exp(-x))
        return z

    @staticmethod
    def backward(x, d_z):
        a = SigmoidGate.forward(x)
        d_x = a * (1 - a) * d_z
        return d_x


class TanhGate:
    """tanh单元"""

    @staticmethod
    def forward(x):
        e1 = np.exp(x)
        e2 = np.exp(-x)
        return (e1 - e2) / (e1 + e2)

    @staticmethod
    def backward(x, d_z):
        a = TanhGate.forward(x)
        d_x = (1 - a**2) * d_z
        return d_x


class ReLUGate:
    """relu单元"""
    @staticmethod
    def forward(x):
        z = np.array(x)
        z[z < 0] = 0
        return z

    @staticmethod
    def backward(x, d_z):
        d_x = np.ones_like(x)
        d_x[x < 0] = 0
        return d_x * d_z


class Conv2DGate:
    """二维卷积单元"""

    mode_dict = {'full', 'same', 'valid'}

    @staticmethod
    def forward(im, kernel, **kwargs):
        if kwargs.get('mode') is None:
            return Conv2DGate.convolve(im, kernel, kwargs['stride'], kwargs['padding_size'])
        else:
            mode = kwargs.get('mode')
            if mode not in Conv2DGate.mode_dict:
                raise RuntimeError('Unkown convolution mode. Only support "full" and "valid".')
            if mode == 'full':
                return Conv2DGate.convolve(im, kernel, 1, [kernel.shape[0] - 1, kernel.shape[1] - 1])
            elif mode == 'valid':
                return Conv2DGate.convolve(im, kernel, 1, [0, 0])

    @staticmethod
    def convolve(im, kernel, stride, padding_size):
        pim = Conv2DGate.padding(im, padding_size)
        conv_shape = Conv2DGate.get_output_shape(im.shape, kernel.shape, stride, padding_size)
        im2col_output = Conv2DGate.c_im2col(pim, kernel.shape, stride, conv_height=conv_shape[0], conv_width=conv_shape[1])
        result = np.dot(kernel.reshape(1, -1), im2col_output).reshape(conv_shape)
        return result

    # @staticmethod
    # def fast_convolve(im, kernel, stride, padding_size, im2col_output):
    #     conv_shape = Conv2DGate.get_output_shape(im.shape, kernel.shape, stride, padding_size)
    #     result = np.dot(kernel.reshape(1, -1), im2col_output).reshape(conv_shape)
    #     return result

    @staticmethod
    def backward(im, kernel, d_z, grad_type=1):
        """
        grad_type: 0:只对kernel求导；1：都求导
        """
        d_im = None
        d_kernel = Conv2DGate.forward(im, d_z, mode='valid')
        if grad_type != 0:
            d_im = Conv2DGate.forward(d_z, Conv2DGate.rot180(kernel), mode='full')
        return d_im, d_kernel

    # @staticmethod
    # def fast_backward(im, kernel, d_z, im2col_output, grad_type=1):
    #     d_im = None
    #     d_kernel = Conv2DGate.fast_convolve(im, kernel, 1, [0, 0], im2col_output)
    #     if grad_type != 0:
    #         d_im = Conv2DGate.forward(d_z, Conv2DGate.rot180(kernel), mode='full')
    #     return d_im, d_kernel


    @staticmethod
    def rot180(im):
        return np.fliplr(np.flipud(im))

    @staticmethod
    def padding(im, padding_size, mode='zero'):
        """零填充"""
        if mode == 'zero':
            pim = np.zeros((im.shape[0] + 2 * padding_size[0], im.shape[1] + 2 * padding_size[1]), dtype=im.dtype)
            pim[padding_size[0]:padding_size[0] + im.shape[0], padding_size[1]:padding_size[1] + im.shape[1]] = im
        return pim

    @staticmethod
    def get_output_shape(input_size, kernel_size, stride, padding_size):
        """计算卷积结果的大小"""
        height = (input_size[0] + 2 * padding_size[0] - kernel_size[0]) / stride + 1
        width = (input_size[1] + 2 * padding_size[1] - kernel_size[1]) / stride + 1
        return height, width

    @staticmethod
    def im2col(im, kernel_size, stride, conv_height=None, conv_width=None):
        """图片列化"""
        # 卷积核大小
        kernel_height, kernel_width = kernel_size
        # 如果没有给出卷积结果大小，则计算一遍
        if conv_height == None or conv_width == None:
            conv_height, conv_width = Conv2DGate.get_output_shape(im.shape, kernel_size, stride, padding_size=[0, 0])
        # 将原图片"列化"后的大小
        im2col_height, im2col_width = (kernel_height * kernel_width, conv_height * conv_width)
        output = np.empty((im2col_height, im2col_width))
        # 列下标，代表输出结果的第几列
        output_idx = 0
        for row in xrange(0, conv_height):
            for col in xrange(0, conv_width):
                row_start_idx = row * stride
                col_start_idx = col * stride
                # 将卷积区域向量化成一列，保存在输出结果output中的第output_idx列中
                output[:, output_idx] = \
                    im[row_start_idx:row_start_idx + kernel_height, col_start_idx:col_start_idx + kernel_width]\
                                        .reshape(im2col_height)
                output_idx += 1
        return output

    # 封装好了的用C实现的im2col算法，和纯Py的im2col具有相同的接口
    @staticmethod
    def c_im2col(im, kernel_size, stride, conv_height, conv_width):
        return core.conv.im2col(im, kernel_size[0], kernel_size[1], stride, conv_height, conv_width)


class MaxPooling2DGate:
    """最大池化计算单元"""

    @staticmethod
    def forward(im, pool_size):
        ch, cw = Conv2DGate.get_output_shape(im.shape, pool_size, pool_size[0], padding_size=[0, 0])
        tmp = Conv2DGate.c_im2col(im, pool_size, pool_size[0], ch, cw)
        # tmp = Conv2DGate.im2col(im, pool_size, pool_size[0]) # 纯Py版本，速度较慢
        return np.max(tmp, axis=0).reshape(im.shape[0] / pool_size[0], im.shape[1] / pool_size[1])

    @staticmethod
    def backward(im, pool_size, d_pim):
        d_im = np.zeros_like(im)
        im_h, im_w = im.shape
        for i in xrange(0, im_h, pool_size[0]):
            for j in xrange(0, im_w, pool_size[1]):
                max_idx = np.argmax(im[i:i+pool_size[0], j:j+pool_size[1]])
                d_im[i:i+pool_size[0], j:j+pool_size[1]][max_idx / pool_size[0], max_idx % pool_size[1]] = d_pim[i / pool_size[0]][j / pool_size[1]]
        return d_im

class MeanPooling2DGate:
    """均值池化计算单元"""

    @staticmethod
    def forward(im, pool_size):
        if pool_size[0] != pool_size[1]:
            raise RuntimeError('pool_size:{} is not equal!'.format(pool_size))
        ch, cw = Conv2DGate.get_output_shape(im.shape, pool_size, pool_size[0], padding_size=[0, 0])
        tmp = Conv2DGate.c_im2col(im, pool_size, pool_size[0], ch, cw)
        return np.mean(tmp, axis=0).reshape(im.shape[0] / pool_size[0], im.shape[1] / pool_size[1])

    @staticmethod
    def backward(pool_size, d_pim):
        return np.kron(d_pim, np.ones(pool_size)) / (pool_size[0] * pool_size[1])