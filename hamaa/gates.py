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


from .clib import im2colutils, col2imutils
from .utils import np_utils

import numpy as np


class MulGate:
    """乘法单元"""

    @staticmethod
    def forward(x, w):
        return np.dot(x, w)

    @staticmethod
    def backward(x, w, d_z):
        d_x = np.dot(d_z, np.transpose(w))
        d_w = np.dot(np.transpose(x), d_z)
        return d_x, d_w


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


class SoftmaxGate:
    """softmax单元"""

    @staticmethod
    def forward(x):
        z = np.exp(x)
        return z / np.sum(z, axis=1, keepdims=True)

    @staticmethod
    def backward(x, d_z):
        raise Exception('Error: SoftmaxGate::backward() is not implemented yet!')



class Conv2DGate:

    @staticmethod
    def forward(x, w, stride, fmt='NCHW'):
        if fmt == 'NCHW':
            N, C, H, W = x.shape
            KN, KC, KH, KW = w.shape
            CH = (H - KH) / stride + 1
            CW = (W - KW) / stride + 1

            col_x = Conv2DGate.im2col_NCHW_1(x, KH, KW, stride)
            # col_x = Conv2DGate.im2col_NCHW_2(x, KH, KW, stride)
            row_w = w.reshape(KN, KC * KH * KW)
            z = MulGate.forward(row_w, col_x)
            z = z.reshape(N, KN, CH, CW)

            cache = (row_w, col_x)
            return z, cache
        elif fmt == 'HW':
            H, W = x.shape
            KH, KW = w.shape
            CH = (H - KH) / stride + 1
            CW = (W - KW) / stride + 1

            col_x = Conv2DGate.im2col_HW(x, KH, KW, stride)
            row_w = w.reshape(1, KH * KW)

            z = MulGate.forward(row_w, col_x)
            z = z.reshape(CH, CW)

            cache = (row_w, col_x)
            return z, cache
        else:
            raise Exception('Invalid Conv2DGate forward format: ' + fmt + '!')

    @staticmethod
    def backward(x, w, stride, d_z, fmt='NCHW', cache=None):
        if fmt == 'NCHW':
            N, C, H, W = x.shape
            KN, KC, KH, KW = w.shape
            CH = (H - KH) / stride + 1
            CW = (W - KW) / stride + 1

            if cache:
                row_w, col_x = cache
            else:
                col_x = Conv2DGate.im2col_NCHW_1(x, KH, KW, stride)
                # col_x = Conv2DGate.im2col_NCHW_3(x, KH, KW, stride)
                row_w = w.reshape(KN, KC * KH * KW)

            d_z = d_z.reshape(KN, N * CH * CW)
            d_row_w, d_col_x = MulGate.backward(row_w, col_x, d_z)

            d_w = d_row_w.reshape(KN, KC, KH, KW)
            d_x = Conv2DGate.col2im_NCHW_naive(d_col_x, KH, KW, CH, CW, stride)
            return d_x, d_w
        elif fmt == 'HW':
            H, W = x.shape
            KH, KW = w.shape
            CH = (H - KH) / stride + 1
            CW = (W - KW) / stride + 1

            if cache:
                row_w, col_x = cache
            else:
                col_x = Conv2DGate.im2col_HW(x, KH, KW, stride)
                row_w = w.reshape(1, KH * KW)

            d_z = d_z.reshape(1, CH * CW)
            d_row_w, d_col_x = MulGate.backward(row_w, col_x, d_z)

            d_w = d_row_w.reshape(KH, KW)
            d_x = Conv2DGate.col2im_HW(d_col_x, KH, KW, CH, CW, stride)
            return d_x, d_w
        else:
            raise Exception('Invalid Conv2DGate backward format: ' + fmt + '!')



class MaxPooling2DGate:
    @staticmethod
    def forward():
        pass

    @staticmethod
    def backward():
        pass


class MeanPooling2DGate:
    @staticmethod
    def forward():
        pass

    @staticmethod
    def backward():
        pass

