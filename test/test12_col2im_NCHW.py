# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test12_col2im_NCHW.py
@time: 2016/10/17 10:40


"""

import numpy as np
from hamaa.gates import Conv2DGate, MulGate
from hamaa.utils.np_utils import eval_numerical_gradient_array, sum_abs_err
from hamaa.utils.time_utils import tic, toc


class Convolution2D:

    def __init__(self):
        self.col_x = None
        self.row_w = None
        self.stride = 1

    def forward(self, x, w):
        N, C, H, W = x.shape
        KN, KC, KH, KW = w.shape
        CH = (H - KH) / self.stride + 1
        CW = (W - KW) / self.stride + 1

        self.col_x = Conv2DGate.im2col_NCHW_1(x, KH, KW, self.stride)
        self.row_w = w.reshape(KN, KC * KH * KW)
        z = MulGate.forward(self.row_w, self.col_x)
        z = z.reshape(N, KN, CH, CW)
        return z

    def backward(self, x, w, d_z):
        N, C, H, W = x.shape
        KN, KC, KH, KW = w.shape
        CH = (H - KH) / self.stride + 1
        CW = (W - KW) / self.stride + 1
        d_z = d_z.reshape(KN, N * CH * CW)

        d_row_w, d_col_x = MulGate.backward(self.row_w, self.col_x, d_z)
        d_w = d_row_w.reshape(KN, KC, KH, KW)
        d_x = Conv2DGate.col2im_NCHW_naive(d_col_x, KH, KW, CH, CW, self.stride)
        return d_x, d_w


def run():
    N, C, H, W = 1, 1, 4, 5
    KN, KC, KH, KW = 1, C, 3, 3
    stride = 1
    CH = (H - KH) / stride + 1
    CW = (W - KW) / stride + 1

    x = np.arange(N * C * H * W).reshape(N, C, H, W).astype(dtype=np.double)
    w = np.ones((KN, KC, KH, KW), dtype=np.double)
    d_z = np.random.rand(N, KN, CH, CW)


    conv = Convolution2D()
    z = conv.forward(x, w)
    d_x, d_w = conv.backward(x, w, d_z)

    grad_x, grad_w = Conv2DGate.backward(x, w, stride, d_z)

    print (d_x == grad_x).all()
    print (d_w == grad_w).all()

    x = x[0][0]
    w = w[0][0]

    grad_x, grad_w = Conv2DGate.backward(x, w, stride, d_z, fmt='HW')


    print (d_x[0][0] == grad_x).all()
    print (d_w[0][0] == grad_w).all()



if __name__ == '__main__':
    run()