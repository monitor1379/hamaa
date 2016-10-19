# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test14_Conv2DGate.py
@time: 2016/10/17 20:17


"""

import numpy as np
from hamaa.gates import Conv2DGate, MulGate
from hamaa.utils.np_utils import eval_numerical_gradient_array, sum_abs_err
from hamaa.utils.time_utils import tic, toc
from hamaa.clib import im2colutils, col2imutils


def test1():
    N, C, H, W = (1, 1, 4, 5)
    KN, KC, KH, KW = (1, C, 3, 3)
    stride = 1
    CH = (H - KH) / stride + 1
    CW = (W - KW) / stride + 1

    x = np.arange(N * C * H * W).reshape(N, C, H, W).astype(np.double)
    w = np.ones((KN, KC, KH, KW)).astype(np.double)

    z, cache = Conv2DGate.forward(x, w, stride)
    d_z = np.random.rand(N, KN, CH, CW)
    # d_z = np.random.rand(CH, CW)
    d_x, d_w = Conv2DGate.backward(x, w, stride, d_z, cache=None)

    grad_x = eval_numerical_gradient_array(lambda t: Conv2DGate.forward(t, w, stride)[0], x, d_z)
    grad_w = eval_numerical_gradient_array(lambda t: Conv2DGate.forward(x, t, stride)[0], w, d_z)

    print sum_abs_err(d_x, grad_x)
    print sum_abs_err(d_w, grad_w)



def run():
    test1()


if __name__ == '__main__':
    run()