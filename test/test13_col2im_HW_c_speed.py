# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test13_col2im_HW_c_speed.py
@time: 2016/10/17 19:44


"""

import numpy as np
from hamaa.gates import Conv2DGate, MulGate
from hamaa.utils.np_utils import eval_numerical_gradient_array, sum_abs_err
from hamaa.utils.time_utils import tic, toc
from hamaa.clib import im2colutils, col2imutils


def test1():
    N, C, H, W = (1, 1, 28, 28)
    KN, KC, KH, KW = (1, C, 5, 5)
    stride = 1
    CH = (H - KH) / stride + 1
    CW = (W - KW) / stride + 1

    x = np.arange(N * C * H * W).reshape(N, C, H, W).astype(np.double)
    w = np.ones((KN, KC, KH, KW)).astype(np.double)

    x = x[0][0]
    w = w[0][0]


    col_x = im2colutils.im2col_HW(x, KH, KW, stride)

    itertimes = 100

    tic('naive')
    for i in xrange(itertimes):
        x1 = Conv2DGate.col2im_HW_naive(col_x, KH, KW, CH, CW, stride)
    toc(True)


    tic('col2imutils')
    for i in xrange(itertimes):
        x2 = col2imutils.col2im_HW(col_x, KH, KW, CH, CW, stride)
    toc(True)

    print (x1 == x2).all()

def run():
    test1()


if __name__ == '__main__':
    run()