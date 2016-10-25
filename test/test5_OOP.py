# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test5_OOP.py
@time: 2016/10/24 15:10


"""

import numpy as np

from hamaa.utils.time_utils import tic, toc
from hamaa.utils.conv_utils import *
from hamaa.gates import *



def run():
    # test1()
    test2()


def test2():
    N, C, H, W = 10, 3, 28, 28
    KN, KC, KH, KW = 15, C, 5, 5
    stride = 1
    CH, CW = (H - KH) / stride + 1, (W - KW) / stride + 1

    x = np.arange(N * C * W * H).reshape(N, C, H, W).astype(np.double)
    w = np.ones((KN, KC, KH, KW))

    tic('reshape')
    for ttt in xrange(1000):
        rowing_w = w.reshape(KN, C * KH * KW)
    toc(True)


def test1():
    N, C, H, W = 10, 3, 28, 28
    KN, KC, KH, KW = 15, C, 5, 5
    stride = 1
    CH, CW = (H - KH) / stride + 1, (W - KW) / stride + 1

    x = np.arange(N * C * W * H).reshape(N, C, H, W).astype(np.double)
    w = np.ones((KN, KC, KH, KW))

    rowing_w = w.reshape(KN, KH * KW * C)

    columnize_x = np.empty((N, KH * KW * C, CH * CW), dtype=np.double)

    tic('old')
    for ttt in xrange(1000):
        # 前向计算
        for n in xrange(N):
            # 将输入变形
            x_n = np.array(x[n]).reshape(1, C, H, W)
            # 计算
            columnize_x_n = im2col_NCHW(x_n, KH, KW, stride)
            rowing_mul_n = MulGate.forward(rowing_w, columnize_x_n)
            # 将输出变形
            mul_n = rowing_mul_n.reshape(1, KN, CH, CW)
            columnize_x[n] = columnize_x_n
    toc(True)

    tic('new')
    for ttt in xrange(1000):
        columnize_x = im2col_NCHW(x, KH, KW, stride)
        mul = np.dot(rowing_w, columnize_x)
    toc(True)



if __name__ == '__main__':
    run()