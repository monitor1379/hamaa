# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test4_Refactor_Covolution2D.py
@time: 2016/10/23 9:44


"""

from hamaa.layers import Dense, Activation, Convolution2D, Flatten , MeanPooling2D
from hamaa.models import Sequential
from hamaa.datasets import datasets
from hamaa.utils import np_utils
from hamaa.optimizers import SGD
from hamaa.utils.conv_utils import *
from hamaa.utils.time_utils import *

import sys
import numpy as np
import matplotlib.pyplot as plt



def print_size(name, var):
    t = sys.getsizeof(var) * 1.0 / (1024 * 1024)
    print "{} 's size:{}MB".format(name, t)

def run():
    # test1()
    test2()


def test2():
    N, C, H, W = 1, 1, 28, 28
    KN, KC, KH, KW = 15, C, 5, 5
    stride = 1
    CH = (H - KH) / stride + 1
    CW = (W - KW) / stride + 1

    x = np.arange(N*C*H*W).reshape(N, C, H, W).astype(np.double)
    w = np.ones((KN, KC, KH, KW)).astype(np.double)


    tic('test')
    z = np.empty((N, KN, CH, CW), dtype=np.double)

    rowing_w = w.reshape(KN, KH*KW*C)
    columnize_x = np.empty((N, KH*KW*C, CH*CW), dtype=np.double)

    # im2col_HW来实现卷积
    for n in xrange(N):
        x_n = np.array(x[n]).reshape(1, C, H, W)
        columnize_x_n = im2col_NCHW(x_n, KH, KW, stride)
        rowing_z_n = np.dot(rowing_w, columnize_x_n)
        z_n = rowing_z_n.reshape(1, KN, CH, CW)
        z[n] = z_n
    toc(True)





def test1():
    N, C, H, W = 10000, 1, 28, 28
    KN, KC, KH, KW = 15, C, 5, 5
    stride = 1
    CH = (H - KH) / stride + 1
    CW = (W - KW) / stride + 1

    x = np.random.rand(N, C, H, W)
    w = np.random.rand(KN, KC, KH, KW)

    layer = Convolution2D(nb_kernel=KN, kernel_height=KH, kernel_width=KW, input_shape=(C, H, W))
    layer.build()

    z = layer.forward(x)

    print_size('x', x)
    print_size('w', w)
    print_size('z', z)


if __name__ == '__main__':
    run()



