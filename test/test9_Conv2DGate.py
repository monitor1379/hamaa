# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test9_Conv2DGate.py
@time: 2016/10/16 9:14


"""

from hamaa.gates import Conv2DGate, MulGate
from hamaa.utils.time_utils import tic, toc
from hamaa.utils.np_utils import rot180
from hamaa.utils import image_utils
from hamaa.clib import im2colutils
from hamaa.datasets import datasets

import matplotlib.pyplot as plt
import numpy as np


def test1():
    N, C, H, W = (2, 1, 4, 5)
    KN, KC, KH, KW = (1, C, 2, 2)
    stride = 1

    CH = (H - KH) / stride + 1
    CW = (W - KW) / stride + 1

    x = np.arange(N * C * H * W).reshape(N, C, H, W).astype(np.double)
    w = np.ones((KN, KC, KH, KW), dtype=np.double) / (KH * KW)

    z = Conv2DGate.forward(x[1][0], w[0][0], stride, fmt='HW')
    print z

    z = Conv2DGate.forward(x, w, stride, fmt='NCHW')
    print z


def test2():
    N, C, H, W = (1, 1, 3, 4)
    x = np.arange(N * C * H * W).reshape(N, C, H, W).astype(np.double)

    im = datasets.load_lena()
    im = image_utils.hwc2chw(im)
    print im.shape
    r_im = rot180(im)
    r_im = image_utils.chw2hwc(r_im)
    print r_im.shape
    plt.imshow(r_im)
    plt.show()


def test3():
    im = datasets.load_lena()[:, :, 0].astype(np.double)
    im = im.reshape(1, 1, im.shape[0], im.shape[1]) / 255

    # w = np.array([[-1], [0], [1]])
    # w = w.reshape(1, 1, w.shape[0], w.shape[1])
    w = np.random.randn(1, 1, 3, 3)

    cim = Conv2DGate.forward(im, w, 1)

    plt.subplot(121)
    plt.imshow(im[0][0])
    plt.subplot(122)
    plt.imshow(cim[0][0])

    plt.show()


def col2im_HW(col_x, KH, KW, CH, CW, stride):
    H = (CH - 1) * stride + KH
    W = (CW - 1) * stride + KW
    # print H, W
    x = np.zeros(shape=(H, W), dtype=col_x.dtype)
    for j in xrange(CH * CW):
        col = j % CW * stride
        row = j / CW * stride
        x[row: row + KH, col: col + KW] += col_x[:, j].reshape(KH, KW)
    return x


def test4():
    N, C, H, W = (1, 1, 3, 4)
    KN, KC, KH, KW = (1, C, 2, 2)
    stride = 1

    CH = (H - KH) / stride + 1
    CW = (W - KW) / stride + 1

    x = np.arange(N * C * H * W).reshape(N, C, H, W).astype(np.double)
    w = np.ones((KN, KC, KH, KW), dtype=np.double)

    x = x[0][0]
    w = w[0][0]

    print x
    print w

    z, cache = Conv2DGate.forward(x, w, stride, fmt='HW')
    d_z = z
    d_x, d_w = Conv2DGate.backward(x, w, stride, d_z, cache=cache, fmt='HW')
    print d_x, d_w


def run():
    # test1()
    # test2()
    # test3()
    test4()



if __name__ == '__main__':
    run()