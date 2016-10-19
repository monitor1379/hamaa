# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test15_conv_utils.py
@time: 2016/10/17 23:14


"""

from hamaa.utils.conv_utils import *
import numpy as np


N, C, H, W = (2, 2, 4, 5)
KN, KC, KH, KW = (1, C, 2, 2)
stride = 1
CH = (H - KH) / stride + 1
CW = (W - KW) / stride + 1

ttx = np.arange(N * C * H * W).reshape(N, C, H, W).astype(np.double)
ttw = np.ones((KN, KC, KH, KW)).astype(np.double)


def run():
    assert test_im2col_HW()
    assert test_im2col_NCHW()
    assert test_col2im_HW()
    assert test_col2im_NCHW()


def test_im2col_HW():
    x = ttx[0][0]
    z1 = im2col_HW(x, KH, KW, stride)
    z2 = im2col_HW_py(x, KH, KW, stride)
    return (z1 == z2).all()


def test_im2col_NCHW():
    z1 = im2col_NCHW(ttx, KH, KW, stride)
    z2 = im2col_NCHW_py(ttx, KH, KW, stride)
    return (z1 == z2).all()


def test_col2im_HW():
    x = ttx[0][0]
    columnize_x = im2col_HW(x, KH, KW, stride)
    # print columnize_x
    z1 = col2im_HW(columnize_x, KH, KW, CH, CW, stride)
    z2 = col2im_HW_py(columnize_x, KH, KW, CH, CW, stride)
    # print z1
    # print z2
    return (z1 == z2).all()
    # return True

def test_col2im_NCHW():
    columnize_x = im2col_NCHW(ttx, KH, KW, stride)
    z1 = col2im_NCHW(columnize_x, KH, KW, CH, CW, stride)
    z2 = col2im_NCHW_py(columnize_x, KH, KW, CH, CW, stride)
    # print z1
    # print z2
    return (z1 == z2).all()
    # return True


if __name__ == '__main__':
    run()




