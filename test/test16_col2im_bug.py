# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test16_col2im_bug.py
@time: 2016/10/18 23:17


"""

from hamaa.utils.conv_utils import *
import numpy as np


def run():
    N, C, H, W = (2, 1, 4, 5)
    KN, KC, KH, KW = (1, C, 2, 2)
    stride = 1
    CH = (H - KH) / stride + 1
    CW = (W - KW) / stride + 1

    ttx = np.arange(N * C * H * W).reshape(N, C, H, W).astype(np.double)
    ttw = np.ones((KN, KC, KH, KW)).astype(np.double)

    columnize_x = im2col_NCHW(ttx, KH, KW, stride)
    z1 = col2im_NCHW_py(columnize_x, KH, KW, CH, CW, stride)
    print z1



if __name__ == '__main__':
    run()