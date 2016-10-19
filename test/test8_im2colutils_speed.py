# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test8_im2colutils_speed.py
@time: 2016/10/13 20:56


"""


from hamaa.utils.time_utils import tic, toc
from hamaa.gates import *
import numpy as np



def run():
    N, C, H, W = (10, 6, 28, 28)
    KN, KC, KH, KW = (1, C, 5, 5)
    itertimes = 1000

    images = np.arange(N * C * H * W).reshape(N, C, H, W).astype(np.double)
    filters = np.arange(KN * KC * KH * KW).reshape(KN, KC, KH, KW).astype(np.double)
    stride = 1

    verbose = True

    tic('im2col_NCHW_1')
    for i in xrange(itertimes):
        col_x_1 = Conv2DGate.im2col_NCHW_naive(images, KH, KW, stride)
    toc(verbose)

    tic('im2col_NCHW_2')
    for i in xrange(itertimes):
        col_x_2 = Conv2DGate.im2col_NCHW_1(images, KH, KW, stride)
    toc(verbose)

    tic('im2col_NCHW_3')
    for i in xrange(itertimes):
        col_x_3 = Conv2DGate.im2col_NCHW_2(images, KH, KW, stride)
    toc(verbose)

    # print '============================'
    # print col_x_1
    # print '============================'
    # print col_x_2
    # print '============================'
    # print col_x_3



if __name__ == '__main__':
    run()