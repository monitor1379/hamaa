# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test11_hamaa5_Convolution.py
@time: 2016/10/17 10:20


"""

import numpy as np
from hamaa.gates import Conv2DGate, MulGate
from hamaa.utils.time_utils import tic, toc

def run():
    N, C, H, W = 1, 1, 4, 5
    KN, KC, KH, KW = 1, C, 3, 3

    stride = 1

    CH = (H - KH) / stride + 1
    CW = (W - KW) / stride + 1

    x = np.ones((N, C, H, W), dtype=np.double)
    w = np.ones((KN, KC, KH, KW), dtype=np.double)


    itertimes = 1000

    tic('forward')
    for i in xrange(itertimes):
        # =====================================================
        # forward
        col_x = Conv2DGate.im2col_NCHW_1(x, KH, KW, stride)
        row_w = w.reshape(KN, KC * KH * KW)
        CH = (H - KH) / stride + 1
        CW = (W - KW) / stride + 1
        z = MulGate.forward(row_w, col_x)
        z = z.reshape(N, KN, CH, CW)
        # =====================================================
    toc(True)

    d_z = z

    tic('backward')
    for i in xrange(itertimes):
        # =====================================================
        # backward
        d_z = d_z.reshape(KN, N * CH * CW)
        d_row_w, d_col_x = MulGate.backward(row_w, col_x, d_z)
        d_w = d_row_w.reshape(KN, KC, KH, KW)
        d_x = Conv2DGate.col2im_HW_naive(d_col_x, KH, KW, CH, CW, stride)
        # =====================================================
    toc(True)

    '''
    @forward, time: 0.015000s
    @backward, time: 0.098000s
    '''

if __name__ == '__main__':
    run()