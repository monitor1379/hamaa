# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test1_new_conv_way.py
@time: 2016/10/8 19:34


"""

import numpy as np
from hamaa.utils.time_utils import tic, toc


def run():
    N, C, H, W = (100, 20, 28, 28)
    KN, KC, KH, KW = (15, C, 5, 5)

    x = np.arange(N * C * H * W).reshape(N, C, H, W).astype(np.double)
    w = np.arange(KN * KC * KH * KW).reshape(KN, KC, KH, KW).astype(np.double)
    b = np.zeros(KN)

    PH, PW = [0, 0]
    stride = 1

    CH = (H + 2 * PH - KH) / stride + 1
    CW = (W + 2 * PW - KW) / stride + 1

    IH = KH * KW
    IW = CH * CW


    col_x = np.empty(((C * KH * KW), (N * CH * CW)), dtype=np.double)
    row_w = np.empty((KN, (KC * KH * KW)), dtype=np.double)
    conv = np.empty((N, KN, CH, CW))

    tic()
    # im2col
    for i in xrange(N):
        for j in xrange(C):
            col_x[j*KH*KW:(j+1)*KH*KW, i*CH*CW:(i+1)*CH*CW] = Conv2DGate.c_im2col(x[i][j], (KH, KW), stride, CH, CW)

    for i in xrange(KN):
        row_w[i] = w[i].reshape(1, -1)
    row_conv = np.dot(row_w, col_x)

    #col2im
    for i in xrange(N):
        for j in xrange(KN):
            conv[i][j] = row_conv[j, i*CH*CW:(i+1)*CH*CW].reshape(CH, CW)
    toc()

if __name__ == '__main__':
    run()