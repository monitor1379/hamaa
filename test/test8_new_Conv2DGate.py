# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test8_new_Conv2DGate.py
@time: 2016/10/13 20:56


"""

import hamaa.clib.im2colutils as im2colutils
from hamaa.utils.time_utils import tic, toc
import numpy as np

class Conv2DGate:

    @staticmethod
    def forward(images, filters, fmt='NCHW'):
        pass

    @staticmethod
    def backward(images, filters, d_z, fmt='NCHW'):
        pass

    @staticmethod
    def im2col_HW(image, KH, KW, stride):
        return im2colutils.im2col_HW(image, KH, KW, stride)

    @staticmethod
    def im2col_NCHW_1(images, KH, KW, stride):
        assert (images.dtype == np.double)
        N, C, H, W = images.shape
        assert ((H - KH) % stride == 0)
        assert ((W - KW) % stride == 0)
        CH = (H - KH) / stride + 1
        CW = (W - KW) / stride + 1

        col_x = np.empty(((C * KH * KW), (N * CH * CW)), dtype=np.double)
        for i in xrange(N):
            for j in xrange(C):
                col_x[j * KH * KW:(j + 1) * KH * KW, i * CH * CW:(i + 1) * CH * CW] = \
                    Conv2DGate.im2col_HW(images[i][j], KH, KW, stride)
        return col_x

    @staticmethod
    def im2col_NCHW_2(images, KH, KW, stride):
        assert (images.dtype == np.double)
        N, C, H, W = images.shape
        assert ((H - KH) % stride == 0)
        assert ((W - KW) % stride == 0)
        return im2colutils.im2col_NCHW_memcpy(images, KH, KW, stride)

    @staticmethod
    def im2col_NCHW_3(images, KH, KW, stride):
        assert (images.dtype == np.double)
        N, C, H, W = images.shape
        assert ((H - KH) % stride == 0)
        assert ((W - KW) % stride == 0)
        return im2colutils.im2col_NCHW(images, KH, KW, stride)


def run():
    N, C, H, W = (2, 1, 4, 5)
    KN, KC, KH, KW = (1, C, 3, 3)

    images = np.arange(N * C * H * W).reshape(N, C, H, W).astype(np.double)
    filters = np.arange(KN * KC * KH * KW).reshape(KN, KC, KH, KW).astype(np.double)
    stride = 1

    verbose = True
    itertimes = 1

    tic('im2col_NCHW_1')
    for i in xrange(itertimes):
        col_x_1 = Conv2DGate.im2col_NCHW_1(images, KH, KW, stride)
    toc()

    col_x_2 = Conv2DGate.im2col_NCHW_2(images, KH, KW, stride)
    col_x_3 = Conv2DGate.im2col_NCHW_3(images, KH, KW, stride)

    print '============================'
    print col_x_1
    print '============================'
    print col_x_2
    print '============================'
    print col_x_3



if __name__ == '__main__':
    run()