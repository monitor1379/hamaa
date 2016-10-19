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

from test_lib import im2col
from hamaa.utils.time_utils import tic, toc


def naive_im2col(im, kernel_size, stride, conv_height, conv_width):
    """图片列化"""
    # 卷积核大小
    kernel_height, kernel_width = kernel_size
    # 如果没有给出卷积结果大小，则计算一遍
    # 将原图片"列化"后的大小
    im2col_height, im2col_width = (kernel_height * kernel_width, conv_height * conv_width)
    output = np.empty((im2col_height, im2col_width))
    # 列下标，代表输出结果的第几列
    output_idx = 0
    for row in xrange(0, conv_height):
        for col in xrange(0, conv_width):
            row_start_idx = row * stride
            col_start_idx = col * stride
            # 将卷积区域向量化成一列，保存在输出结果output中的第output_idx列中
            output[:, output_idx] = \
                im[row_start_idx:row_start_idx + kernel_height, col_start_idx:col_start_idx + kernel_width] \
                    .reshape(im2col_height)
            output_idx += 1
    return output


def test1():
    N, C, H, W = (100, 20, 28, 28)
    KN, KC, KH, KW = (15, C, 5, 5)

    x = np.arange(N * C * H * W).reshape(N, C, H, W).astype(np.double)
    w = np.arange(KN * KC * KH * KW).reshape(KN, KC, KH, KW).astype(np.double)
    b = np.zeros(KN)

    PH, PW = [0, 0]
    stride = 1

    CH = (H + 2 * PH - KH) / stride + 1
    CW = (W + 2 * PW - KW) / stride + 1

    col_x = np.empty(((C * KH * KW), (N * CH * CW)), dtype=np.double)
    row_w = np.empty((KN, (KC * KH * KW)), dtype=np.double)
    conv = np.empty((N, KN, CH, CW))

    verbose = True

    tic('totle')
    tic('im2col')
    # im2col
    for i in xrange(N):
        for j in xrange(C):
            # col_x[j*KH*KW:(j+1)*KH*KW, i*CH*CW:(i+1)*CH*CW] = naive_im2col(x[i][j], (KH, KW), stride, CH, CW)
            col_x[j*KH*KW:(j+1)*KH*KW, i*CH*CW:(i+1)*CH*CW] = c_im2col(x[i][j], (KH, KW), stride, CH, CW)
    toc(verbose)

    tic('reshape w')
    for i in xrange(KN):
        row_w[i] = w[i].reshape(1, -1)
    toc(verbose)


    print row_w.shape, col_x.shape
    tic('dot')
    row_conv = np.dot(row_w, col_x)
    toc(verbose)

    #col2im
    tic('col2im')
    for i in xrange(N):
        for j in xrange(KN):
            conv[i][j] = row_conv[j, i*CH*CW:(i+1)*CH*CW].reshape(CH, CW)
    toc(verbose)
    toc(verbose)


def test2_speed():
    H, W = (28, 28)
    KH, KW = (5, 5)

    x = np.arange(H * W).reshape(H, W).astype(np.double)
    w = np.arange(KH * KW).reshape(KH, KW).astype(np.double)

    PH, PW = [0, 0]
    stride = 1

    CH = (H + 2 * PH - KH) / stride + 1
    CW = (W + 2 * PW - KW) / stride + 1

    itertimes = 1000

    tic('naive im2col')
    for i in xrange(itertimes):
        naive_im2col(x, w.shape, stride, CH, CW)
    t = toc(True)
    print t / itertimes

    tic('im2col im2col')
    for i in xrange(itertimes):
        im2col_im2col(x, w, stride)
    t = toc(True)
    print t / itertimes

    '''
    @naive im2col, time: 5.286000s
    0.00528600001335

    @im2col im2col, time: 0.082000s
    8.20000171661e-05
    '''


def test3_im2col_NCHW():
    N, C, H, W = (10, 10, 28, 28)
    KN, KC, KH, KW = (1, C, 25, 25)

    x = np.arange(N * C * H * W).reshape(N, C, H, W).astype(np.double)
    w = np.arange(KN * KC * KH * KW).reshape(KN, KC, KH, KW).astype(np.double)
    b = np.zeros(KN)

    PH, PW = [0, 0]
    stride = 1

    CH = (H + 2 * PH - KH) / stride + 1
    CW = (W + 2 * PW - KW) / stride + 1


    verbose = True


    itertimes = 100


    tic('im2col_HW')
    for itertime in xrange(itertimes):
        col_x = np.empty(((C * KH * KW), (N * CH * CW)), dtype=np.double)
        for i in xrange(N):
            for j in xrange(C):
                col_x[j*KH*KW:(j+1)*KH*KW, i*CH*CW:(i+1)*CH*CW] = im2col.im2col_HW(x[i][j], KH, KW, stride)
    toc(verbose)


    tic('im2col_NCHW')
    for itertime in xrange(itertimes):
        aaa = im2col.im2col_NCHW(x, KH, KW, stride)
    toc(verbose)

    tic('im2col_NCHW_memcpy')
    for itertime in xrange(itertimes):
        bbb = im2col.im2col_NCHW_memcpy(x, KH, KW, stride)
    toc(verbose)


def run():
    # test1()
    # test2_speed()
    test3_im2col_NCHW()


if __name__ == '__main__':
    run()