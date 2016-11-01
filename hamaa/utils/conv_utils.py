# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: conv_utils.py
@time: 2016/10/17 21:50

将im2colutils模块与col2imutils模块封装一遍
"""


from ..ext import im2colutils, col2imutils, im2rowutils
import numpy as np


# ==========================================
# 以下为对外统一调用接口，使用C编写的Python扩展

def get_conv_shape(H, W, KH, KW, stride):
    return (H - KH) / stride + 1, (W - KW) / stride + 1

def im2col_HW(x, KH, KW, stride):
    # 注意，x！并不是一个真正的数组！
    # 如果传某数组的切片进C扩展函数中，数据会出错！
    # 数组切片是原始数组的视图。这意味着数据不会被复制，任何修改都会直接反映到源数组上
    x = np.array(x)
    return im2colutils.im2col_HW(x, KH, KW, stride)


def im2col_NCHW(x, KH, KW, stride):
    x = np.array(x)
    return im2colutils.im2col_NCHW(x, KH, KW, stride)


def im2row_HW(x, KH, KW, stride):
    x = np.array(x)
    return im2rowutils.im2row_HW(x, KH, KW, stride)


def im2row_NCHW(x, KH, KW, stride):
    x = np.array(x)
    return im2rowutils.im2row_NCHW(x, KH, KW, stride)


def col2im_HW(columnize_x, KH, KW, CH, CW, stride):
    # 注意，此处的columnize_x可能是某个数组的一块切片！并不是一个真正的数组！
    # 如果传某数组的切片进C扩展函数中，数据会出错！
    # 数组切片是原始数组的视图。这意味着数据不会被复制，任何修改都会直接反映到源数组上
    columnize_x = np.array(columnize_x)
    return col2imutils.col2im_HW(columnize_x, KH, KW, CH, CW, stride)


def col2im_NCHW(columnize_x, KH, KW, CH, CW, stride):
    columnize_x = np.array(columnize_x)
    return col2imutils.col2im_NCHW(columnize_x, KH, KW, CH, CW, stride)


# ==========================================
# 以下为Python版本的实现，从运行速度上不建议使用，
# 但是可以作为编写C语言版本的prototype。

def check_conv_shape(H, W, KH, KW, stride):
    """检测在步长为stride时能否使用KH*KW的w对H*W的x进行完整地卷积"""
    assert ((H - KH) % stride == 0 and (W - KW) % stride == 0)


def check_dtype(x):
    """检测数据类型是否为np.double类型"""
    assert (x.dtype == np.double)


def check_columnize_HW_shape(OH, OW, KH, KW, CH, CW):
    """检测columnize_x的形状是否与给定的参数KH/KW/CH/CW相符合"""
    assert (OH == KH * KW and OW == CH * CW)


def check_columnize_NCHW_shape(OH, OW, KH, KW, CH, CW):
    """检测columnize_x的形状是否与给定的参数KH/KW/CH/CW相符合"""
    assert (OH % (KH * KW) == 0 and OW % (CH * CW) == 0)


def im2col_HW_py(x, KH, KW, stride):
    """原生Python实现的im2col_HW"""
    H, W = x.shape
    CH = (H - KH) / stride + 1
    CW = (W - KW) / stride + 1
    OH = KH * KW
    OW = CH * CW
    columnize_x = np.empty((OH, OW))
    ocol = 0
    for crow in xrange(CH):
        for ccol in xrange(CW):
            row = crow * stride
            col = ccol * stride
            patch = x[row: row + KH, col: col + KW]
            columnize_x[:, ocol] = patch.reshape(1, OH)
            ocol += 1
    return columnize_x


def im2col_NCHW_py(x, KH, KW, stride):
    """原生Python实现的im2col_NCHW, 其中im2col_HW使用了C语言编写的扩展"""
    N, C, H, W = x.shape
    check_dtype(x)
    check_conv_shape(H, W, KH, KW, stride)
    CH = (H - KH) / stride + 1
    CW = (W - KW) / stride + 1
    OH = C * KH * KW
    OW = N * CH * CW
    columnize_x = np.empty((OH, OW), dtype=np.double)

    for i in xrange(N):
        for j in xrange(C):
            columnize_x[j * KH * KW:(j + 1) * KH * KW,
                        i * CH * CW:(i + 1) * CH * CW] = \
                im2colutils.im2col_HW(x[i][j], KH, KW, stride)
    return columnize_x


def col2im_HW_py(columnize_x, KH, KW, CH, CW, stride):
    """col2im过程"""
    OH, OW = columnize_x.shape
    check_columnize_HW_shape(OH, OW, KH, KW, CH, CW)
    H = (CH - 1) * stride + KH
    W = (CW - 1) * stride + KW
    x = np.zeros((H, W), dtype=columnize_x.dtype)
    for ocol in xrange(OW):
        col = ocol % CW * stride
        row = ocol / CW * stride
        x[row: row + KH, col: col + KW] = \
            columnize_x[:, ocol].reshape(KH, KW)
    return x


def col2im_NCHW_py(columnize_x, KH, KW, CH, CW, stride):
    OH, OW = columnize_x.shape
    check_dtype(columnize_x)
    check_columnize_NCHW_shape(OH, OW, KH, KW, CH, CW)
    N = OW / (CH * CW)
    C = OH / (KH * KW)
    H = (CH - 1) * stride + KH
    W = (CW - 1) * stride + KW
    x = np.zeros((N, C, H, W), dtype=columnize_x.dtype)
    for i in xrange(N):
        for j in xrange(C):
            orow = j * KH * KW
            ocol = i * CH * CW
            # 将col_x分割成多个patch进行col2im_HW任务来进行还原
            # 注意，此处的patch是一个切片！并不是一个真正的数组！
            # 数组切片是原始数组的视图。这意味着数据不会被复制，
            # 任何修改都会直接反映到源数组上！
            patch = columnize_x[orow: orow + KH * KW, ocol: ocol + CH * CW]
            x[i][j] = col2imutils.col2im_HW(patch, KH, KW, CH, CW, stride)
    return x
