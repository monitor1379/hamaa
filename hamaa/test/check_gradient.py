# encoding: utf-8
"""
@author: monitor1379
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: check_gradient.py
@time: 2016/10/4 9:59


"""


import numpy as np
import matplotlib.pyplot as plt



from hamaa.gates import *
from hamaa.layers import *
from hamaa.utils.np_utils import eval_numerical_gradient_array, sum_abs_err



def check_LinearGate():
    x = np.random.randn(10, 4).astype(np.double)
    dy = np.random.randn(10, 4).astype(np.double)
    grad_x = eval_numerical_gradient_array(LinearGate.forward, x, dy)
    dx = LinearGate.backward(x, dy)

    print grad_x
    print dx
    print np.sum(np.abs(grad_x - dx))


def check_MultiGate():
    x = np.random.randn(10, 4)
    w = np.random.randn(4, 6)
    df = np.random.randn(10, 6)

    grad_x = eval_numerical_gradient_array(lambda a: MulGate.forward(w, a), x, df)
    grad_w = eval_numerical_gradient_array(lambda b: MulGate.forward(b, x), w, df)
    dw, dx = MulGate.backward(w, x, df)
    print np.sum(np.abs(grad_x - dx))
    print np.sum(np.abs(grad_w - dw))


def check_AddGate():
    x = np.random.randn(10, 3)
    # b = np.random.randn(1, 3)
    b = np.random.randn(1, 1)
    df = np.random.randn(10, 3)
    grad_x = eval_numerical_gradient_array(lambda t: AddGate.forward(t, b), x, df)
    grad_b = eval_numerical_gradient_array(lambda t: AddGate.forward(x, t), b, df)
    dx, db = AddGate.backward(x, b, df)
    print sum_abs_err(grad_x, dx)
    print sum_abs_err(grad_b, db)


def check_SigmoidGate():
    x = np.random.randn(10, 4)
    df = np.random.randn(10, 4)

    grad_x = eval_numerical_gradient_array(SigmoidGate.forward, x, df)
    dx = SigmoidGate.backward(x, df)

    print sum_abs_err(grad_x, dx)


def check_TanhGate():
    x = np.random.randn(10, 4)
    df = np.random.randn(10, 4)

    grad_x = eval_numerical_gradient_array(TanhGate.forward, x, df)
    dx = TanhGate.backward(x, df)

    print sum_abs_err(grad_x, dx)


def check_ReLUGate():
    x = np.random.randn(10, 4)
    df = np.random.randn(10, 4)

    grad_x = eval_numerical_gradient_array(ReLUGate.forward, x, df)
    dx = ReLUGate.backward(x, df)

    print sum_abs_err(grad_x, dx)


def check_Convolution2D():
    N, C, H, W = 1, 1, 4, 5
    KN, KC, KH, KW = 1, C, 3, 3
    stride = 1

    x = np.random.rand(N, C, H, W)
    w = np.random.rand(KN, KC, KH, KW)

    conv_layer = Convolution2D(nb_kernel=KN, kernel_height=KH, kernel_width=KW, input_shape=(C, H, W))
    conv_layer.build()

    # =========================================================
    # test x
    z = conv_layer.forward(x)
    d_z = z
    d_x = conv_layer.backward(d_z)
    grad_x = eval_numerical_gradient_array(conv_layer.forward, x, d_z)

    # print d_x
    # print grad_x
    print sum_abs_err(d_x, grad_x)

    # =========================================================
    # test w
    z = conv_layer.forward(x)
    d_z = z
    conv_layer.backward(d_z)
    d_w = conv_layer.grads[0]

    def w_forword(layer, xx, ww):
        layer.w = ww
        return layer.forward(xx)

    grad_w = eval_numerical_gradient_array(lambda t: w_forword(conv_layer, x, t), w, d_z)

    # print d_w
    # print grad_w
    print sum_abs_err(d_w, grad_w)


def check_MeanPooling2D():
    layer = MeanPooling2D(pooling_size=[2, 2], input_shape=(1, 6, 6))
    layer.build()

    x = np.arange(36).reshape(1, 1, 6, 6).astype(np.double)
    # print x

    z = layer.forward(x)
    d_z = z
    d_x = layer.backward(d_z)
    grad_x = eval_numerical_gradient_array(layer.forward, x, d_z)
    # print d_x
    # print grad_x

    print sum_abs_err(d_x, grad_x)

def run():
    # check_LinearGate()
    # check_MultiGate()
    # check_AddGate()
    # check_SigmoidGate()
    # check_TanhGate()
    # check_ReLUGate()
    check_Convolution2D()
    # check_MeanPooling2D()

if __name__ == '__main__':
    run()