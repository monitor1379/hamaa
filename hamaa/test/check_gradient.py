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

def check_Conv2DGate():
    N, C, H, W = (1, 1, 4, 5)
    KN, KC, KH, KW = (1, C, 3, 3)
    stride = 1
    CH = (H - KH) / stride + 1
    CW = (W - KW) / stride + 1

    x = np.arange(N * C * H * W).reshape(N, C, H, W).astype(np.double)
    w = np.ones((KN, KC, KH, KW)).astype(np.double)

    z, cache = Conv2DGate.forward(x, w, stride)
    d_z = np.random.rand(N, KN, CH, CW)
    # d_z = np.random.rand(CH, CW)
    d_x, d_w = Conv2DGate.backward(x, w, stride, d_z, cache=cache)

    grad_x = eval_numerical_gradient_array(lambda t: Conv2DGate.forward(t, w, stride)[0], x, d_z)
    grad_w = eval_numerical_gradient_array(lambda t: Conv2DGate.forward(x, t, stride)[0], w, d_z)

    print sum_abs_err(d_x, grad_x)
    print sum_abs_err(d_w, grad_w)

    # ========================================================

    x = x[0][0]
    w = w[0][0]

    z, cache = Conv2DGate.forward(x, w, stride, fmt='HW')
    d_z = np.random.rand(CH, CW)
    d_x, d_w = Conv2DGate.backward(x, w, stride, d_z, cache=cache, fmt='HW')

    grad_x = eval_numerical_gradient_array(lambda t: Conv2DGate.forward(t, w, stride, fmt='HW')[0], x, d_z)
    grad_w = eval_numerical_gradient_array(lambda t: Conv2DGate.forward(x, t, stride, fmt='HW')[0], w, d_z)

    print sum_abs_err(d_x, grad_x)
    print sum_abs_err(d_w, grad_w)


def run():
    # check_LinearGate()
    # check_MultiGate()
    # check_AddGate()
    # check_SigmoidGate()
    # check_TanhGate()
    # check_ReLUGate()
    check_Conv2DGate()

if __name__ == '__main__':
    run()