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
    x = np.random.randn(28, 28).astype(np.double)
    w = np.random.randn(5, 5).astype(np.double)

    dfh, dfw = Conv2DGate.get_conv_output_shape(x.shape, w.shape, 1, [0, 0])
    df = np.random.randn(dfh, dfw)

    grad_x = eval_numerical_gradient_array(lambda t: Conv2DGate.forward(t, w, mode='valid'), x, df)
    grad_w = eval_numerical_gradient_array(lambda t: Conv2DGate.forward(x, t, mode='valid'), w, df)

    dx, dw = Conv2DGate.backward(x, w, df)

    print grad_x
    print dx

    print sum_abs_err(grad_x, dx)
    print sum_abs_err(grad_w, dw)


def check_MaxPooling2DGate():
    x = np.random.randn(6, 6)
    pool_size = [2, 2]
    df = np.random.randn(x.shape[0] / pool_size[0], x.shape[1] / pool_size[1])

    grad_x = eval_numerical_gradient_array(lambda t: MaxPooling2DGate.forward(t, pool_size), x, df)
    dx = MaxPooling2DGate.backward(x, pool_size, df)

    print sum_abs_err(grad_x, dx)


def check_MeanPooling2DGate():
    x = np.random.randn(6, 6)
    pool_size = [2, 2]
    df = np.random.randn(x.shape[0] / pool_size[0], x.shape[1] / pool_size[1])

    grad_x = eval_numerical_gradient_array(lambda t: MeanPooling2DGate.forward(t, pool_size), x, df)
    dx = MeanPooling2DGate.backward(pool_size, df)

    print grad_x
    print dx

    print sum_abs_err(grad_x, dx)


def check_Convolution2D():
    np.seterr(all='raise')
    conv_layer = Convolution2D(input_shape=(2, 5, 5), nb_kernel=1, kernel_height=1, kernel_width=1, activation='tanh')
    conv_layer.w = np.ones_like(conv_layer.w)
    conv_layer.b = np.zeros_like(conv_layer.b)
    y = np.zeros((1, 1, 5, 5))

    x = np.arange(50).reshape(1, 2, 5, 5).astype(np.double)

    for i in range(100):
        t = conv_layer.forward(x)
        dy = -(y - t)
        dx = conv_layer.backward(dy)
        # conv_layer.update(0.1)
    print conv_layer.forward(x)


def run():
    check_LinearGate()
    # check_MultiGate()
    # check_AddGate()
    # check_SigmoidGate()
    # check_TanhGate()
    # check_ReLUGate()
    # check_Conv2DGate()
    # check_MaxPooling2DGate()
    # check_MeanPooling2DGate()
    # check_Convolution2D()

if __name__ == '__main__':
    run()