# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test2_check_Convolution2D_gradient.py
@time: 2016/10/21 10:40


"""


from hamaa.layers import Dense, Activation, Convolution2D
from hamaa.models import Sequential
from hamaa.datasets import datasets
from hamaa.utils import np_utils
from hamaa.optimizers import SGD
from hamaa.gates import *
from hamaa.utils.conv_utils import *
from hamaa.utils.time_utils import *
from hamaa.utils.np_utils import eval_numerical_gradient_array, sum_abs_err

import numpy as np


def test1():
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

    print d_w
    print grad_w
    print sum_abs_err(d_w, grad_w)







def run():
    test1()


if __name__ == '__main__':
    run()