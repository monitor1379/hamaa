# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: np_utils.py
@time: 2016/9/11 14:35


"""

import numpy as np


def to_one_hot(y, nb_class):
    n = np.shape(y)[0]
    new_y = np.zeros((n, nb_class), dtype=np.float64)
    new_y[range(n), y] = 1
    return new_y


def to_categorical(y):
    new_y = np.argmax(y, axis=1)
    return new_y


def split_training_data(data, split_ratio):
    x, y = data
    n = np.shape(x)[0]
    idx = int(n * split_ratio)
    training_x = x[:idx]
    training_y = y[:idx]
    validation_x = x[idx:]
    validation_y = y[idx:]
    training_data = (training_x, training_y)
    validation_data = (validation_x, validation_y)
    return training_data, validation_data


# 给定函数f，自变量x以及对f的梯度df，求对x的梯度
def eval_numerical_gradient_array(f, x, df=None, verbose=False, h=1e-4):
    it = np.nditer(x, flags=['multi_index'])
    grad = np.zeros_like(x)
    while not it.finished:
        xi = it.multi_index
        old_val = x[xi]

        x[xi] = old_val + h
        y_top = f(x)

        x[xi] = old_val - h
        y_bottom = f(x)

        if df is not None:
            grad[xi] = np.sum((y_top - y_bottom) * df) / (2 * h)
        else:
            grad[xi] = np.sum(y_top - y_bottom) / (2 * h)

        if verbose:
            print it.multi_index, grad[xi]
        x[xi] = old_val
        it.iternext()

    return grad


def sum_abs_err(u, v):
    return np.sum(np.abs(u - v))