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

def to_categorical(y, nb_class):
    n = np.shape(y)[0]
    new_y = np.zeros((n, nb_class), dtype=np.float64)
    new_y[range(n), y] = 1
    return new_y


def to_real(y):
    new_y = np.argmax(y, axis=1)
    return new_y

def split_training_data(x, y, split_ratio):
    """
    根据划分比例split_ratio，从训练集中划分出相应比例作为验证集
    :param training_x:
    :param training_y:
    :param split_ratio:
    :return:
    """
    n = np.shape(x)[0]
    idx = int(n * split_ratio)
    training_x = x[:idx]
    training_y = y[:idx]
    validation_x = x[idx:]
    validation_y = y[idx:]
    return training_x, training_y, validation_x, validation_y