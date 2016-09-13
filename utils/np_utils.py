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
