# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: initializations.py
@time: 2016/9/20 9:59

初始化器
"""

import numpy as np


def zeros(shape, dtype=np.double):
    return np.zeros(shape, dtype)


def uniform(shape, scale=0.05):
    return np.random.uniform(low=-scale, high=scale, size=shape)


def normal(shape, scale=0.05):
    return np.random.normal(loc=0.0, scale=scale, size=shape)


def glorot_normal(shape):
    if len(shape) == 2:
        nb_in, nb_out = shape[0], shape[1]
    # 如果是4维NCHW数据格式
    elif len(shape) == 4:
        nb_in, nb_out = shape[2], shape[3]
    else:
        raise Exception("Invalid shape: shape's length must be 2D!")
    # 标准差
    s = np.sqrt(2.0 / (nb_in + nb_out))
    return np.random.normal(loc=0.0, scale=s, size=shape)


_dict = {'zeros': zeros,
         'uniform': uniform,
         'normal': normal,
         'glorot_normal': glorot_normal,
         }


def get(identifier):
    return _dict.get(identifier)
