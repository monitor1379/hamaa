# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: losses.py
@time: 2016/9/11 12:07

损失函数
"""

import numpy as np


class MeanSquareError:

    @staticmethod
    def loss(y_real, y_pred):
        return np.mean(np.square(y_real - y_pred))

    @staticmethod
    def diff(y_real, y_pred):
        return - (y_real - y_pred) / y_real.shape[0]


_dict = {'mse': MeanSquareError,
         'mean_square_error': MeanSquareError,
         }


def get(identifier):
    return _dict.get(identifier)

