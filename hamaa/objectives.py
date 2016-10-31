# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: objectives.py
@time: 2016/9/11 12:07

损失函数
"""

import numpy as np
from .utils import np_utils
from .gates import SoftmaxGate

class MeanSquareError:

    @staticmethod
    def loss(y_real, y_pred):
        return np.mean(np.square(y_real - y_pred))

    @staticmethod
    def diff(y_real, y_pred):
        return - (y_real - y_pred) / y_real.shape[0]



class CategoricalCrossEntropy:

    @staticmethod
    def loss(y_real, y_pred):
        probs = SoftmaxGate.forward(y_pred)
        categorical_y_real = np_utils.to_categorical(y_real)
        n = categorical_y_real.shape[0]
        cost = np.sum(-np.log(probs[range(n), categorical_y_real])) / n
        return cost

    @staticmethod
    def diff(y_real, y_pred):
        probs = SoftmaxGate.forward(y_pred)
        categorical_y_real = np_utils.to_categorical(y_real)
        n = categorical_y_real.shape[0]
        probs[range(n), categorical_y_real] -= 1
        return probs


_dict = {'mse': MeanSquareError,
         'mean_square_error': MeanSquareError,
         'categorical_crossentropy': CategoricalCrossEntropy,
         }


def get(identifier):
    return _dict.get(identifier)

