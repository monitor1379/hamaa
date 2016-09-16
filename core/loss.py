# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: loss.py
@time: 2016/9/11 12:07

损失函数
"""

import numpy as np
from utils import np_utils

class QuadraticLoss:
    """平方损失函数"""

    @staticmethod
    def loss(y, t):
        n = np.shape(y)[0]
        return 0.5 * np.sum(np.square(y - t)) / n
        # return 0.5 * np.sum(np.square(y - t))

    @staticmethod
    def diff_loss(y, t):
        n = np.shape(y)[0]
        return -(y - t) / n
        # return -(y - t)

class CategoricalCrossEntropy:
    """交叉熵损失函数"""

    @staticmethod
    def get_probs(t):
        z = np.exp(t)
        return z / np.sum(z, axis=1, keepdims=True)

    @staticmethod
    def loss(y, t):
        probs = CategoricalCrossEntropy.get_probs(t)
        real_y = np_utils.to_real(y)
        n = real_y.shape[0]
        cost = np.sum(-np.log(probs[range(n), real_y])) / n
        return cost


    @staticmethod
    def diff_loss(y, t):
        probs = CategoricalCrossEntropy.get_probs(t)
        real_y = np_utils.to_real(y)
        n = real_y.shape[0]
        probs[range(n), real_y] -= 1
        return probs
