# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: loss.py
@time: 2016/9/11 12:07


"""

import numpy as np


class QuadraticLoss:
    """平方损失函数"""

    @staticmethod
    def loss(y, t):
        n = np.shape(y)[0]
        return 0.5 * np.sum(np.square(y - t)) / n

    @staticmethod
    def diff_loss(y, t):
        n = np.shape(y)[0]
        return -(y - t) / n
