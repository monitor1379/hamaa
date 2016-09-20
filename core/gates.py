# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: gates.py
@time: 2016/9/11 9:01

计算单元
"""


import numpy as np


class MulGate:
    """乘法单元"""

    @staticmethod
    def forward(w, x):
        return np.dot(x, w)

    @staticmethod
    def backward(w, x, d_z):
        d_w = np.dot(np.transpose(x), d_z)
        d_x = np.dot(d_z, np.transpose(w))
        # print d_w
        return d_w, d_x


class AddGate:
    """加法单元"""

    @staticmethod
    def forward(x, b):
        return x + b

    @staticmethod
    def backward(x, b, d_z):
        d_x = np.ones_like(x) * d_z
        d_b = np.ones((1, d_z.shape[0]), dtype=np.float64).dot(d_z)
        return d_x, d_b


class SigmoidGate:
    """sigmoid单元"""

    @staticmethod
    def forward(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def backward(x, d_z):
        a = SigmoidGate.forward(x)
        d_x = a * (1 - a) * d_z
        return d_x


class TanhGate:
    """tanh单元"""

    @staticmethod
    def forward(x):
        e1 = np.exp(x)
        e2 = np.exp(-x)
        return (e1 - e2) / (e1 + e2)

    @staticmethod
    def backward(x, d_z):
        a = TanhGate.forward(x)
        d_x = (1 - a**2) * d_z
        return d_x


class ReLUGate:
    """relu单元"""
    @staticmethod
    def forward(x):
        pass

    @staticmethod
    def backward(x, d_z):
        pass

class Conv2dGate:
    """二维卷积单元"""

    @staticmethod
    def forward(x):
        pass

    @staticmethod
    def backward(x, d_z):
        pass

