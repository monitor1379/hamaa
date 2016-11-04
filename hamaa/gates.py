# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: gates.py
@time: 2016/9/11 9:01

计算单元
"""

import numpy as np


class MulGate:
    """乘法单元"""

    @staticmethod
    def forward(x, w):
        return np.dot(x, w)

    @staticmethod
    def backward(x, w, d_z):
        d_x = np.dot(d_z, np.transpose(w))
        d_w = np.dot(np.transpose(x), d_z)
        return d_x, d_w


class AddGate:
    """专用于n*m维与1*m维的加法单元"""

    @staticmethod
    def forward(x, b):
        return x + b

    @staticmethod
    def backward(x, b, d_z):
        d_x = np.array(d_z)
        d_b = np.array(d_z)
        for i in range(len(b.shape)):
            # 如果b在第i维被扩展了，则需要压缩回去
            if b.shape[i] == 1:
                d_b = np.sum(d_b, axis=i, keepdims=True)
        return d_x, d_b


class LinearGate:
    """线性计算单元"""

    @staticmethod
    def forward(x):
        return np.array(x)

    @staticmethod
    def backward(x, d_z):
        return np.ones_like(x) * d_z

class SigmoidGate:
    """sigmoid单元"""

    @staticmethod
    def forward(x):
        z = 1.0 / (1.0 + np.exp(-x))
        return z

    @staticmethod
    def backward(x, z, d_z):
        d_x = z * (1 - z) * d_z
        return d_x


class TanhGate:
    """tanh单元"""

    @staticmethod
    def forward(x):
        e1 = np.exp(x)
        e2 = np.exp(-x)
        return (e1 - e2) / (e1 + e2)

    @staticmethod
    def backward(x, z, d_z):
        d_x = (1 - z**2) * d_z
        return d_x


class ReLUGate:
    """relu单元"""
    @staticmethod
    def forward(x):
        z = np.array(x)
        z[z < 0] = 0
        return z

    @staticmethod
    def backward(x, z, d_z):
        d_x = np.ones_like(x)
        d_x[x < 0] = 0
        return d_x * d_z


class SoftmaxGate:
    """softmax单元"""

    @staticmethod
    def forward(x):
        ex = np.exp(x)
        # ex = np.exp(x - np.max(x, axis=1, keepdims=True))
        return ex / np.sum(ex, axis=1, keepdims=True)

    @staticmethod
    def backward(x, z, d_z):
        import warnings
        warnings.warn('Do not use SoftmaxGate.backward() when backpropagation!')
        n, m = x.shape
        d_x = np.empty_like(x)
        eyemat = np.eye(m)
        for i in xrange(n):
            z_i = z[i].reshape(1, m)
            d_x[i] = np.dot(d_z[i], (eyemat - z_i) * z_i.T)
        return d_x


if __name__ == '__main__':
    pass