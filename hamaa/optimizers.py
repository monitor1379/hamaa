# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: optimizers.py
@time: 2016/9/20 10:47


"""

from abc import ABCMeta, abstractmethod


class Optimizer(object):
    """优化器顶层抽象类"""

    __metaclass__ = ABCMeta

    opt_type = 'Optimizer'

    def __init__(self):
        pass

    @abstractmethod
    def update(self, **kwargs):
        pass


class SGD(Optimizer):

    def __init__(self, lr=0.01, momentum=0.0, decay=0.0, nesterov=False):
        """随机梯度下降法，支持动量参数，支持学习衰减率，支持Nesterov动量"""
        super(SGD, self).__init__()
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.nesterov = nesterov

    def update(self, params, grads):
        # print '======================='
        for param, grad in zip(params, grads):
            # print '-----------'
            # print param
            # print '-----------'
            # print grad
            param -= self.lr * grad


