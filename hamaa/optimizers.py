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
import numpy as np


class Optimizer(object):
    """优化器顶层抽象类"""

    __metaclass__ = ABCMeta

    opt_type = 'Optimizer'

    def __init__(self):
        pass

    @abstractmethod
    def update(self, **kwargs):
        pass

    @abstractmethod
    def next_iteration(self):
        pass


class SGD(Optimizer):

    def __init__(self, lr=0.01, momentum=0.0, decay=0.0):
        """随机梯度下降法，支持动量参数，支持学习衰减率"""
        super(SGD, self).__init__()
        self.lr = lr
        self.momentum = momentum
        self.decay = decay

        # 当前迭代期的学习率
        self.cur_lr = lr

        # 保存上次迭代期的参数改变量（速度）
        self.pre_velocities = []
        # 保存本次迭代期的参数该变量（速度）
        self.now_velocities = []

        # 当前迭代期数
        self.iterations = 0

    def update(self, params, grads):
        # 如果是第一轮，则初始化“上次参数改变量”为0
        if self.iterations == 0:
            self.pre_velocities = [0] * len(params)

        # 开始更新模型参数，同时保存“本次参数改变量”
        for param, grad, pre_velocity in zip(params, grads, self.pre_velocities):
            now_velocity = - self.cur_lr * grad + pre_velocity * self.momentum
            param += now_velocity
            self.now_velocities.append(now_velocity)

        self.next_iteration()

    def next_iteration(self):
        """更新优化器的超参数"""
        self.iterations += 1
        self.cur_lr = self.lr * 1.0 / (1.0 + self.decay * self.iterations)
        self.pre_velocities = self.now_velocities
        self.now_velocities = []



