# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: optimizer.py
@time: 2016/9/20 10:47


"""


class OptimizerManager:
    pass

class SGD:
    """使用随机梯度下降法的优化器"""

    def __init__(self, lr=0.1):
        self.lr = lr
