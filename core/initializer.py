# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: initializer.py
@time: 2016/9/20 9:59


"""

import numpy as np




class StandardNormalInitializer:
    """标准正太分布初始化器"""

    @staticmethod
    def initialize(shape):
        return np.random.randn(shape[0], shape[1])


class NormalSquareInitializer:
    """标准正太分布初始化器"""

    @staticmethod
    def initialize(shape):
        return np.random.randn(shape[0], shape[1]) / shape[0]


class ZeroInitializer:
    """全零初始化器"""

    @staticmethod
    def initialize(shape):
        return np.zeros(shape, dtype=np.float64)



class InitializerManager:
    """初始化管理器"""

    initializers = {
        'normal': StandardNormalInitializer,
        'normal_square': NormalSquareInitializer,
        'zero': ZeroInitializer,
    }

    @staticmethod
    def get(name):
        return InitializerManager.initializers[name]