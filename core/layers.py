# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: layers.py
@time: 2016/9/20 9:36


"""

from gates import *
from initializer import InitializerManager

class Dense:
    """全连接层"""

    layer_type = 'Dense'

    def __init__(self, input_dim, output_dim, init, layer_name=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init = init

        self.layer_name = layer_name

        self.w = InitializerManager.get(init).initialize(shape=[input_dim, output_dim])
        self.b = InitializerManager.get(init).initialize(shape=[1, output_dim])

        # 保存前向计算的中间结果，加快计算速度
        self.input = 0
        self.mul = 0
        self.add = 0
        self.output = 0

        # 保存后向传播的中间结果，加快计算速度
        self.d_input = 0
        self.d_w = 0
        self.d_b = 0
        self.d_mul = 0
        self.d_add = 0
        self.d_output = 0

        # 表示该层具有参数
        self.updatable = True

    def forward(self, x):
        self.input = x
        self.mul = MulGate.forward(self.w, self.input)
        self.add = AddGate.forward(self.mul, self.b)
        self.output = self.add
        return self.add

    def backward(self, d_output):
        self.d_output = d_output
        self.d_mul, self.d_b = AddGate.backward(self.mul, self.b, self.d_output)
        self.d_w, self.d_input = MulGate.backward(self.w, self.input, self.d_mul)
        return self.d_input

    def update(self, lr):
        self.w -= lr * self.d_w
        self.b -= lr * self.d_b


class Activation:
    """激活层"""

    layer_type = 'Activation'

    def __init__(self, act_type, layer_name=None):
        self.act_type = act_type

        self.layer_name = layer_name
        self.act_obj = ActivationManager.get(act_type)

        self.input = 0
        self.output = 0

        self.d_input = 0
        self.d_output = 0

        self.updatable = False

    def forward(self, x):
        self.input = x
        self.output = self.act_obj.forward(x)
        return self.output

    def backward(self, d_output):
        self.d_input = self.act_obj.backward(self.input, d_output)
        return self.d_input




class ActivationManager:
    """激活函数管理类"""

    activations = {
            'sigmoid': SigmoidGate,
            'tanh': TanhGate,
            'relu': ReLUGate,
    }

    @staticmethod
    def get(name):
        return ActivationManager.activations[name]