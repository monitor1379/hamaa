# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: activations.py
@time: 2016/10/9 20:29


"""

from .gates import SigmoidGate, TanhGate, ReLUGate, LinearGate, SoftmaxGate


_dict = {'sigmoid': SigmoidGate,
         'tanh': TanhGate,
         'relu': ReLUGate,
         'linear': LinearGate,
         'softmax': SoftmaxGate,
         }


def get(identifier):
    return _dict.get(identifier)

