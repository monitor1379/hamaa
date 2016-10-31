# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: test_gates.py
@time: 2016/10/27 23:40


"""
from hamaa.gates import *
import numpy as np


def test_forward():
    a = np.random.rand(4, 5)
    b = np.random.rand(5, 6)
    m = MulGate.forward(a, b)
    t = np.dot(a, b)
    assert (m == t).all()


def test_backward():
    a = np.random.rand(4, 5)
    b = np.random.rand(5, 6)
    m = MulGate.forward(a, b)
    t = np.dot(a, b)
    assert (m == t).all()

