# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test3_PoolingGate.py
@time: 2016/9/27 16:31


"""

import numpy as np
from core.gates import *


def test_mean_pooling_gate():
    im = np.arange(36).reshape((6, 6))
    pim = MeanPooling2DGate.forward(im, pool_size=[3, 3])
    d_im = MeanPooling2DGate.backward(pool_size=[3, 3], d_pim=pim)
    print im
    print pim
    print d_im


def test_max_pooling_gate():
    # im = np.arange(36).reshape((6, 6))
    im = np.array([[9, 1, 2, 3],
                   [2, 3, 8, 3],
                   [11, 2, 4, 32],
                   [5, 4, 3, 1]])

    pim = MaxPooling2DGate.forward(im, pool_size=[2, 2])
    d_pim = np.array([[1, 2],
                      [3, 4]])
    d_im = MaxPooling2DGate.backward(im, pool_size=[2, 2], d_pim=d_pim)
    print im
    print d_im

def run():
    # test_mean_pooling_gate()
    test_max_pooling_gate()

if __name__ == '__main__':
    run()