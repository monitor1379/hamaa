# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test8_python_and_c_conv_speed.py
@time: 2016/10/2 22:54


"""

import numpy as np
# from __future__ import absolute_import
from datasets import datasets
from core.gates import *
import core.conv
from utils.time_utils import tic, toc
import matplotlib.pyplot as plt



def test_python_conv_speed():
    im = datasets.load_lena()[:, :, 0]
    print 'im.shape = {}'.format(im.shape)
    for i in range(10):
        tic('python')
        Conv2DGate.im2col(im, [2, 2], 1)
        toc()

def test_c_conv_speed():
    im = np.array(datasets.load_lena()[:, :, 0], dtype=np.double)
    kernel_shape = [2, 2]
    ch, cw = Conv2DGate.get_output_shape(im.shape, kernel_shape, 1, [0, 0])
    for i in range(10):
        tic('c')
        c_nim = core.conv.im2col(im, 2, 2, 1, ch, cw)
        toc()

def run():
    test_python_conv_speed()
    test_c_conv_speed()

if __name__ == '__main__':
    run()