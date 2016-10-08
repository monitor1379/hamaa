# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test5_Flatten.py
@time: 2016/9/27 23:34


"""

import numpy as np
import matplotlib.pyplot as plt

from core.layers import *
from datasets import datasets
from utils import image_utils


def test():
    ims = np.arange(36).reshape(2, 1, 3, 6)
    f = Flatten()
    fims = f.forward(ims)
    d_fims = fims
    d_ims = f.backward(d_fims)

    print ims
    print fims
    print d_ims


def run():
    test()


if __name__ == '__main__':
    run()