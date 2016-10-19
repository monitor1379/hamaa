# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test10_pad.py
@time: 2016/10/16 9:59


"""

import numpy as np
from hamaa.utils.np_utils import padding


def test1():
    x = np.ones((2, 2, 2, 2))

    print x

    top, bottom, left, right = (1, 2, 3, 4)
    npad = ((0, 0), (0, 0), (top, bottom), (left, right))
    px = np.pad(x, pad_width=npad, mode='constant', constant_values=0)
    print px

    print (padding(x, [1, 2, 3, 4]) == px).all()


def run():
    test1()


if __name__ == '__main__':
    run()