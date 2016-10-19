# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: image_utils.py
@time: 2016/9/25 15:04


"""

import numpy as np


def chw2hwc(images):
    assert (np.ndim(images) >= 3)
    return np.swapaxes(np.swapaxes(images, -3, -2), -2, -1)


def hwc2chw(images):
    assert (np.ndim(images) >= 3)
    return np.swapaxes(np.swapaxes(images, -1, -2), -2, -3)

