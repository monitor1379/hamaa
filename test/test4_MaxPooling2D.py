# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test4_MaxPooling2D.py
@time: 2016/9/27 21:43


"""

import numpy as np
import matplotlib.pyplot as plt

from core.layers import *
from datasets import datasets
from utils import image_utils

def test1():
    im = datasets.load_lena()
    ims = im.reshape((1, im.shape[0], im.shape[1], im.shape[2]))
    ims = image_utils.batch_hwd2dhw(ims)
    layer = MaxPooling2D(input_shape=ims.shape[1:], pool_size=[2, 2])
    pims = layer.forward(ims)
    d_pims = pims
    d_ims = layer.backward(d_pims)

    print ims.shape
    print pims.shape
    print d_ims.shape

    d_ims = image_utils.batch_dhw2hwd(d_ims)
    pims = image_utils.batch_dhw2hwd(pims)

    plt.gray()
    plt.subplot(121)
    plt.imshow(d_ims[0][:, :, 0])
    plt.subplot(122)
    plt.imshow(pims[0][:, :, 0])
    plt.show()


def run():
    test1()


if __name__ == '__main__':
    run()