# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test1_Convolution2D.py
@time: 2016/9/25 15:15


"""


import numpy as np
import matplotlib.pyplot as plt
from core.layers import *
from core.gates import *
from datasets import datasets
from utils import image_utils
import theano


def test_lena():
    im = datasets.load_lena()
    ims = im.reshape((1, im.shape[0], im.shape[1], im.shape[2]))
    ims = image_utils.batch_hwd2dhw(ims)

    conv_layer = Convolution2D(input_shape=ims[0].shape, nb_kernel=1, kernel_height=1, kernel_width=2,
                               activation='relu')
    conv_layer.w[0][0] = np.array([-1, 1])
    conv_layer.w[0][1] = np.array([-1, 1])
    conv_layer.w[0][2] = np.array([-1, 1])
    cims = conv_layer.forward(ims)
    plt.gray()
    plt.subplot(121)
    plt.imshow(im)
    plt.subplot(122)
    plt.imshow(cims[0][0])
    plt.show()



def test_mnist():
    im = datasets.load_mnist_data(nb_training=1, nb_test=0)[0][0]
    ims = im.reshape((1, 1, im.shape[0], im.shape[1]))

    conv_layer = Convolution2D(input_shape=ims[0].shape, nb_kernel=1, kernel_height=1, kernel_width=2,
                               activation='linear')
    conv_layer.w[0][0] = np.array([-1, 1])
    cims = conv_layer.forward(ims)
    plt.gray()
    plt.subplot(121)
    plt.imshow(im)
    plt.subplot(122)
    plt.imshow(cims[0][0])
    plt.show()


def run():
    test_lena()
    # test_mnist()


if __name__ == '__main__':
    run()
