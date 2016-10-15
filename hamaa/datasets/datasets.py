# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: datasets.py
@time: 2016/9/11 9:00

数据集加载文件
"""

import os

import numpy as np
import skimage.io
import sklearn.datasets

from .mnist import mnist_decoder
from ..utils import np_utils


def load_or_data():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    return x, y


def load_and_data():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    return x, y


def load_xor_data():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    return x, y


def load_moons_data(nb_data, noise):
    return sklearn.datasets.make_moons(nb_data, noise=noise)


def load_mnist_data(nb_training, nb_test, preprocess=False, flatten=True):
    training_x = mnist_decoder.load_train_images(num_data=nb_training)
    training_y = mnist_decoder.load_train_labels(num_data=nb_training)
    test_x = mnist_decoder.load_test_images(num_data=nb_test)
    test_y = mnist_decoder.load_test_labels(num_data=nb_test)

    training_y = np_utils.to_one_hot(training_y, 10)
    test_y = np_utils.to_one_hot(test_y, 10)

    if preprocess:
        training_x /= 255.
        test_x /= 255.

    if flatten:
        training_x = training_x.reshape(training_x.shape[0], 784)
        test_x = test_x.reshape(test_x.shape[0], 784)

    return (training_x, training_y), (test_x, test_y)


def load_lena():
    module_path = os.path.dirname(__file__)
    image_path = module_path + os.sep + 'images' + os.sep + 'lena.jpg'
    im = skimage.io.imread(image_path)
    return im