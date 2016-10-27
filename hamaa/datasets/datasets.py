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
import urllib
import gzip

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


def download_mnist_data():
    filenames = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                 "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    
    # 创建本地存放文件夹，存放地点为hamaa/datasets/mnist/gz
    module_path = os.path.dirname(__file__)
    mnist_gz_dir = module_path + os.sep + 'mnist' + os.sep + 'gz' + os.sep

    # 如果不存在gz文件夹，则下载mnist的gz压缩包
    if not os.path.exists(mnist_gz_dir):
        os.mkdir(mnist_gz_dir)
        # 下载压缩包
        addr = 'http://yann.lecun.com/exdb/mnist/'
        for filename in filenames:
            url = addr + filename
            print 'downloading ' + filename + ' from ' + url
            urllib.urlretrieve(url, mnist_gz_dir + filename)

    # 如果不存在bin文件夹，则解压mnist的gz压缩包
    # 创建解压文件夹，存放地点为hamaa/datasets/mnist/bin
    mnist_bin_dir = module_path + os.sep + 'mnist' + os.sep + 'bin' + os.sep
    if not os.path.exists(mnist_bin_dir):
        os.mkdir(mnist_bin_dir)

        # 开始解压
        for filename in filenames:
            print 'un gzip ' + filename + ' ...'
            fn = filename.split()
            in_file = gzip.GzipFile(mnist_gz_dir + filename, 'rb')
            out_file = open(mnist_bin_dir + filename.split('.')[0], 'wb')
            out_file.write(in_file.read())
            in_file.close()
            out_file.close()


def load_mnist_data(nb_training, nb_test, preprocess=False, flatten=True):
    # 如果数据文件不存在则会先自动下载
    download_mnist_data()
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
    else:
        training_x = training_x.reshape((training_x.shape[0], 1, training_x.shape[1], training_x.shape[2]))
        test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]))

    return (training_x, training_y), (test_x, test_y)


def load_lena():
    module_path = os.path.dirname(__file__)
    image_path = module_path + os.sep + 'images' + os.sep + 'lena.jpg'
    im = skimage.io.imread(image_path)
    return im