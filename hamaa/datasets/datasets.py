# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: datasets.py
@time: 2016/9/11 9:00

数据集加载文件
"""

import gzip
import os
import urllib

import numpy as np
from PIL import Image

from .mnist import mnist_decoder
from ..utils import np_utils


def load_or_data(one_hot=True):
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    if one_hot:
        y = np_utils.to_one_hot(y, 2)
    return x, y


def load_and_data(one_hot=True):
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    if one_hot:
        y = np_utils.to_one_hot(y, 2)
    return x, y


def load_xor_data(one_hot=True):
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    if one_hot:
        y = np_utils.to_one_hot(y, 2)
    return x, y



def load_moons_data(nb_data, noise, one_hot=True, shuffle=True):
    """Make two interleaving half circles.

       A simple toy dataset to visualize clustering and classification
       algorithms.

       # Argument:
           nb_data: int, optional (default=100).
           The total number of points generated.

           noise: double or None (default=None).
           Standard deviation of Gaussian noise added to the data.


       # Note:
           x: array of shape [nb_data, 2].

           y: array of shape [nb_data].
           The integer labels (0 or 1) for class membership of each sample.
       """

    n_samples_out = nb_data // 2
    n_samples_in = nb_data - n_samples_out

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - .5

    x = np.vstack((np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y))).T
    y = np.hstack([np.zeros(n_samples_in, dtype=np.intp),
                   np.ones(n_samples_out, dtype=np.intp)])

    # 打乱顺序
    if shuffle:
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        x = x[idx]
        y = y[idx]

    # 加上方差为noise的服从正态分布的噪声数据
    x += np.random.normal(scale=noise, size=x.shape)
    if one_hot:
        y = np_utils.to_one_hot(y, 2)
    return x, y


def download_mnist_data():
    filenames = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                 "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]

    # 设置下载文件的本地存放文件夹，存放地点为hamaa/datasets/mnist/gz
    module_path = os.path.dirname(__file__)
    mnist_gz_dir = module_path + os.sep + 'mnist' + os.sep + 'gz' + os.sep

    # 检查是否缺失gz压缩包
    miss_gz_file = False
    for filename in filenames:
        file_path = mnist_gz_dir + filename
        if not os.path.exists(file_path):
            miss_gz_file = True
            break

    # 如果不存在gz文件夹或者缺失了某个压缩包，则重新下载mnist的所有gz压缩包
    if not os.path.exists(mnist_gz_dir) or miss_gz_file:
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

    # 检查是否缺失解压后的数据文件
    miss_bin_file = False
    for filename in filenames:
        file_path = mnist_bin_dir + filename.split('.')[0]
        if not os.path.exists(file_path):
            miss_bin_file = True
            break

    if not os.path.exists(mnist_bin_dir) or miss_bin_file:
        if not os.path.exists(mnist_bin_dir):
            os.mkdir(mnist_bin_dir)

        # 开始解压
        for filename in filenames:
            print 'unzip ' + filename + ' ...'
            fn = filename.split()
            in_file = gzip.GzipFile(mnist_gz_dir + filename, 'rb')
            out_file = open(mnist_bin_dir + filename.split('.')[0], 'wb')
            out_file.write(in_file.read())
            in_file.close()
            out_file.close()


def load_mnist_data(nb_training, nb_test, preprocess=False, flatten=True):
    # 自动检查数据，如果数据文件不存在则会先自动下载
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
    im = Image.open(image_path)
    return np.array(im)
