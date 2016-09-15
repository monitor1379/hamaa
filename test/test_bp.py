# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test_bp.py
@time: 2016/9/13 23:44


"""

import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def test1():
    x = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    y = np.array([[1, 0],
                  [0, 1],
                  [0, 1],
                  [1, 0]])

    np.random.seed(1)
    from utils import np_utils
    from dataset import dataset
    training_x, training_y, test_x, test_y = dataset.load_mnist_data(nb_training=1000, nb_test=1000)
    # 数据格式预处理
    training_n, h, w = training_x.shape
    test_n = test_x.shape[0]
    training_x = training_x.reshape(training_n, h * w)
    test_x = test_x.reshape(test_n, h * w)

    training_x[training_x > 0] = 1
    test_x[test_x > 0] = 1
    training_y = np_utils.to_categorical(training_y, 10)
    test_y = np_utils.to_categorical(test_y, 10)
    x = training_x
    y = training_y

    n = x.shape[0]
    layer_sizes = [784, 10, 10]
    w1 = np.random.randn(layer_sizes[0], layer_sizes[1]) / 100
    w2 = np.random.randn(layer_sizes[1], layer_sizes[2]) / 100
    b1 = np.random.randn(1, layer_sizes[1])
    b2 = np.random.randn(1, layer_sizes[2])

    epochs = 10000
    print_epoch = 1000

    for epoch in xrange(epochs):

        act0 = x
        act1 = sigmoid(np.dot(act0, w1) + b1)
        act2 = sigmoid(np.dot(act1, w2) + b2)
        error = y - act2

        if (epoch % print_epoch) == 0:
            print "Error:" + str(np.mean(np.abs(error))), np.sum(np.abs(error))

        d_mul2 = - (y - act2) * act2 * (1 - act2)
        d_w2 = act1.T.dot(d_mul2) / n
        d_b2 = np.ones((n, 1)).T.dot(d_mul2) / n

        d_mul1 = d_mul2.dot(w2.T) * act1 * (1 - act1)
        d_w1 = act0.T.dot(d_mul1)
        d_b1 = np.ones((n, 1)).T.dot(d_mul1)

        w2 -= d_w2
        b2 -= d_b2

        w1 -= d_w1
        b1 -= d_b1

    t = act2.argmax(axis=1)
    r = np_utils.to_real(training_y)
    print t
    print r
    print np.sum(t == r) * 1.0 / n

def run():
    test1()


if __name__ == '__main__':
    run()