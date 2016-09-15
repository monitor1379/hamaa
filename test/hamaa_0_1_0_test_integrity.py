# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: hamaa_0_1_0_test_integrity.py
@time: 2016/9/11 9:02


"""

import numpy as np
from dataset import dataset
from core.model import Network
from utils import np_utils
import matplotlib.pyplot as plt
from core.loss import CategoricalCrossEntropy


def test_xor_data():
    np.random.seed(0)
    x, y = dataset.load_xor_data()
    cy = np_utils.to_categorical(y, 2)
    training_x, training_y, validation_x, validation_y = x, cy, x, cy
    training_data = (training_x, training_y)
    validation_data = (validation_x, validation_y)

    model = Network(layer_sizes=[2, 4, 2], activation_names=['sigmoid', 'sigmoid'], loss_name='categorical_crossentropy')
    model.train(training_data=training_data, epochs=1000, learning_rate=1, reg_lambda=0.0, mini_batch_size=4,
                verbose=1, validation_data=validation_data, print_epoch=100)
    model.plot_training_iteration()
    model.plot_prediction(x, y)


def test_moons_data():
    np.random.seed(0)
    x, y = dataset.load_moons_data(200, 0.2)
    cy = np_utils.to_categorical(y, 2)
    training_x, training_y, validation_x, validation_y = np_utils.split_training_data(x, cy, 0.8)
    training_data = (training_x, training_y)
    validation_data = (validation_x, validation_y)

    model = Network(layer_sizes=[2, 4, 2], activation_names=['sigmoid', 'sigmoid'], loss_name='categorical_crossentropy')
    model.train(training_data=training_data, epochs=100, learning_rate=1, reg_lambda=0.0001, mini_batch_size=5, verbose=1, validation_data=validation_data, print_epoch=10)
    model.plot_training_iteration()
    model.plot_prediction(x, y)


def test_mnist_data():
    # 加载数据
    training_x, training_y, test_x, test_y = dataset.load_mnist_data(nb_training=60000, nb_test=10000)
    # 数据格式预处理
    training_n, h, w = training_x.shape
    test_n = test_x.shape[0]
    training_x = training_x.reshape(training_n, h * w)
    test_x = test_x.reshape(test_n, h * w)

    training_x[training_x > 0] = 1
    test_x[test_x > 0] = 1
    training_y = np_utils.to_categorical(training_y, 10)
    test_y = np_utils.to_categorical(test_y, 10)

    training_x, training_y, validation_x, validation_y = np_utils.split_training_data(training_x, training_y, 0.90)
    training_data = (training_x, training_y)
    validation_data = (validation_x, validation_y)
    test_data = (test_x, test_y)
    print 'training data: \t\t', training_x.shape, training_y.shape
    print 'validation data: \t', validation_x.shape, validation_y.shape
    print 'test data: \t\t\t', test_x.shape, test_y.shape


    model = Network(layer_sizes=[h * w, 300, 10], activation_names=['sigmoid', 'sigmoid'], loss_name='categorical_crossentropy')
    model.train(training_data=training_data, epochs=50, learning_rate=0.025, reg_lambda=0.00001, mini_batch_size=10, verbose=1, validation_data=validation_data, print_epoch=1)
    print '测试集准确率：', model.evaluate_accuracy(test_x, test_y)
    model.plot_training_iteration()




def run():
    # test_xor_data()
    # test_moons_data()
    test_mnist_data()

if __name__ == '__main__':
    run()