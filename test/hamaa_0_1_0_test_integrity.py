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


def test_xor_data():
    x, y = dataset.load_xor_data()
    y = np_utils.to_categorical(y, 2)
    model = Network(layer_sizes=[2, 4, 2], activation_names=['sigmoid', 'sigmoid'], loss_name='quadratic_loss')
    model.train(training_data=(x, y), epochs=200, learning_rate=5, mini_batch_size=4, verbose=1, validation_data=(x, y))
    plt.subplot(211)
    plt.plot(model.train_accuracy)
    plt.plot(model.validation_accuracy)
    plt.subplot(212)
    plt.plot(model.train_loss)
    plt.plot(model.validation_loss)
    plt.show()


def test_moons_data():
    np.random.seed(0)
    x, y = dataset.load_moons_data(200, 0.2)
    cy = np_utils.to_categorical(y, 2)
    training_x, training_y, validation_x, validation_y = np_utils.split_training_data(x, cy, 0.8)
    training_data = (training_x, training_y)
    validation_data = (validation_x, validation_y)

    model = Network(layer_sizes=[2, 4, 2], activation_names=['sigmoid', 'sigmoid'], loss_name='quadratic_loss')
    model.train(training_data=training_data, epochs=2000, learning_rate=0.5, mini_batch_size=10, verbose=1, validation_data=validation_data)
    model.plot_training_iteration()
    model.plot_prediction(x, y)


def test_mnist_data():
    x, y = dataset.load_mnist_data(10000)
    cy = np_utils.to_categorical(y, 2)

    training_x, training_y, validation_x, validation_y = np_utils.split_training_data(x, cy, 0.8)
    training_data = (training_x, training_y)
    validation_data = (validation_x, validation_y)

    model = Network(layer_sizes=[2, 4, 2], activation_names=['sigmoid', 'sigmoid'], loss_name='quadratic_loss')
    model.train(training_data=training_data, epochs=2000, learning_rate=0.5, mini_batch_size=10, verbose=1, validation_data=validation_data)
    model.plot_training_iteration()
    model.plot_prediction(x, y)


def run():
    # test_xor_data()
    # test_moons_data()
    test_mnist_data()

if __name__ == '__main__':
    run()