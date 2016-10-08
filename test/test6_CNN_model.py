# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test6_CNN_model.py
@time: 2016/9/27 23:45


"""

import numpy as np
from core.layers import Convolution2D, MaxPooling2D, Dense, Activation, Flatten, MeanPooling2D
from core.models import Sequential
from core.optimizer import SGD
from datasets import datasets
from utils import np_utils
from utils.time_utils import tic, toc, T
import matplotlib.pyplot as plt

def run():
    # T.debug = False
    np.random.seed(0)
    training_x, training_y, test_x, test_y = datasets.load_mnist_data(nb_training=3000, nb_test=1000)

    training_x = training_x.reshape((training_x.shape[0], 1, training_x.shape[1], training_x.shape[2]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]))

    # =============
    # z-score
    training_x /= 255
    test_x /= 255

    training_x -= np.mean(training_x, axis=0)
    test_x -= np.mean(test_x, axis=0)

    training_x = np.true_divide(training_x, np.std(training_x, axis=0))
    training_x[np.isnan(training_x)] = 0.0

    test_x = np.true_divide(test_x, np.std(test_x, axis=0))
    test_x[np.isnan(test_x)] = 0.0
    # =============

    training_x, training_y, validation_x, validation_y = np_utils.split_training_data(training_x, training_y, 0.9)

    training_y = np_utils.to_categorical(training_y, 10)
    validation_y = np_utils.to_categorical(validation_y, 10)
    test_y = np_utils.to_categorical(test_y, 10)

    training_data = (training_x, training_y)
    validation_data = (validation_x, validation_y)

    model = Sequential()
    model.add(Convolution2D(input_shape=(1, 28, 28), nb_kernel=10, kernel_height=5, kernel_width=5, activation='tanh'))
    model.add(MeanPooling2D(input_shape=(10, 24, 24), pool_size=[2, 2]))
    model.add(Convolution2D(input_shape=(10, 12, 12), nb_kernel=20, kernel_height=5, kernel_width=5, activation='tanh'))
    model.add(MaxPooling2D(input_shape=(20, 8, 8), pool_size=[2, 2]))
    model.add(Flatten())
    model.add(Dense(input_dim=320, output_dim=10, init='normal_square'))
    model.add(Activation('tanh'))

    model.compile(optimizer=SGD(lr=0.02), loss='categorical_crossentropy')
    model.train(training_data=training_data, epochs=20, mini_batch_size=20, verbose=2, validation_data=validation_data,
                print_epoch=1)

    # =====================================================================
    evaluate_test = True
    if evaluate_test:
        print '测试准确率'
        print model.evaluate_accuracy(test_x, test_y)

    # =====================================================================
    # 可视化第一个卷积层的kernel
    plot_w = True
    if plot_w:
        plt.figure()
        print model.layers[0].w.shape
        N, _, _, _ = model.layers[0].w.shape
        row = int(np.ceil(np.sqrt(N)))
        col = row
        for i in range(N):
            plt.subplot(row, col, i+1)
            plt.axis('off')
            plt.imshow(model.layers[0].w[i][0], cmap='gray', interpolation='None')

        # 可视化第二个卷积层的kernel
        plt.figure()
        plt.axis('off')
        plt.gcf().set_size_inches(5, 5)
        print model.layers[2].w.shape
        N, _, _, _ = model.layers[2].w.shape
        row = int(np.ceil(np.sqrt(N)))
        col = row
        for i in range(N):
            plt.subplot(row, col, i+1)
            plt.axis('off')
            plt.imshow(model.layers[2].w[i][0], cmap='gray',  interpolation='None')

        plt.show()
    # ===============================================================
    # 可视化训练过程
    plot_training_iteration = True
    if plot_training_iteration:
        model.plot_training_iteration()


if __name__ == '__main__':
    run()
