# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test1_cnn.py
@time: 2016/10/20 23:49


"""


from hamaa.layers import Dense, Activation, Convolution2D, Flatten, MeanPooling2D
from hamaa.models import Sequential
from hamaa.datasets import datasets
from hamaa.utils import np_utils
from hamaa.optimizers import SGD
from hamaa.utils.time_utils import tic, toc

import numpy as np
import matplotlib.pyplot as plt


def run():
    training_data, test_data = datasets.load_mnist_data(nb_training=6000, nb_test=10000, preprocess=True, flatten=False)
    training_data, validation_data = np_utils.split_training_data(training_data, nb_validation=2000)

    print 'training_data:', training_data[0].shape
    print 'validation_data:', validation_data[0].shape
    print 'test_data:', test_data[0].shape

    model = Sequential()
    model.add(Convolution2D(nb_kernel=20, kernel_height=5, kernel_width=5, activation='tanh', input_shape=(1, 28, 28)))
    model.add(MeanPooling2D(pooling_size=(2, 2)))
    model.add(Convolution2D(nb_kernel=15, kernel_height=5, kernel_width=5, activation='tanh'))
    model.add(MeanPooling2D(pooling_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(output_dim=200, init='glorot_normal'))
    model.add(Activation('sigmoid'))
    model.add(Dense(output_dim=10, init='glorot_normal'))
    model.add(Activation('sigmoid'))

    model.set_loss(loss='categorical_crossentropy')
    model.set_optimizer(optimizer=SGD(lr=0.08, momentum=0.3, decay=1e-6))

    print model.summary()

    model.train(training_data=training_data,
                nb_epochs=50,
                mini_batch_size=10,
                verbose=2,
                validation_data=validation_data,
                log_epoch=1
                )

    print model.evaluate_accuracy(test_data[0], test_data[1])

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
            plt.subplot(row, col, i + 1)
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
            plt.subplot(row, col, i + 1)
            plt.axis('off')
            plt.imshow(model.layers[2].w[i][0], cmap='gray', interpolation='None')

        plt.show()
        # ===============================================================

    model.plot_training_iteration()


if __name__ == '__main__':
    run()



