# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: example2_moons_nn.py
@time: 2016/10/9 17:45

Build a multilayer neural network to classify a dataset named "moons",
which is linear inseparable.
"""

from hamaa.datasets import datasets
from hamaa.layers import Dense, Activation
from hamaa.models import Sequential
from hamaa.optimizers import SGD
from hamaa.utils.np_utils import split_training_data


def run():
    model = Sequential()
    model.add(Dense(input_dim=2, output_dim=4, init='normal'))
    model.add(Activation('sigmoid'))
    model.add(Dense(output_dim=2))
    model.add(Activation('softmax'))
    model.set_objective('categorical_crossentropy')
    model.set_optimizer(SGD(lr=0.03, momentum=0.5))

    print model.summary()

    x, y = datasets.load_moons_data(nb_data=2000, noise=0.1)

    # split nine in tenth of original data as training data, and the rest as validation data
    training_data, validation_data = split_training_data(data=(x, y), split_ratio=0.9)

    # "verbose" means display mode of training information
    # "log_epoch" means display training information every log_epoch times
    model.train(training_data=training_data,
                nb_epochs=40,
                mini_batch_size=100,
                verbose=1,
                validation_data=validation_data,
                log_epoch=1)

    print 'test accuracy: ', model.evaluate_accuracy(x, y)

    # plot the prediction on training_data and validation_data
    model.plot_prediction(data=training_data)
    model.plot_prediction(data=validation_data)

    # plot a line chart about accuracy and loss with epoch.
    model.plot_training_iteration()


if __name__ == '__main__':
    run()
