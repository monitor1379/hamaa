# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test4_get.py
@time: 2016/10/9 17:45


"""

from hamaa.layers import Dense, Activation
from hamaa.models import Sequential
from hamaa.datasets import datasets
from hamaa.utils import np_utils
from hamaa.optimizers import SGD

def run():
    model = Sequential()
    model.add(Dense(input_dim=2, output_dim=4, init='glorot_normal'))
    model.add(Activation('sigmoid'))
    model.add(Dense(output_dim=2))
    model.add(Activation('sigmoid'))
    model.set_loss('mse')
    model.set_optimizer(SGD(lr=0.6))

    print model.summary()

    # x, y = datasets.load_xor_data()
    # y = np_utils.to_one_hot(y, 2)

    x, y = datasets.load_moons_data(nb_data=200, noise=0.1)
    y = np_utils.to_one_hot(y, 2)

    model.train(training_data=(x, y), nb_epochs=400, mini_batch_size=20, verbose=2, validation_data=(x, y),
                log_epoch=50)
    y = np_utils.to_categorical(y)

    model.plot_prediction(x, y)
    model.plot_training_iteration()

if __name__ == '__main__':
    run()