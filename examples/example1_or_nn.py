# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: example1_or_nn.py
@time: 2016/10/11 0:07


"""

from hamaa.layers import Dense, Activation
from hamaa.models import Sequential
from hamaa.datasets import datasets
from hamaa.utils import np_utils
from hamaa.optimizers import SGD


def run():
    model = Sequential()
    model.add(Dense(input_dim=2, output_dim=2, init='normal'))
    model.add(Activation('sigmoid'))
    model.set_loss('mse')
    model.set_optimizer(SGD(lr=0.9, momentum=0.9, decay=1e-6))

    print model.summary()

    x, y = datasets.load_or_data()
    y = np_utils.to_one_hot(y, 2)
    data = (x, y)

    model.train(training_data=data,
                nb_epochs=10,
                mini_batch_size=1,
                verbose=1,
                validation_data=data,
                log_epoch=1)

    model.plot_prediction(data=data)
    model.plot_training_iteration()


if __name__ == '__main__':
    run()
