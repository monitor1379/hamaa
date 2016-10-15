# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test6_new_optimizer.py
@time: 2016/10/12 22:29


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
    model.set_optimizer(SGD(lr=0.9, momentum=0.9, decay=0.0))

    print model.summary()

    x, y = datasets.load_or_data()
    y = np_utils.to_one_hot(y, 2)
    data = (x, y)
    model.train(training_data=data,
                nb_epochs=10,
                mini_batch_size=1,
                verbose=0,
                validation_data=data,
                log_epoch=1)
    print model.optimizer.iterations
    print model.optimizer.cur_lr

    model.plot_prediction(data=data)
    model.plot_training_iteration()


if __name__ == '__main__':
    run()
