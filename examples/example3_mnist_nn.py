# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: example3_mnist_nn.py
@time: 2016/10/13 0:07


"""


from hamaa.layers import Dense, Activation
from hamaa.models import Sequential
from hamaa.datasets import datasets
from hamaa.utils import np_utils
from hamaa.optimizers import SGD


def run():
    training_data, test_data = datasets.load_mnist_data(nb_training=6000, nb_test=10000, preprocess=True, flatten=True)
    training_data, validation_data = np_utils.split_training_data(training_data, 0.95)

    print 'training_data:', training_data[0].shape
    print 'validation_data:', validation_data[0].shape
    print 'test_data:', test_data[0].shape

    model = Sequential()
    model.add(Dense(input_dim=784, output_dim=50, init='glorot_normal'))
    model.add(Activation('sigmoid'))
    model.add(Dense(output_dim=50, init='glorot_normal'))
    model.add(Activation('sigmoid'))
    model.add(Dense(output_dim=10, init='glorot_normal'))
    model.add(Activation('sigmoid'))

    model.set_objective(loss='categorical_crossentropy')
    model.set_optimizer(optimizer=SGD(lr=0.3, momentum=0.2, decay=1e-6))

    print model.summary()

    model.train(training_data=training_data,
                nb_epochs=30,
                mini_batch_size=10,
                verbose=1,
                validation_data=validation_data,
                log_epoch=1
                )

    print model.evaluate_accuracy(test_data[0], test_data[1])
    model.plot_training_iteration()


if __name__ == '__main__':
    run()