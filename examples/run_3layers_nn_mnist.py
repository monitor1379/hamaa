# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: run_3layers_nn_mnist.py
@time: 2016/10/3 16:44


"""

from core.models import Sequential
from core.layers import Dense, Activation
from core.optimizer import SGD
from datasets import datasets
from utils import np_utils

nb_training = 60000
nb_test = 10000
lr = 0.25
epochs = 30
mini_batch_size = 10
hidden_dim = 100
print_epoch = 1

verbose = 2
input_dim = 28 * 28
nb_classes = 10

def preprocess_data():
    training_x, training_y, test_x, test_y = datasets.load_mnist_data(nb_training, nb_test)
    training_x, training_y, validation_x, validation_y = np_utils.split_training_data(training_x, training_y, 0.9)

    training_x = training_x.reshape(training_x.shape[0], input_dim)
    validation_x = validation_x.reshape(validation_x.shape[0], input_dim)
    test_x = test_x.reshape(test_x.shape[0], input_dim)

    training_x[training_x > 0] = 1.0
    validation_x[validation_x > 0] = 1.0
    test_x[test_x > 0] = 1.0

    training_y = np_utils.to_categorical(training_y, nb_classes)
    validation_y = np_utils.to_categorical(validation_y, nb_classes)
    test_y = np_utils.to_categorical(test_y, nb_classes)

    return (training_x, training_y), (validation_x, validation_y), (test_x, test_y)

def run():
    training_data, validation_data, test_data = preprocess_data()
    model = Sequential()
    model.add(Dense(input_dim=input_dim, output_dim=hidden_dim, init='normal_square'))
    model.add(Activation('relu'))
    model.add(Dense(input_dim=hidden_dim, output_dim=nb_classes, init='normal_square'))
    model.add(Activation('sigmoid'))

    model.compile(optimizer=SGD(lr), loss='quadratic_loss')
    model.train(training_data=training_data, epochs=epochs, mini_batch_size=mini_batch_size,
                verbose=verbose, validation_data=validation_data, print_epoch=print_epoch)
    model.plot_training_iteration()
    print model.evaluate_accuracy(test_data[0], test_data[1])

if __name__ == '__main__':
    run()