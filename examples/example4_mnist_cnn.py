# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: example4_mnist_cnn.py
@time: 2016/10/20 23:49

Build a convolution neural network on MNIST.
"""

from hamaa.datasets.datasets import load_mnist_data
from hamaa.layers import Dense, Activation, Convolution2D, Flatten, MeanPooling2D
from hamaa.models import Sequential
from hamaa.optimizers import SGD
from hamaa.utils import vis_utils
from hamaa.utils.np_utils import split_training_data


print 'loading MNIST dataset...'
training_data, test_data = load_mnist_data(nb_training=6000, nb_test=10000, preprocess=True, flatten=False)
training_data, validation_data = split_training_data(training_data, nb_validation=500)

print 'training_data:', training_data[0].shape
print 'validation_data:', validation_data[0].shape
print 'test_data:', test_data[0].shape

model = Sequential()
model.add(Convolution2D(nb_kernel=32, kernel_height=5, kernel_width=5, activation='relu', input_shape=(1, 28, 28)))
model.add(MeanPooling2D(pooling_size=(2, 2)))
model.add(Convolution2D(nb_kernel=64, kernel_height=5, kernel_width=5, activation='relu'))
model.add(MeanPooling2D(pooling_size=(2, 2)))
model.add(Flatten())
model.add(Dense(output_dim=200, init='glorot_normal'))
model.add(Activation('sigmoid'))
model.add(Dense(output_dim=10, init='glorot_normal'))
model.add(Activation('softmax'))

model.set_objective('categorical_crossentropy')
model.set_optimizer(SGD(lr=0.002, momentum=0.9))

print model.summary()

model.train(training_data=training_data,
            nb_epochs=20,
            mini_batch_size=64,
            verbose=2,
            validation_data=validation_data,
            log_epoch=1)

print model.evaluate_accuracy(test_data[0], test_data[1])

# visualize the kernels of convolution layer
vis_utils.visualize_convolution_kernel(model.layers[0], title='layer 0')
vis_utils.visualize_convolution_kernel(model.layers[2], title='layer 2')

model.plot_training_iteration()


