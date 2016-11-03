# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: example4_mnist_cnn.py
@time: 2016/10/20 23:49

构建一个卷积神经网络来对MNIST数据集进行分类。
使用进度条功能来显示过程，并使用可视化工具对卷积层
权重进行可视化。
"""

from hamaa.datasets.datasets import load_mnist_data
from hamaa.layers import Dense, Activation, Convolution2D, Flatten, MeanPooling2D
from hamaa.models import Sequential
from hamaa.optimizers import SGD
from hamaa.utils import vis_utils
from hamaa.utils.np_utils import split_training_data


def run():
    print 'loading MNIST dataset...'
    training_data, test_data = load_mnist_data(nb_training=3500, nb_test=10000, preprocess=True, flatten=False)
    training_data, validation_data = split_training_data(training_data, nb_validation=500)

    print 'training_data:', training_data[0].shape
    print 'validation_data:', validation_data[0].shape
    print 'test_data:', test_data[0].shape

    model = Sequential()
    model.add(Convolution2D(nb_kernel=6, kernel_height=5, kernel_width=5, activation='tanh', input_shape=(1, 28, 28)))
    model.add(MeanPooling2D(pooling_size=(2, 2)))
    model.add(Convolution2D(nb_kernel=10, kernel_height=5, kernel_width=5, activation='tanh'))
    model.add(MeanPooling2D(pooling_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(output_dim=30, init='glorot_normal'))
    model.add(Activation('sigmoid'))
    model.add(Dense(output_dim=10, init='glorot_normal'))
    model.add(Activation('sigmoid'))

    model.set_objective('categorical_crossentropy')
    model.set_optimizer(SGD(lr=0.09, momentum=0.3, decay=1e-6))

    print model.summary()

    model.train(training_data=training_data,
                nb_epochs=20,
                mini_batch_size=50,
                verbose=2,
                validation_data=validation_data,
                log_epoch=1
                )

    print model.evaluate_accuracy(test_data[0], test_data[1])

    # 使用可视化工具对卷积层的权重进行可视化
    vis_utils.visualize_convolution_weight(model.layers[0], title='layer 0')
    vis_utils.visualize_convolution_weight(model.layers[2], title='layer 2')

    model.plot_training_iteration()


if __name__ == '__main__':
    run()
