# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: run_moons.py
@time: 2016/9/20 18:32


"""

import numpy as np
from utils import np_utils
from datasets import datasets
from core.optimizer import SGD
from core.models import Sequential
from core.layers import Dense, Activation


def run():
    # 加载数据集
    # x, y = datasets.load_and_data()
    # x, y = datasets.load_xor_data()
    x, y = datasets.load_moons_data(200, 0.1)

    # 数据预处理
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    y = np_utils.to_categorical(y, 2)

    # 搭建模型
    model = Sequential()
    model.add(Dense(input_dim=2, output_dim=4, init='normal_square'))
    model.add(Activation('sigmoid'))
    model.add(Dense(input_dim=4, output_dim=2, init='normal_square'))
    model.add(Activation('sigmoid'))

    # 编译与训练模型
    model.compile(optimizer=SGD(lr=0.5), loss='categorical_crossentropy')
    model.train(training_data=(x, y), epochs=900, mini_batch_size=30, verbose=1, validation_data=(x, y), print_epoch=10)

    # 模型可视化（只适用于二维数据）
    model.plot_training_iteration()
    model.plot_prediction(x, np_utils.to_real(y))


if __name__ == '__main__':
    run()