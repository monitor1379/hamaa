# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: example2_moons_nn.py
@time: 2016/10/9 17:45

构建一个神经元数目为 [2->3->2] 的多层神经网络来对线性不可分数据集进行分类。
"""

from hamaa.layers import Dense, Activation
from hamaa.models import Sequential
from hamaa.datasets import datasets
from hamaa.utils import np_utils
from hamaa.optimizers import SGD


def run():
    # 构建一个神经元数目为[2->3->2] 的多层神经网络来对moons数据进行分类
    model = Sequential()
    model.add(Dense(input_dim=2, output_dim=3, init='normal'))
    model.add(Activation('sigmoid'))
    model.add(Dense(output_dim=2))
    model.add(Activation('sigmoid'))
    model.set_objective('categorical_crossentropy')
    model.set_optimizer(SGD(lr=0.9, momentum=0.5))

    print model.summary()

    # 加载moons数据
    x, y = datasets.load_moons_data(nb_data=1000, noise=0.1)
    # 切分数据集中的10%作为验证集
    training_data, validation_data = np_utils.split_training_data(data=(x, y), split_ratio=0.9)

    model.train(training_data=training_data,        # 设置训练集
                nb_epochs=100,                      # 设置训练周期
                mini_batch_size=10,                 # 设置每次mini_batch的数据量
                verbose=1,                          # 设置训练过程显示方式，0代表不输出，1代表简单输出，2代表使用进图条功能
                validation_data=validation_data,    # 设置验证集
                log_epoch=10)                       # 设置每隔多少个周期才在控制台上显示一次训练过程的详细信息
    print '分类准确率: ', model.evaluate_accuracy(x, y)

    model.plot_prediction(data=training_data)       # 对训练集进行分类的结果可视化
    model.plot_prediction(data=validation_data)     # 对验证集进行分类的结果可视化
    model.plot_training_iteration()                 # 画出训练过程中准确率和损失函数值随着训练周期的变化图


if __name__ == '__main__':
    run()
