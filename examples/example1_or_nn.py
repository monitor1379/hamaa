# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: example1_or_nn.py
@time: 2016/10/11 0:07

使用Hama构建单层神经网络来解决或问题的一个简洁的例子。
其中包含了创建、训练、测试一个神经网络必备的所有函数。
"""

from hamaa.datasets import datasets
from hamaa.layers import Dense, Activation
from hamaa.models import Sequential
from hamaa.optimizers import SGD


def run():
    model = Sequential()                                        # 创建一个神经网络模型
    model.add(Dense(input_dim=2, output_dim=2, init='uniform')) # 添加一个输入神经元数是2、输出神经元数是2的全连接层
    model.add(Activation('sigmoid'))                            # 添加一个激活函数为sigmoid的激活层
    model.set_objective('mse')                                  # 设置目标函数/损失函数为均方差
    model.set_optimizer(SGD(lr=0.9, momentum=0.9, decay=1e-6))  # 设置优化器为随机梯度下降法
    print model.summary()                                       # 打印模型的详细信息
    x, y = datasets.load_or_data()                              # 加载数据
    model.train(training_data=(x, y), nb_epochs=10)             # 开始训练，设置训练周期为10
    print 'test accuracy: ', model.evaluate_accuracy(x, y)      # 评估模型的准确率


if __name__ == '__main__':
    run()
