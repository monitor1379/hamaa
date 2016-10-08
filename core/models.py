# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: models.py
@time: 2016/9/20 9:01


"""

import numpy as np
import matplotlib.pyplot as plt
from layers import Dense
from core.optimizer import OptimizerManager
from core.loss import LossManager
from utils import np_utils
from datetime import datetime
from utils.time_utils import tic, toc

class Sequential:
    def __init__(self):
        self.layers = []
        self.nb_layers = 0
        self.optimizer = None
        self.loss = None

        self.train_epoch = []
        self.train_accuracy = []
        self.train_loss = []
        self.validation_accuracy = []
        self.validation_loss = []

    def add(self, layer):
        self.layers.append(layer)
        layer.order_number = self.nb_layers
        self.nb_layers += 1

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = LossManager.get(loss)

    def forward(self, input_data):
        output_data = input_data
        for layer in self.layers:
            # tic(layer.layer_type)
            output_data = layer.forward(output_data)
            # toc()
        return output_data

    def backward_and_update(self, x, y, top_diff):
        d_output = top_diff
        for i in range(len(self.layers)-1, -1, -1):
            # tic(self.layers[i].layer_type)
            d_output = self.layers[i].backward(d_output)
            if self.layers[i].trainable:
                self.layers[i].update(self.optimizer.lr)
            # toc()

    def train(self, training_data, epochs, mini_batch_size, verbose, validation_data, shuffle=True, print_epoch=100):
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        n = training_x.shape[0]
        random_idx = range(n)
        batch_times = np.ceil(n * 1.0 / mini_batch_size).astype(np.int32)  # 除法，向上取整
        for epoch in range(epochs):
            t_start = datetime.now()
            if shuffle:
                np.random.shuffle(random_idx)
            # batch training过程
            for i in range(batch_times):
                batch_range = random_idx[i*mini_batch_size:(i+1)*mini_batch_size]
                batch_training_x = training_x[batch_range]
                batch_training_y = training_y[batch_range]
                # 前向计算
                batch_output = self.forward(batch_training_x)
                # 顶层导数
                top_diff = self.loss.diff_loss(batch_training_y, batch_output)
                # 反向传播
                self.backward_and_update(batch_training_x, batch_training_y, top_diff)
            t_end = datetime.now()
            # 打印log
            if verbose == 0 and (epoch % print_epoch == 0 or epoch == (epochs - 1)):
                t_end = datetime.now()
                t_delta = t_end - t_start
                print 'epoch:%d,\t t:%d.%ds' % (epoch, t_delta.seconds, t_delta.microseconds / 1000)
            elif verbose == 1 and (epoch % print_epoch == 0 or epoch == (epochs - 1)):
                train_acc, train_loss = self.evaluate_accuracy_and_loss(training_x, training_y)
                self.train_epoch.append(epoch)
                self.train_accuracy.append(train_acc)
                self.train_loss.append(train_loss)
                t_end = datetime.now()
                t_delta = t_end - t_start
                print 'epoch:%d,\t train_acc:%f,\t train_loss:%f,\t t:%d.%ds' % (epoch, train_acc, train_loss, t_delta.seconds, t_delta.microseconds / 1000)
            elif verbose == 2 and (epoch % print_epoch == 0 or epoch == (epochs - 1)):
                # 记录
                train_accuracy = self.evaluate_accuracy(training_x, training_y)
                train_loss = self.evaluate_loss(training_x, training_y)
                validation_accuracy = self.evaluate_accuracy(validation_x, validation_y)
                validation_loss = self.evaluate_loss(validation_x, validation_y)
                self.train_epoch.append(epoch)
                self.train_accuracy.append(train_accuracy)
                self.train_loss.append(train_loss)
                self.validation_accuracy.append(validation_accuracy)
                self.validation_loss.append(validation_loss)
                t_end = datetime.now()
                t_delta = t_end - t_start
                print 'epoch:%d,\t train_acc:%f,\t train_loss:%f,\t vali_acc:%f,\t vali_loss:%f,\t t:%d.%ds' \
                      % (epoch, train_accuracy, train_loss, validation_accuracy, validation_loss, t_delta.seconds,
                         t_delta.microseconds / 1000)

    def predict(self, x):
        output = self.forward(x)
        return output.argmax(axis=1)

    def evaluate_accuracy(self, x, y):
        output = self.predict(x)
        y = np_utils.to_real(y)
        return np.sum(output == y) * 1.0 / len(y)

    def evaluate_loss(self, x, y):
        output = self.forward(x)
        return self.loss.loss(y, output)

    def evaluate_accuracy_and_loss(self, x, y):
        output = self.forward(x)
        loss = self.loss.loss(y, output)
        pred = output.argmax(axis=1)
        y = np_utils.to_real(y)
        acc = np.sum(pred == y) * 1.0 / len(y)
        return acc, loss

    def summary(self):
        pass

    def plot_training_iteration(self):
        """
        绘画迭代过程中，训练准确率、验证准确率、训练损失、验证损失随着迭代期变化的图像
        :return:
        """
        plt.subplot(311)
        plt.ylim([0, 1.0])
        plt.plot(self.train_epoch, self.train_accuracy)

        plt.plot(self.train_epoch, self.validation_accuracy)

        plt.subplot(312)
        plt.ylim([0, 0.1 + max(np.max(self.train_loss), np.max(self.validation_loss))])
        plt.plot(self.train_epoch, self.train_loss)
        plt.plot(self.train_epoch, self.validation_loss)

        plt.subplot(313)
        m = max(np.max(self.train_accuracy), np.max(self.validation_accuracy))
        plt.ylim([m - 0.05, m + 0.01])
        plt.plot(self.train_epoch, self.train_accuracy)
        plt.plot(self.train_epoch, self.validation_accuracy)

        plt.show()

    def plot_prediction(self, x, y):
        """
        绘画出决策边界。只适用于数据维数为2的情况。
        :param x:
        :param y:
        :return:
        """


        real_x_max, real_x_min = np.max(x[:, 0]), np.min(x[:, 0])
        real_y_max, real_y_min = np.max(x[:, 1]), np.min(x[:, 1])
        x_padding = 0.1 * (real_x_max - real_x_min)
        y_padding = 0.1 * (real_y_max - real_y_min)
        x_max = real_x_max + x_padding
        x_min = real_x_min - x_padding
        y_max = real_y_max + y_padding
        y_min = real_y_min - y_padding

        h = 0.002 * (real_x_max - real_x_min)
        xx = np.arange(x_min, x_max, h)
        yy = np.arange(y_min, y_max, h)
        xx, yy = np.meshgrid(xx, yy)
        Z = np.c_[xx.ravel(), yy.ravel()]
        T = self.predict(Z)
        T = T.reshape(xx.shape)
        plt.figure('prediction')
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.contourf(xx, yy, T, cmap=plt.cm.Spectral)
        plt.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap=plt.cm.Spectral)
        plt.show()

