# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: model.py
@time: 2016/9/11 9:02


"""

import numpy as np
from gate import AddGate, MulGate, SigmoidGate, TanhGate
from loss import QuadraticLoss, CategoricalCrossEntropy
from utils import np_utils
import matplotlib.pyplot as plt
from datetime import datetime

class Network:
    """前向神经网络"""

    activation_dict = {
        'sigmoid': SigmoidGate,
        'tanh': TanhGate,
    }

    loss_dict = {
        'quadratic_loss': QuadraticLoss,
        'categorical_crossentropy': CategoricalCrossEntropy,
    }

    def __init__(self, layer_sizes, activation_names, loss_name):
        self.layer_sizes = layer_sizes
        self.activation_names = activation_names
        self.loss_name = loss_name

        self.nb_layers = len(layer_sizes)
        self.weights = []
        self.biases = []
        self.activations = []
        self.train_accuracy = []
        self.train_loss = []
        self.validation_accuracy = []
        self.validation_loss = []

        for i in range(self.nb_layers-1):
            self.weights.append(np.random.rand(layer_sizes[i], layer_sizes[i+1]) / np.sqrt(layer_sizes[i]))
            self.biases.append(np.random.rand(1, layer_sizes[i+1]))
            self.activations.append(Network.activation_dict[activation_names[i]])

        self.loss = Network.loss_dict[loss_name]


    def forward(self, x):
        """
        前向计算输出值
        :param x:
        :return:
        """
        act = x
        for i in range(self.nb_layers-1):
            mul = MulGate.forward(self.weights[i], act)
            add = AddGate.forward(mul, self.biases[i])
            act = self.activations[i].forward(add)
        output = act
        return output

    def forward_and_keep_mid_result(self, x):
        """
        前向计算输出值，且保留中间计算结果
        :param x:
        :return:
        """
        act = x
        mid_result = [(None, None, act)]
        for i in range(self.nb_layers-1):
            mul = MulGate.forward(self.weights[i], act)
            add = AddGate.forward(mul, self.biases[i])
            act = self.activations[i].forward(add)
            mid_result.append((mul, add, act))
        output = act
        return output, mid_result

    def predict(self, x):
        """
        前向计算，根据输出返回预测值，
        :param x:
        :return:
        """
        output = self.forward(x)
        return output.argmax(axis=1)

    def evaluate_accuracy(self, x, y):
        """
        根据预测值输出准确率
        :param x:
        :param y:
        :return:
        """
        output = self.predict(x)
        y = np_utils.to_real(y)
        return np.sum(output == y) * 1.0 / len(y)

    def evaluate_loss(self, x, y):
        output = self.forward(x)
        return self.loss.loss(y, output)

    def train(self, training_data, epochs, mini_batch_size, learning_rate, reg_lambda, verbose, validation_data, print_epoch=100):
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        n, m = np.shape(training_x)
        random_index = range(n)
        # 计算batch批处理计算的轮数
        batch_times = np.ceil(n * 1.0 / mini_batch_size).astype(np.int) # 整数除法
        # 对于每个迭代期
        for epoch in xrange(1, epochs+1, 1):
            t_start = datetime.now()
            np.random.shuffle(random_index)
            # if epoch % 5 == 0:
            #     learning_rate = learning_rate / 10
            #     print 'learning_rate:', learning_rate
            for i in range(batch_times):
                batch_index = random_index[i*mini_batch_size:(i+1)*mini_batch_size]
                batch_training_x = training_x[batch_index]
                batch_training_y = training_y[batch_index]
                # 前向
                batch_output, mid_result = self.forward_and_keep_mid_result(batch_training_x)
                top_diff = self.loss.diff_loss(batch_training_y, batch_output)
                # 反向传播
                self.backward_and_update(batch_training_x, batch_training_y, mid_result, top_diff, learning_rate, reg_lambda)
            t_end = datetime.now()
            # 打印log
            if verbose == 1 and epoch % print_epoch == 0:
                # 记录
                train_accuracy = self.evaluate_accuracy(training_x, training_y)
                train_loss = self.evaluate_loss(training_x, training_y)
                validation_accuracy = self.evaluate_accuracy(validation_x, validation_y)
                validation_loss = self.evaluate_loss(validation_x, validation_y)
                self.train_accuracy.append(train_accuracy)
                self.train_loss.append(train_loss)
                self.validation_accuracy.append(validation_accuracy)
                self.validation_loss.append(validation_loss)
                t_delta = t_end - t_start
                print 'epoch:%d,\t train_acc:%f,\t train_loss:%f,\t vali_acc:%f,\t vali_loss:%f,\t t:%d.%ds' \
                      % (epoch, train_accuracy, train_loss, validation_accuracy, validation_loss, t_delta.seconds, t_delta.microseconds/1000)

    def backward_and_update(self, x, y, mid_result, top_diff, learning_rate, reg_lambda):
        """
        反向传播。核心算法
        :param x:
        :param y:
        :param mid_result:中间计算结果
        :param top_diff:
        :param learning_rate:
        :return:
        """
        d_act = top_diff
        for i in range(self.nb_layers-1, 0, -1):
            d_add = self.activations[i-1].backward(mid_result[i][1], d_act)
            d_mul, d_b = AddGate.backward(mid_result[i][0], self.biases[i-1], d_add)
            d_w, d_act = MulGate.backward(self.weights[i-1], mid_result[i-1][2], d_mul)
            self.weights[i-1] -= reg_lambda * self.weights[i-1] / x.shape[0]
            self.weights[i-1] -= learning_rate * d_w
            self.biases[i-1] -= learning_rate * d_b

    def plot_training_iteration(self):
        """
        绘画迭代过程中，训练准确率、验证准确率、训练损失、验证损失随着迭代期变化的图像
        :return:
        """
        plt.subplot(311)
        plt.ylim([0, 1.0])
        plt.plot(self.train_accuracy)

        plt.plot(self.validation_accuracy)

        plt.subplot(312)
        plt.ylim([0, 0.1 + max(np.max(self.train_loss), np.max(self.validation_loss))])
        plt.plot(self.train_loss)
        plt.plot(self.validation_loss)

        plt.subplot(313)
        m = max(np.max(self.train_accuracy), np.max(self.validation_accuracy))
        plt.ylim([m - 0.05, m + 0.01])
        plt.plot(self.train_accuracy)
        plt.plot(self.validation_accuracy)

        plt.show()


    def plot_prediction(self, x, y):
        """
        绘画出决策边界。只适用于数据维数为2的情况。
        :param x:
        :param y:
        :return:
        """
        x_min = np.min(x[:, 0]) - 0.5
        x_max = np.max(x[:, 0]) + 0.5
        y_min = np.min(x[:, 1]) - 0.5
        y_max = np.max(x[:, 1]) + 0.5

        h = 0.01
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

