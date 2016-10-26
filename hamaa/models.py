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
from datetime import datetime
import sys

from . import losses
from .utils import np_utils
from .utils.time_utils import ProgressBar


class Sequential(object):

    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

        # 可训练参数
        self.trainable_params = []
        self.grads = []

        # 保存训练迭代过程信息
        self.logger = {}

        # 当前计算模式
        # train模式会保留各层中间计算结果
        # test只保留每层输入与输出

    def add(self, layer):
        # 连接层与层
        if len(self.layers) > 0:
            self.layers[-1].latter_layer = layer
            layer.previous_layer = self.layers[-1]
        layer.build()
        self.layers.append(layer)
        # 将每一层的可训练参数囊括进来
        self.trainable_params.extend(layer.trainable_params)

    def set_loss(self, loss):
        self.loss = losses.get(loss)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def summary(self):
        s = '=====================  summary =====================\n'
        for layer in self.layers:
            s += layer.summary() + '\n'
        s += 'loss: ' + str(self.loss.__name__) + '\n'
        s += 'optimizer: ' + str(self.optimizer.__class__.__name__) + '\n'
        s += '===================================================='
        return s

    def forward(self, _input, mode='experiment'):
        """前向计算"""
        data = _input
        for layer in self.layers:
            layer.mode = mode
            data = layer.forward(data)
        return data

    def predict(self, x):
        """预测label"""
        output = self.forward(x)
        return output.argmax(axis=1)

    def backward(self, _d_output):
        """后向计算"""
        d_data = _d_output
        for i in xrange(len(self.layers)-1, -1, -1):
            # 调用每一层的backward方法，反向传播误差
            # 并计算每一层里trainable_params的grads
            d_data = self.layers[i].backward(d_data)
        return d_data

    def train(self, training_data, nb_epochs, mini_batch_size, verbose=1, log_epoch=1, validation_data=None,
              shuffle=True, evaluate_batch_size=100, **kwargs):
        # 准备工作
        training_x, training_y = training_data
        n = training_x.shape[0]
        random_idx = range(n)
        # 计算每个epoch需要批量计算的次数
        batch_times = np.ceil(n * 1.0 / mini_batch_size).astype(np.int32)  # 除法，向上取整
        # 清空logger
        self.reset_logger()
        self.logger['start_time_is_valid'] = False
        # 初始化进度条
        bar = ProgressBar(total=n, width=50)

        for epoch in xrange(nb_epochs):
            if not self.logger['start_time_is_valid']:
                self.logger['start_time_is_valid'] = True
                self.logger['start_time'] = datetime.now()
            if shuffle:
                np.random.shuffle(random_idx)
            # 每个epoch重置进度条的进度
            bar.reset()
            if verbose == 2:
                bar.show()

            for i in xrange(batch_times):
                batch_range = random_idx[i * mini_batch_size:(i + 1) * mini_batch_size]
                batch_training_x = training_x[batch_range]
                batch_training_y = training_y[batch_range]
                # ========================================================
                # 前向计算
                batch_output = self.forward(batch_training_x, mode='train')
                # 损失函数求导
                diff = self.loss.diff(batch_training_y, batch_output)
                # 反向传播，每一层调用自己的backward
                # 方法来计算本层维护着的参数的梯度。
                self.backward(diff)
                # ========================================================
                # 依次获取每层的梯度
                self.grads = []
                for layer in self.layers:
                    self.grads.extend(layer.grads)
                # 将所有层的参数与梯度交给优化器来处理
                self.optimizer.update(self.trainable_params, self.grads)
                # ========================================================
                # 更新当前进度
                bar.move(batch_training_x.shape[0])
                # 显示进度条。设立了显示间隔以避免刷新频繁
                if verbose == 2 and (i + 1 == batch_times or i % (max(batch_times / 100, 1)) == 0):
                    bar.show(head='epoch: %2d/%d' % (epoch+1, nb_epochs))
            self.evaluate_train_performance(epoch, nb_epochs, training_data, validation_data, verbose,
                                            evaluate_batch_size, log_epoch)

    def evaluate_train_performance(self, epoch, nb_epochs, training_data, validation_data,
                                   verbose, evaluate_batch_size, log_epoch):

        sys.stdout.write('evaluating the whole training_data and validation_data...')
        sys.stdout.flush()
        if epoch + 1 == nb_epochs or epoch % log_epoch == 0:
            self.logger['epoch'].append(epoch)
            # 训练集acc以及loss
            training_x, training_y = training_data
            training_acc, training_loss = self.evaluate_accuracy_and_loss(training_x, training_y, evaluate_batch_size)
            text = 'epoch: %2d,  training_acc: %.5f,  training_loss: %.5f' % (epoch+1, training_acc, training_loss)
            self.logger['training_acc'].append(training_acc)
            self.logger['training_loss'].append(training_loss)

            # 验证集acc以及loss
            if validation_data:
                validation_x, validation_y = validation_data
                validation_acc, validation_loss = self.evaluate_accuracy_and_loss(validation_x,
                                                                                  validation_y,
                                                                                  evaluate_batch_size)
                text += ',  validation_acc: %.5f,  validation_loss: %.5f' % (validation_acc, validation_loss)
                self.logger['validation_acc'].append(validation_acc)
                self.logger['validation_loss'].append(validation_loss)

            # 时间统计
            self.logger['end_time'] = datetime.now()
            self.logger['start_time_is_valid'] = False
            delta_time = self.logger['end_time'] - self.logger['start_time']
            sec = delta_time.seconds + (delta_time.microseconds / 1000000.0)
            text += ',\t time:%fs' % sec
            text += ',\t lr:%f' % self.optimizer.cur_lr
            # 清空输出内容
            sys.stdout.write('\r')
            sys.stdout.write(' ' * 100)
            sys.stdout.write('\r')
            sys.stdout.flush()

            if verbose == 0:
                pass
            elif verbose == 1 or verbose == 2:
                print text

    def evaluate_accuracy(self, x, y, evaluate_batch_size=None):
        n = x.shape[0]
        idx = range(n)
        if not evaluate_batch_size:
            evaluate_batch_size = n
        batch_times = np.ceil(n * 1.0 / evaluate_batch_size).astype(np.int32)  # 除法，向上取整
        acc = 0.
        for i in xrange(batch_times):
            batch_range = idx[i * evaluate_batch_size:(i + 1) * evaluate_batch_size]
            batch_x = x[batch_range]
            batch_y = y[batch_range]
            batch_n = batch_x.shape[0]

            y_pred = self.predict(batch_x)
            y_real = np_utils.to_categorical(batch_y)
            acc += np.ones(batch_n)[y_pred == y_real].sum()
        acc /= n
        return acc

    def evaluate_loss(self, x, y, evaluate_batch_size=None):
        n = x.shape[0]
        idx = range(n)
        if not evaluate_batch_size:
            evaluate_batch_size = n
        batch_times = np.ceil(n * 1.0 / evaluate_batch_size).astype(np.int32)  # 除法，向上取整
        loss = 0.
        for i in xrange(batch_times):
            batch_range = idx[i * evaluate_batch_size:(i + 1) * evaluate_batch_size]
            batch_x = x[batch_range]
            batch_y = y[batch_range]
            batch_n = batch_x.shape[0]

            t = self.forward(batch_x)
            loss += self.loss.loss(batch_y, t) * n

        loss /= n
        return loss

    def evaluate_accuracy_and_loss(self, x, y, evaluate_batch_size=None):
        """评估模式对于测试集x与标签y的准确率与损失函数值，对于复杂模型与大型数据集支持批量评估"""
        n = x.shape[0]
        idx = range(n)
        if not evaluate_batch_size:
            evaluate_batch_size = n
        batch_times = np.ceil(n * 1.0 / evaluate_batch_size).astype(np.int32)  # 除法，向上取整
        loss = 0.
        acc = 0.
        for i in xrange(batch_times):
            batch_range = idx[i * evaluate_batch_size:(i + 1) * evaluate_batch_size]
            batch_x = x[batch_range]
            batch_y = y[batch_range]

            batch_n = batch_x.shape[0]
            output = self.forward(batch_x)
            loss += self.loss.loss(batch_y, output) * batch_n

            y_pred = output.argmax(axis=1)
            y_real = np_utils.to_categorical(batch_y)
            acc += np.ones(batch_n)[y_pred == y_real].sum()

        loss /= n
        acc /= n
        return acc, loss

    def reset_logger(self):
        self.logger = {}
        self.logger.setdefault('epoch', [])
        self.logger.setdefault('training_acc', [])
        self.logger.setdefault('training_loss', [])
        self.logger.setdefault('validation_acc', [])
        self.logger.setdefault('validation_loss', [])

    def plot_training_iteration(self):
        """画出迭代过程中，训练准确率、验证准确率、训练损失、验证损失随着迭代期变化的图像"""

        epoch = self.logger.get('epoch')
        training_acc = self.logger.get('training_acc')
        training_loss = self.logger.get('training_loss')
        validation_acc = self.logger.get('validation_acc')
        validation_loss = self.logger.get('validation_loss')

        plt.subplot(311)
        plt.ylim([0, 1.0])
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.plot(epoch, training_acc, label='train')
        plt.plot(epoch, validation_acc, label='validation')
        plt.legend(loc=0)

        plt.subplot(312)
        m = max(np.max(training_acc), np.max(validation_acc))
        plt.ylim([m - 0.05, m + 0.01])
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.plot(epoch, np.ones_like(epoch), 'k--')
        plt.plot(epoch, training_acc, label='train')
        plt.plot(epoch, validation_acc, label='validation')
        plt.legend(loc=0)

        plt.subplot(313)
        plt.ylim([0, 0.1 + max(np.max(training_loss), np.max(validation_loss))])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(epoch, training_loss, label='training')
        plt.plot(epoch, validation_loss, label='training')
        plt.legend(loc=0)

        plt.show()

    def plot_prediction(self, data):
        """绘画出决策边界。只适用于数据维数为2的情况。"""
        x, y = data
        # 如果y是one-hot，则重置为categorical
        if np.shape(y)[1] != 1:
            y = np_utils.to_categorical(y)
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

