# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: trainers.py
@time: 2016/10/29 19:25


"""

import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

from .utils import np_utils
from .utils.time_utils import ProgressBar


class SequentialTrainer(object):
    def __init__(self):
        self.logger = {}

    def reset_logger(self):
        self.logger = {}
        self.logger.setdefault('epoch', [])
        self.logger.setdefault('training_acc', [])
        self.logger.setdefault('training_loss', [])
        self.logger.setdefault('validation_acc', [])
        self.logger.setdefault('validation_loss', [])

    def train_by_batch(self, model, batch_training_data):
        """一个批次的训练过程"""
        batch_training_x, batch_training_y = batch_training_data
        # ========================================================
        # 前向计算；损失函数对输出求导；反向计算。
        batch_output = model.forward(batch_training_x, mode='train')  # 前向计算
        diff = model.objective.diff(batch_training_y, batch_output)  # 损失函数求导
        model.backward(diff)  # 反向传播，每一层调用自己的backward
        # ========================================================
        # 依次获取每层的梯度，并将所有层的参数与梯度交给优化器来处理
        model.grads = []
        for layer in model.layers:
            model.grads.append(layer.grads)
        # ========================================================
        # 更新模型的参数
        model.optimizer.update(model.trainable_params, model.grads)

    def naive_train(self, model, training_data, nb_epochs, mini_batch_size, shuffle=True, **kwargs):
        """一个源代码最简洁的批量训练函数框架，仅用于源代码参考与扩展，不建议使用。"""
        warnings.warn("naive_train() is deprecated. Use train() instead.", DeprecationWarning)

        training_x, training_y = training_data
        n = training_x.shape[0]

        # 用一个列表来当做数据的索引，如果要随机抽取training_x中的数据，
        # 可以通过打乱该索引来获取。这样做的目的是为了避免打乱原数据的顺序。
        random_idx = range(n)

        # 求每个epoch中批量计算的次数
        batch_times = np.ceil(n * 1.0 / mini_batch_size).astype(np.int32)

        for epoch in xrange(1, nb_epochs + 1):  # 对于每个epoch(从1到nb_epochs)
            if shuffle:
                np.random.shuffle(random_idx)  # 打乱数据索引
            for i in xrange(batch_times):  # 对于每次批量计算
                batch_range = random_idx[i * mini_batch_size:(i + 1) * mini_batch_size]
                batch_training_x = training_x[batch_range]
                batch_training_y = training_y[batch_range]
                self.train_by_batch(model, (batch_training_x, batch_training_y))
            print 'epoch:{}, loss:{}'.format(epoch, model.evaluate_loss(training_x, training_y))

    def train(self, model, training_data, nb_epochs, mini_batch_size, verbose=1, log_epoch=1,
              validation_data=None, shuffle=True, evaluate_batch_size=100, **kwargs):
        """训练模型.TODO"""

        assert log_epoch > 0, 'log_epoch:{} must be larger than zero!'.format(log_epoch)
        assert verbose in [0, 1, 2], 'invalid verbose:{}'.format(verbose)

        # 当log_epoch不为1时，不推荐使用进度条功能。
        if log_epoch != 1 and verbose == 2:
            raise Exception('Invalid log_epoch and verbose: verbose为2时log_epoch只能为1!\n'
                            '原因：进图条功能用于在控制台显示每个epoch的完成进度，\n'
                            '如果log_epoch > 1，说明控制台每隔log_epoch个epoch才显示一次，\n'
                            '会和进图条的显示功能产生冲突。')

        training_x, training_y = training_data
        n = training_x.shape[0]
        random_idx = range(n)
        batch_times = np.ceil(n * 1.0 / mini_batch_size).astype(np.int32)

        # logger负责记录训练过程，同时统计两次log之间的时间间隔
        self.reset_logger()
        self.logger['need_to_refresh_start_time'] = True

        bar = ProgressBar(total=n, width=20)
        bar.reset()
        if verbose == 2:
            bar.show(head='epoch: %2d/%d' % (0, nb_epochs))

        for epoch in xrange(1, nb_epochs + 1):
            if self.logger['need_to_refresh_start_time']:
                self.logger['need_to_refresh_start_time'] = False
                self.logger['start_time'] = time.time()

            bar.reset()

            if shuffle:
                np.random.shuffle(random_idx)
            for i in xrange(batch_times):
                batch_range = random_idx[i * mini_batch_size:(i + 1) * mini_batch_size]
                batch_training_x = training_x[batch_range]
                batch_training_y = training_y[batch_range]
                self.train_by_batch(model, (batch_training_x, batch_training_y))

                # 更新当前进度
                bar.move(batch_training_x.shape[0])
                # 显示进度条。设立了显示间隔以避免刷新频繁
                if verbose == 2 and (i + 1 == batch_times or i % (max(batch_times / 20, 1)) == 0):
                    bar.show(head='epoch: %2d/%d' % (epoch, nb_epochs))

            bar.clear()

            if epoch == 1 or epoch == nb_epochs or epoch % log_epoch == 0:
                self.__evaluate_train_performance(model,
                                                  epoch,
                                                  training_data,
                                                  validation_data,
                                                  verbose,
                                                  evaluate_batch_size)

    def __evaluate_train_performance(self, model, epoch, training_data, validation_data,
                                     verbose, evaluate_batch_size):
        sys.stdout.write('evaluating the whole training_data and validation_data...')
        sys.stdout.flush()
        self.logger['epoch'].append(epoch)

        # 训练集acc以及loss
        training_x, training_y = training_data
        training_acc, training_loss = model.evaluate_accuracy_and_loss(training_x,
                                                                       training_y,
                                                                       evaluate_batch_size)
        text = 'epoch: {:3},  train_acc:{:7.3f}%,  train_loss:{:.4f}'.format(epoch,
                                                                                   training_acc * 100,
                                                                                   training_loss)
        self.logger['training_acc'].append(training_acc)
        self.logger['training_loss'].append(training_loss)

        # 验证集acc以及loss
        if validation_data:
            validation_x, validation_y = validation_data
            validation_acc, validation_loss = model.evaluate_accuracy_and_loss(validation_x,
                                                                               validation_y,
                                                                               evaluate_batch_size)
            text += ',  valid_acc:{:7.3f}%,  valid_loss:{:.4f}'.format(validation_acc * 100,
                                                                                 validation_loss)
            self.logger['validation_acc'].append(validation_acc)
            self.logger['validation_loss'].append(validation_loss)

        # 时间统计
        self.logger['end_time'] = time.time()
        self.logger['need_to_refresh_start_time'] = True
        delta_time = self.logger['end_time'] - self.logger['start_time']
        text += ',  time: %.3fs' % delta_time
        text += ',  lr: %f' % model.optimizer.cur_lr

        # 清空输出内容
        sys.stdout.write('\r')
        sys.stdout.write(' ' * 100)
        sys.stdout.write('\r')
        sys.stdout.flush()

        if verbose == 0:
            pass
        elif verbose == 1 or verbose == 2:
            print text

    def plot_training_iteration(self):
        """画出迭代过程中，训练准确率、验证准确率、训练损失、验证损失随着迭代期变化的图像"""
        epoch = self.logger.get('epoch')
        training_acc = self.logger.get('training_acc')
        training_loss = self.logger.get('training_loss')
        validation_acc = self.logger.get('validation_acc')
        validation_loss = self.logger.get('validation_loss')

        # 绘画准确率随着训练周期的折线图
        plt.subplot(311)
        plt.xlim([1, epoch[-1]])
        plt.ylim([0, 1.0])
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.plot(epoch, training_acc, label='train')
        if validation_acc:
            plt.plot(epoch, validation_acc, label='validation')
        plt.legend(loc=0)

        # 绘画准确率随着训练周期的折线图的放大版本
        plt.subplot(312)
        if validation_acc:
            top = max(np.max(training_acc), np.max(validation_acc))
            bottom = min(np.min(training_acc), np.min(validation_acc))
        else:
            top = np.max(training_acc)
            bottom = np.min(training_acc)
        plt.xlim([1, epoch[-1]])
        plt.ylim([bottom - 0.01, top + 0.01])
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.plot(epoch, np.ones_like(epoch), 'k--')
        plt.plot(epoch, training_acc, label='train')
        if validation_acc:
            plt.plot(epoch, validation_acc, label='validation')
        plt.legend(loc=0)

        # 绘画损失函数值随着训练周期的折线图
        plt.subplot(313)
        if validation_acc:
            top = max(np.max(training_loss), np.max(validation_loss))
        else:
            top = np.max(training_loss)
        plt.xlim([1, epoch[-1]])
        plt.ylim([0, 0.1 + top])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(epoch, training_loss, label='training')
        if validation_loss:
            plt.plot(epoch, validation_loss, label='validation')
        plt.legend(loc=0)

        plt.show()

    def plot_prediction(self, model, data):
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
        T = model.predict(Z)
        T = T.reshape(xx.shape)
        plt.figure('prediction')
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.contourf(xx, yy, T, cmap=plt.cm.Spectral)
        plt.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap=plt.cm.Spectral)
        plt.show()
