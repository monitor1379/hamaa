# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: models.py
@time: 2016/9/20 9:01


"""

import numpy as np

from . import objectives
from .trainers import SequentialTrainer
from .utils import np_utils


class Sequential(object):
    """一种由网络层线性堆叠而成的模型。

    # Arguments
        layers: 由网络层对象组成的列表。

    # Note
        网络层的第一层必须指定输入数据的形状，既在层的构造函数
        中添加`input_dim`参数（比如Dense）或者添加`input_shape`
        参数（比如Convolution2D层）。

    # Example
        ```python
            # TODO
        ```
    """

    def __init__(self, layers=None):
        self.layers = []  # 模型成员：网络层
        self.trainable_params = []  # 模型的所有可训练参数
        self.grads = []  # 模型的所有可训练参数的梯度

        self.objective = None  # 目标函数，即损失函数
        self.optimizer = None  # 优化器
        self.trainer = SequentialTrainer()  # 训练器，负责管理模型的训练过程

        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        """添加一个网络层作为网络的最后一层。

        # Arguments:
            layer: 网络层对象。

        """
        if len(self.layers) > 0:
            self.layers[-1].latter_layer = layer
            layer.previous_layer = self.layers[-1].latter_layer
        self.layers.append(layer)
        # 调用网络层对象的build方法来初始化网络层对象的参数
        layer.build()
        self.trainable_params.append(layer.trainable_params)

    def pop(self):
        """将网络的最后一层网络层移除掉。"""
        if not self.layers:
            raise Exception('There are no layers in the model.')

        self.layers.pop()
        self.layers[-1].latter_layer = None
        self.trainable_params.pop()

    def set_objective(self, objective):
        """设置目标函数/损失函数类型。

        # Arguments:
            objective: 字符串，表示目标函数的类型，比如'mse'或者
            'categorical_crossentropy'等等。

        # Note:
            参数是str类型。
        """
        self.objective = objectives.get(objective)

    def set_optimizer(self, optimizer):
        """设置优化器。

        # Arguments:
            optimizers: 优化器对象，比如`SGD(lr=0.1)`，

        # Note:
            参数不是str类型。
        """
        self.optimizer = optimizer

    def summary(self):
        """返回一个描述了模型构造信息的字符串。"""
        info = '=====================  summary =====================\n'
        for layer in self.layers:
            info += layer.summary() + '\n'
        info += 'loss: ' + str(self.objective.__name__) + '\n'
        info += 'optimizer: ' + str(self.optimizer.__class__.__name__) + '\n'
        info += '===================================================='
        return info

    def forward(self, input_data, mode='test'):
        """前向计算过程。

        # Arguments:
            input_data: 输入数据，numpy.ndarray对象，
            遵循NM格式或者NCHW格式。

            mode: 前向计算的模式。
                - `train`: 前向计算过程会保留中间计算结果。
                - `test`: 前向计算过程不会保留中间计算结果。

        """
        data = input_data
        for layer in self.layers:
            layer.mode = mode
            data = layer.forward(data)
        return data

    def predict(self, input_data):
        """根据输入数据计算预测结果。

        # Arguments:
            input_data: 输入数据，numpy.ndarray对象，
            遵循NM格式或者NCHW格式。
        """
        output_data = self.forward(input_data)
        return output_data.argmax(axis=1)

    def backward(self, d_output_data):
        """后向计算过程，依次将误差从后向前传播，并依次调用
        每一层的backward方法来求目标函数objective对每一层
        的参数的梯度。
        每一层的参数的梯度保存在每一层的grads列表中。

        # Arguments:
            d_output_data: 目标函数objective对于网络输出
            output_data的梯度。前缀`d_`表示这是一个梯度值。

        # Note:
            该过程不会更新参数。
        """
        d_data = d_output_data
        for i in xrange(len(self.layers)-1, -1, -1):
            d_data = self.layers[i].backward(d_data)
        return d_data

    def evaluate_accuracy(self, x, y, evaluate_batch_size=None):
        """根据数据集x及其标签y，评估模型的分类准确率。

        对于大型数据集支持批量计算。

        # Arguments:
            x: 数据集，遵循NM格式或者NCHW格式。

            y: one_hot化的标签，n*d维numpy.ndarray对象，
            一个维度代表一个类别。

            evaluate_batch_size: 每次批量计算的数据个数，如果不指定
            则表示不使用批量计算，而是直接将所有数据一次性喂进网络中。

        # Note:
            如果数据集个数比较多、数据维度比较大或者模型复杂度比较高
            的时候，建议选择批量计算以防止爆内存。

            另外该函数不支持y的维度d为1的情况。
        """
        d = y.shape[1]
        assert d != 1, '数据集的类别数量不能为1！'

        n = x.shape[0]
        idx = range(n)

        if not evaluate_batch_size:
            evaluate_batch_size = n

        # 批量计算的次数（向上取整）
        batch_times = np.ceil(n * 1.0 / evaluate_batch_size).astype(np.int32)
        k = 0.  # 准确预测的数据个数

        for i in xrange(batch_times):
            batch_range = idx[i * evaluate_batch_size:(i + 1) * evaluate_batch_size]
            batch_x = x[batch_range]
            batch_y = y[batch_range]
            batch_n = batch_x.shape[0]

            y_pred = self.predict(batch_x)
            y_real = np_utils.to_categorical(batch_y)  # 将one hot数组转换为离散类别值
            k += np.ones(batch_n)[y_pred == y_real].sum()

        acc = k / n
        return acc

    def evaluate_loss(self, x, y, evaluate_batch_size=None):
        """根据数据集x及其标签y，评估模型的损失函数值。

        对于大型数据集支持批量计算。

        # Arguments:
            x: 数据集，遵循NM格式或者NCHW格式。

            y: 标签，n*d维numpy.ndarray对象。
            当m = 1时表示y是连续值，当前问题是回归问题；
            当m > 1时表示y是one hot化的类别标签。

            evaluate_batch_size: 每次批量计算的数据个数，如果不指定
            则表示不使用批量计算，而是直接将所有数据一次性喂进网络中。

        # Note:
            如果数据集个数比较多、数据维度比较大或者模型复杂度比较高
            的时候，建议选择批量计算以防止爆内存。
        """
        n = x.shape[0]
        idx = range(n)

        if not evaluate_batch_size:
            evaluate_batch_size = n

        # 批量计算的次数（向上取整）
        batch_times = np.ceil(n * 1.0 / evaluate_batch_size).astype(np.int32)  # 除法，向上取整
        loss = 0.

        for i in xrange(batch_times):
            batch_range = idx[i * evaluate_batch_size:(i + 1) * evaluate_batch_size]
            batch_x = x[batch_range]
            batch_y = y[batch_range]
            batch_n = batch_x.shape[0]

            output = self.forward(batch_x)
            loss += self.objective.loss(batch_y, output) * batch_n

        loss /= n
        return loss

    def evaluate_accuracy_and_loss(self, x, y, evaluate_batch_size=None):
        """根据数据集x及其标签y，评估模型的分类准确率以及损失函数值。对于大型数据集
        支持批量计算。

        因为评估模型准确率以及求损失函数值存在重复的计算过程，因此该函数将两个过程
        合并，以减少重复计算。

        # Arguments:
            x: 数据集，遵循NM格式或者NCHW格式。

            y: one_hot化的标签，n*d维numpy.ndarray对象，
            一个维度代表一个类别。

            evaluate_batch_size: 每次批量计算的数据个数，如果不指定
            则表示不使用批量计算，而是直接将所有数据一次性喂进网络中。

        # Note:
            如果数据集个数比较多、数据维度比较大或者模型复杂度比较高
            的时候，建议选择批量计算以防止爆内存。

            另外该函数不支持y的维度d为1的情况。
        """
        n = x.shape[0]
        idx = range(n)

        if not evaluate_batch_size:
            evaluate_batch_size = n

        batch_times = np.ceil(n * 1.0 / evaluate_batch_size).astype(np.int32)
        loss = 0.
        k = 0.  # 正确预测的数据个数

        for i in xrange(batch_times):
            batch_range = idx[i * evaluate_batch_size:(i + 1) * evaluate_batch_size]
            batch_x = x[batch_range]
            batch_y = y[batch_range]
            batch_n = batch_x.shape[0]

            output = self.forward(batch_x)
            loss += self.objective.loss(batch_y, output) * batch_n

            y_pred = output.argmax(axis=1)
            y_real = np_utils.to_categorical(batch_y)
            k += np.ones(batch_n)[y_pred == y_real].sum()

        acc = k / n
        loss /= n
        return acc, loss

    def train(self, training_data, nb_epochs, mini_batch_size, verbose=1, log_epoch=1,
              validation_data=None, shuffle=True, evaluate_batch_size=100, **kwargs):
        # TODO
        """训练模型。

        # Arguments:
            training_data:

        # Note:

        # Example:

            训练模型的方法：
            ```python
                # 下面的两种方法是等价的。

                # 方法一
                model = Sequential()
                model.train(...)

                # 方法二
                model = Sequential()
                trainer = SequentialTrainer()
                trainer.train(model, ...)

            ```


        """

        self.trainer.train(self,
                           training_data,
                           nb_epochs,
                           mini_batch_size,
                           verbose,
                           log_epoch=log_epoch,
                           validation_data=validation_data,
                           shuffle=shuffle,
                           evaluate_batch_size=evaluate_batch_size)

    def plot_prediction(self, data):
        self.trainer.plot_prediction(self, data)

    def plot_training_iteration(self):
        self.trainer.plot_training_iteration()


