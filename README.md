<p align="center">
	<img width=300 src="https://rawgit.com/monitor1379/hamaa/dev/docs/images/hamaa-logo.svg" />
</p>

## Hamaa: a Simple and Naive Deep Learning library 

[![Documentation Status](https://readthedocs.org/projects/hamaa/badge/?version=latest)](http://hamaa.readthedocs.io/zh_CN/latest/?badge=latest) 
[![Build Status](https://travis-ci.org/monitor1379/hamaa.svg?branch=master)](https://travis-ci.org/monitor1379/hamaa)
[![GitHub license](https://img.shields.io/badge/license-AGPL-blue.svg)](https://raw.githubusercontent.com/monitor1379/hamaa/master/LICENSE)
[![codecov](https://codecov.io/gh/monitor1379/hamaa/branch/master/graph/badge.svg)](https://codecov.io/gh/monitor1379/hamaa)

### What is Hamaa

- Hamaa是一个构建于Python/NumPy之上的深度学习库，
类[Keras](http://keras.io)的API以及模块化的设计使得用Hamaa来搭建神经网络就像搭建积木一样简单。
- 之所以编写Hamaa，是因为作者在学习DL时发现，
虽然网络上与DL相关的论文、教程与框架浩如烟海，
但是这些要么全是数学公式、要么并没有教怎么去实现、要么为了提升速度牺牲了源代码的可读性，
所以作者尝试着自己编写一个简单、朴素的深度学习库，写着写着就有了Hamaa。
- 所以，Hamaa的出现**并不是为了取代(当然也没有这个能力去取代)工业界上常用的深度学习框架**
(诸如
[TensorFlow](https://www.tensorflow.org/)、
[Theano](http://www.deeplearning.net/software/theano/)、
[Caffe](http://caffe.berkeleyvision.org/)、
[MXNet](http://mxnet.readthedocs.io/en/latest/)
等等)；
- 相反，Hamaa被创造的目的是**希望让深度学习的初学者能够通过使用、
阅读Hamaa的源代码甚至去重现Hamaa来加深对神经网络/深度学习的理解**，
以便更好地去使用上述几个工业级别的深度学习框架去解决实际问题（而不是成为一个调包/调参侠 :D）。

---

### The Design Philosophy of Hamaa


Hamaa始终遵循着`too SIMPLE and sometimes NAIVE`的原则来设计：
- **每一个可配置项都抽象成简单的模块**。 具体地，网络层、损失函数、优化器、初始化策略、激活函数都是独立
的模块，能够通过自由组装的方式来搭建模型。
- **所有模块都使用朴素的代码实现**。每一段源代码都希望能在第一次阅读时显得直观易懂，具有良好的可读性，
并且不过分使用trick。


Hamaa尽管在速度优化上没有工业级别的深度学习框架要好，
但Hamaa也具有以下优点:

- **安装依赖少，无需配置**。如果你想要进行一些简单的快速实验，又不想花大量时间在搭建与配置环境上的话， 
那么基于Python的Hamaa可能是不错的选择。
- **使用简单**。学会使用Hamaa搭建一个完整的神经网络并进行数据可视化仅需不到1分钟(可查看下面的Getting Started一节)。
此外如果你熟悉[Keras](http://keras.io)的话将能更快地上手。
- **源码易读**。朴素的设计模式、丰富的源码注释以及完备的设计文档能够让你深入理解Hamaa是如何将复杂的数学公式转化成实际的工程代码。
- **扩展性强**。Hamaa支持通过实现接口的方式来自定义模块。

更多链接：

- Hamaa使用文档 : [hamaa.readthedocs.io](http://hamaa.readthedocs.io)。
- Hamaa设计文档 : [完善中](FIXME)
- Hamaa已支持的所有特性 : [DEVELOP.md](DEVELOP.md)
- Hamaa的未来计划：[TODO.md](TODO.md)

--- 

### Installation

Hamaa使用了下述4个依赖库: 

```
numpy>=1.9
matplotlib>=1.5
nose>=1.3
Pillow>=3.4
```

打开`shell`或者`cmd`，输入下述命令

```
# 下载源代码到本地
>> git clone git@github.com:monitor1379/hamaa.git

# 输入下述命令，依次进行安装依赖、编译Hamaa中的Python C扩展、安装Hamaa
>> cd hamaa
>> pip install -r requirements.txt
>> python setup.py build_ext
>> pip install .
```


目前Hamaa仅支持：Python 2.7.

--- 

### Getting started: 1 minutes to Hamaa

在Hamaa中，一个神经网络模型被称为一个model，
其中最基础的一种model叫做`Sequential`，即将网络层按序列依次堆叠而成。

- 这是一个`Sequential`模型:
```python
from hamaa.models import Sequential

model = Sequential()
```

- 添加网络层只需使用`add()`:

```python
from hamaa.layers import Dense, Activation

model.add(Dense(input_dim=2, output_dim=3))
model.add(Activation('sigmoid'))
model.add(Dense(output_dim=2))
model.add(Activation('sigmoid'))
```

- 设置模型的目标(损失)函数以及优化器:
```python
from hamaa.optimizers import SGD

model.set_objective('categorical_crossentropy')
model.set_optimizer(SGD(lr=0.9, momentum=0.5))
```

- 打印模型的详细信息:
```python
print model.summary()
```

![print_model_summary](docs/images/README/print_model_summary.png)

- 加载数据集（此处采用moons数据集，为两个半弧形组成）,并切分其中0.9作为训练集，剩下0.1作为验证集
```python
from hamaa.datasets import datasets
from hamaa.utils.np_utils import split_training_data

x, y = datasets.load_moons_data(nb_data=2000, noise=0.1)
training_data, validation_data = split_training_data(data=(x, y), split_ratio=0.9)
```

- 接下来就可以开始训练模型:
```python
model.train(training_data=training_data,
			nb_epochs=10,
			mini_batch_size=100,
			validation_data=validation_data)
```

训练信息如下图所示:
![train](docs/images/README/train.png)

如果每个epoch耗时比较长，还可以使用进度条功能:
![train_gif](docs/images/README/train.gif)

- 训练完之后评估模型的准确率:
```python
print model.evaluate_accuracy(x, y) 
```

- 如果你想直观地查看模型的训练/验证准确率与损失函数值随着训练周期的变化图，可以:
```python
model.plot_training_iteration()
```

<p align="center">
	<img width=600 src="docs/images/README/epochs.png" alt="epochs" />
</p>


- 最后，如果数据集是二维数据，那么还可以画出决策边界:

```python
model.plot_prediction(data=training_data)
```

<p align="center">
	<img width=600 src="docs/images/README/prediction.png" alt="prediction" />
</p>

---

### Examples

更多样例程序:

#### examples/example1_or_nn.py

使用Hama构建单层神经网络来解决或问题的一个简洁的例子。
其中包含了创建、训练、测试一个神经网络必备的所有函数。

```python
from hamaa.datasets import datasets
from hamaa.layers import Dense, Activation
from hamaa.models import Sequential
from hamaa.optimizers import SGD


model = Sequential()                                        # 创建一个神经网络模型
model.add(Dense(input_dim=2, output_dim=2, init='uniform')) # 添加一个输入神经元数是2、输出神经元数是2的全连接层
model.add(Activation('sigmoid'))                            # 添加一个激活函数为sigmoid的激活层
model.set_objective('mse')                                  # 设置目标函数/损失函数为均方差
model.set_optimizer(SGD(lr=0.9, momentum=0.9, decay=1e-6))  # 设置优化器为随机梯度下降法

print model.summary()                                       # 打印模型的详细信息

x, y = datasets.load_or_data()                              # 加载数据
model.train(training_data=(x, y), nb_epochs=10)             # 开始训练，设置训练周期为10

print 'test accuracy: ', model.evaluate_accuracy(x, y)		# 评估模型的准确率
``` 

---

#### examples/example2_moons_nn.py

构建一个神经元数目为 [2->3->2] 的多层神经网络来对线性不可分数据集进行分类。

```python

from hamaa.layers import Dense, Activation
from hamaa.models import Sequential
from hamaa.datasets import datasets
from hamaa.utils import np_utils
from hamaa.optimizers import SGD

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
x, y = datasets.load_moons_data(nb_data=2000, noise=0.1)
# 切分数据集中的10%作为验证集
training_data, validation_data = np_utils.split_training_data(data=(x, y), split_ratio=0.9)

model.train(training_data=training_data,        # 设置训练集
            nb_epochs=10,                       # 设置训练周期
            mini_batch_size=100,                # 设置每次mini_batch的数据量
            verbose=1,                          # 设置训练过程显示方式，0代表不输出，1代表简单输出，2代表使用进图条功能
            validation_data=validation_data,    # 设置验证集
            log_epoch=1)                        # 设置每隔多少个周期才在控制台上显示一次训练过程的详细信息
print 'test accuracy: ', model.evaluate_accuracy(x, y)

model.plot_prediction(data=training_data)       # 对训练集进行分类的结果可视化
model.plot_prediction(data=validation_data)     # 对验证集进行分类的结果可视化
model.plot_training_iteration()                 # 画出训练过程中准确率和损失函数值随着训练周期的变化图

```

---

#### examples/example3_mnist_nn.py

构建一个多层神经网络来对MNIST数据集进行分类。
使用进度条功能来显示过程。

```python

from hamaa.datasets.datasets import load_mnist_data
from hamaa.layers import Dense, Activation
from hamaa.models import Sequential
from hamaa.optimizers import SGD
from hamaa.utils.np_utils import split_training_data


# 加载MNIST数据集，preprocess表示是否进行归一化预处理，flatten表示是否将二维图像平铺成一维
print '正在加载MNIST数据集...'
training_data, test_data = load_mnist_data(nb_training=60000, nb_test=10000, preprocess=True, flatten=True)
training_data, validation_data = split_training_data(training_data, split_ratio=0.95)

print 'training_data:', training_data[0].shape
print 'validation_data:', validation_data[0].shape
print 'test_data:', test_data[0].shape

# 构建一个每层神经元数为 [784->100->10] 的神经网络
model = Sequential()
model.add(Dense(input_dim=784, output_dim=100, init='glorot_normal'))
model.add(Activation('sigmoid'))
model.add(Dense(output_dim=10, init='glorot_normal'))
model.add(Activation('sigmoid'))
model.set_objective('categorical_crossentropy')
model.set_optimizer(SGD(lr=0.2, momentum=0.2, decay=1e-3))

print model.summary()

model.train(training_data=training_data,
            nb_epochs=10,
            mini_batch_size=100,
            verbose=2,  # 使用进图条功能
            validation_data=validation_data,
            log_epoch=1
            )

print 'test accuracy:', model.evaluate_accuracy(test_data[0], test_data[1])
model.plot_training_iteration()
```

---

#### examples/examples4_mnist_cnn.py

构建一个卷积神经网络来对MNIST数据集进行分类。
使用进度条功能来显示过程，并使用可视化工具对卷积层
权重进行可视化。

```python

from hamaa.datasets.datasets import load_mnist_data
from hamaa.layers import Dense, Activation, Convolution2D, Flatten, MeanPooling2D
from hamaa.models import Sequential
from hamaa.optimizers import SGD
from hamaa.utils import vis_utils
from hamaa.utils.np_utils import split_training_data


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

```

在MNIST数据集上测试准确率达到99.3%的卷积神经网络卷积核可视化示意图：
(**注**: 非来自上述程序)

<p align="center">
      <img width=200 src="docs/images/README/cnn_first_layer_weight.png" alt="kernel" />
</p>


### dev branch

[![Documentation Status](https://readthedocs.org/projects/hamaa/badge/?version=latest)](http://hamaa.readthedocs.io/zh_CN/latest/?badge=latest) 
[![Build Status](https://travis-ci.org/monitor1379/hamaa.svg?branch=dev)](https://travis-ci.org/monitor1379/hamaa)
[![codecov](https://codecov.io/gh/monitor1379/hamaa/branch/dev/graph/badge.svg)](https://codecov.io/gh/monitor1379/hamaa)

