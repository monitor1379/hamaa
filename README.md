<p align="center">
	<img width=500 src="https://rawgit.com/monitor1379/hamaa/dev/docs/images/hamaa-logo.svg" />
</p>

# Hamaa

[![Documentation Status](https://readthedocs.org/projects/hamaa/badge/?version=latest)](http://hamaa.readthedocs.io/zh_CN/latest/?badge=latest) 
[![Build Status](https://travis-ci.org/monitor1379/hamaa.svg?branch=master)](https://travis-ci.org/monitor1379/hamaa)
[![codecov](https://codecov.io/gh/monitor1379/hamaa/branch/master/graph/badge.svg)](https://codecov.io/gh/monitor1379/hamaa)

## What is Hamaa

Hamaa是一个构建于Python/NumPy之上的深度学习库，
采用类[Keras](http://keras.io)接口，
拥有与Keras一样极简和高度模块化的特点。

Hamaa库的出现并不是为了取代工业界上常用的深度学习框架
(诸如
[TensorFlow](https://www.tensorflow.org/)、
[Theano](http://www.deeplearning.net/software/theano/)、
[Caffe](http://caffe.berkeleyvision.org/)、
[MXNet](http://mxnet.readthedocs.io/en/latest/)
等等)。

相反，Hamaa被创造的目的是希望通过提供一份简单、朴素的深度学习框架源代码，
来让深度学习的初学者能够通过阅读Hamaa的源代码甚至重现Hamaa来加深对神经网络/深度学习的理解，
以便未来更好地去运用上述几个工业级别的深度学习框架（而不是成为一个调包/调参侠 :D）。

因此，Hamaa始终遵循着Simple and Naive的原则来设计，具体地：

- 每一个可配置项都抽象成可相互组合的模块。
具体而言，网络层、损失函数、优化器、初始化策略、激活函数都是独立的模块，你可以使用它们来构建自己的模型。

- 每个模块都希望尽量的简洁。每一段代码都应该在初次阅读时都显得直观易懂。
并且没有任何的黑魔法，因为它将给阅读和理解带来麻烦。

不可避免地，Hamaa在速度优化上远没有工业级别的深度学习框架要好，
但如果你想要进行一些简单的快速实验，又不想花大量时间在搭建与配置环境上的话，
那么基于Python的Hamaa无疑是更好的选择。

# Getting Started with Hamaa


# Usage 

# License

[![GitHub license](https://img.shields.io/badge/license-AGPL-blue.svg)](https://raw.githubusercontent.com/monitor1379/hamaa/master/LICENSE)

### dev branch

[![Documentation Status](https://readthedocs.org/projects/hamaa/badge/?version=latest)](http://hamaa.readthedocs.io/zh_CN/latest/?badge=latest) 
[![Build Status](https://travis-ci.org/monitor1379/hamaa.svg?branch=dev)](https://travis-ci.org/monitor1379/hamaa)
[![codecov](https://codecov.io/gh/monitor1379/hamaa/branch/dev/graph/badge.svg)](https://codecov.io/gh/monitor1379/hamaa)
