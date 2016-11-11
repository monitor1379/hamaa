# Hamaa 中文文档

<p align="center">
	<img width=500 src="images/hamaa-logo.svg" />
</p>


## What is Hamaa

### 没错，你刚发现了Hamaa!


Hamaa是专为研究人员与有一定深度学习基础的初学者设计的一款深度学习库。
它构建于Python/NumPy之上，
通过高度的抽象与模块化使得用Hamaa来搭建神经网络就像搭建积木一样简单。
研究人员能够通过使用Hamaa来减少实际开发中的重复工作并加快实验速度。

此外，Hamaa遵循简单与朴素的设计原则，并提供了充足的示例与教程来帮助初学者应用深度学习，
因此对于初学者来说它的源代码十分适合进行学习。

---

## Why build Hamaa

如你所见，Hamaa的出现**并不是为了取代(当然也没有这个能力去取代)工业界上常用的深度学习框架**
(诸如
[TensorFlow](https://www.tensorflow.org/)、
[Theano](http://www.deeplearning.net/software/theano/)、
[Caffe](http://caffe.berkeleyvision.org/)、
[MXNet](http://mxnet.readthedocs.io/en/latest/)
等等)。
相反，Hamaa被创造的目的是**希望让深度学习的初学者能够通过使用、
阅读Hamaa的源代码甚至去重现Hamaa来加深对神经网络/深度学习的理解**，
以便更好地去使用上述几个工业级别的深度学习框架去解决实际问题（而不是成为一个调包/调参侠 ：D）。

如果你是深度学习的老司机并追求更复杂的模型与更快的速度，那么强烈建议你使用上述几个更加强大的深度学习库。


!!! Note
	我们建议你在[Github](https://github.com/monitor1379/hamaa)上star和watch 官方项目，
	这样当官方有更新时，你会立即知道。

---

## The Design Philosophy of Hamaa

Hamaa始终遵循着`too SIMPLE and sometimes NAIVE`的原则来设计：

- **每一个可配置项都抽象成简单的模块**。 具体地，网络层、损失函数、优化器、初始化器、激活函数都是独立
的模块，能够通过自由组装的方式来搭建模型。

- **所有模块都使用朴素的代码实现**。每一段源代码都希望能在第一次阅读时显得直观易懂，具有良好的可读性，
并且不过分使用trick。

Hamaa尽管在速度优化上没有工业级别的深度学习框架要好，
但Hamaa也具有以下优点:

- **安装依赖少，无需配置**。如果你想要进行一些简单的快速实验，又不想花大量时间在搭建与配置环境上的话， 
那么基于Python的Hamaa可能是不错的选择。安装教程链接: [安装 Installation](user/installation.md)。

- **使用简单**。学会使用Hamaa搭建一个完整的神经网络并进行数据可视化仅需不到1分钟。
此外如果你熟悉[Keras](http://keras.io)的话将能更快地上手。快速教程链接: [教程 Tutorials](user/tutorials.md)

- **源码易读**。朴素的设计模式、丰富的源码注释以及完备的设计文档能够让你深入理解Hamaa是如何将复杂的数学公式转化成实际的工程代码。

- **扩展性强**。Hamaa支持通过实现接口的方式来自定义模块。

---

## Future about Hamaa

Hamaa仍处于开发阶段，代码与文档仍然有要完善的地方。1.0 版本计划在未来几周内发布。
如果您想参与到项目中可以查看: [开发 Development](developer/development.md)。
