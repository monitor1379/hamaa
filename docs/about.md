# 关于 About

## 作者：关于Hamaa的诞生

Hamaa源自于我当初的一个小想法。
当时我学习Deep Learning已有两个月，看了很多论文、教程与博客，
于是尝试着去阅读Keras的源代码来学习别人是怎么实现的，尤其是back propagation这一块。
但是Keras的backend使用的是Theano/TensorFlow，
这两个深度学习库都是“符号主义”派，这意味着神经网络的求导是自动的(autograd)。

所以最后还是决定硬啃论文和数学公式来重现，写着写着发现代码越来越多，添加一个网络层动辄就要修改数十行代码。突然某一天想到，为什么不学习Keras的API设计呢？于是在不断的重构中逐渐实现了模块化，也就有了现在的Hamaa。

Hamaa吸收了许多开源深度学习库的设计理念，比如Keras的API，Caffe的Blob/Layer/Net/Solver架构，
TensorFlow/Theano的Operator求导机制（Hamaa中为手动实现Operator的forward/backward以实现自动求导）等等。

而我很高兴地说，在实现Hamaa的过程中，我深入了解与学习到了以下几点：

1. 彻底弄懂了神经网络中全连接层、激活层、卷积层、池化层的backpropagation过程及其向量化（vectorization）实现。

2. 了解到了softmax输出层为什么通常配合cross entropy损失函数以及negative log likelihood优化方法一起使用。

- 了解到了神经网络权重初始化的原因与各种方法。

- 学会了梯度下降法（Stochastic Gradient Descent）优化方法中，learning rate、momentum与decay参数对收敛速度与收敛稳定性的影响。

- 有了一定的CNN调参经验。

- 学会了卷积计算的加速方法: im2col与col2im。

- 了解到TensorFlow的NHWC数据格式与Theano的NCHW数据格式之间的差异性。

- 弄懂了在训练卷积神经网络时影响速度与内存的因素。

- 学会了如何编写Python C Extension。

- 学会使用以下工具链来发布一个完整的库：
	- Python工具：
		- distutils：编译Python扩展
		- setuptools：分发包
		- nose：测试
		- pip：包管理
		- virtualenv：虚拟环境
		- coverage：代码覆盖率统计
	- 文档编写工具：
		- Sphinx：用reStructuredText写文档
		- MkDocs：用Markdown写文档
	- GitHub webhook：
		- Readthedocs：文档托管
		- Travis-CI：集成测试托管
		- Codecov：代码覆盖率统计托管

鉴于我水平有限，在某些实现上难免会出现不足或错误之处。
如果您发现了，十分欢迎在GitHub上提出[issues](https://github.com/monitor1379/hamaa)或者发邮件到作者邮箱:yy4f5da2(at)hotmail.com。
