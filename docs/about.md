# 关于 About

## 作者：关于Hamaa的诞生

Hamaa源自于我当初的一个小想法。
当时我学习Deep Learning已有两个月，看了很多论文、教程与博客，
于是尝试着去阅读Keras的源代码来学习别人是怎么实现的，尤其是back propagation这一块。
但是Keras的backend使用的是Theano/TensorFlow，
这两个深度学习库都是“符号主义”派，这意味着神经网络的求导是自动的(autograd)。

所以最后还是决定硬啃论文和数学公式来实现，写着写着发现代码越来越多，添加一个网络层动辄就要修改数十行代码。突然某一天想到，为什么不学习Keras的API设计呢？于是在不断的重构中逐渐实现了模块化，也就有了现在的Hamaa。

Hamaa吸收了许多开源深度学习库的设计理念，比如Keras的API，Caffe的Blob/Layer/Net/Solver架构，
TensorFlow/Theano的Operator求导机制（Hamaa中为手动实现Operator的forward/backward以实现自动求导）等等。

而我很高兴地说，在实现Hamaa的过程中，深入了解到了：
