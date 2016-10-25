[![badge]("https://readthedocs.org/projects/hamaa/badge/?version=latest")]("http://hamaa.readthedocs.io/zh_CN/latest/?badge=latest")

# Hamaa 前言

以文档主要面向Hamaa的开发者而不是使用者，
想了解更多关于Hamaa的使用教程与最新消息可关注
[github.com/monitor1379/hamaa](http://github.com/monitor1379/hamaa)。


# Hamaa-0.5 设计

- [layers.py](#layers_py)
- [models.py](#models_py)

## <span id="layers_py">layers.py</span>

### Layer

#### 介绍

所有层的顶层抽象类，所有网络层都继承自该层。

#### 成员变量

成员变量如下表所示：

| 成员变量 | 数据类型  |  含义 |
|:-------:|:-------:|:-------|
| layer_type | str  |  静态成员，表示该层的类型，一般为类名 |
||||
||||
| input | numpy.ndarray | 该层的输入数据 |
| output |  numpy.ndarray  | 该层的输出数据|
| d_input | numpy.ndarray | 损失函数对该层输入input的梯度 |
| d_output | numpy.ndarray | 损失函数对该层输出output的梯度 |
| input_shape | list | 输入数据的形状 |
| output_shape | list | 输出数据的形状 |
||||
||||
| trainable | True / False | 表示该层是否可训练 |
| trainable_params | list | 该层的所有可训练参数必须放在该列表中， 比如[w, b] |
| grads | list | 该层的所有可训练参数的梯度必须放在该列表中，且保存顺序需与trainable_params中的一致，比如[d_w, d_b] |
||||
||||
| previous_layer | Layer | 该层的前一层 |
| latter_layer | Layer | 该层的后一层 |
||||
||||
| mode | str | 取值为"train"时，调用该层的forward方法会保存中间计算变量，而取值为"test"时只会保留output计算结果 |
| mid | dict | 保存中间计算结果。注意：该变量的类内访问需要通过save与take方法，save方法根据当前mode不同来决定是否真正保存该变量以减少内存消耗 |
| config | dict | 保存该层的网络配置信息 |



#### 成员方法

成员方法如下表所示：

| 成员方法 | 含义 |
|:-------|:-------|
| `__init__`(**kwargs) | 网络层的构造方法 |
| build() | 在构造层的时候，某些成员变量可能需要已知前一层的某些信息才能进行初始化，这些初始化操作可以放在build方法中。由于layer由model进行管理，layer之间的连接都是在model中进行的，所以build方法也是由model进行调用。 |
| forward(input) | 根据网络层的输入input来计算网络层的输出output。该方法是一个wrapper，子类只需要实现`forward_train`以及`forward_test`方法即可。 |
| forward_train(input) | 以train模式来前向计算。需要保留backward中要用到的中间计算结果。一般来说，采取`mini_batch`方式训练的话input的数据量都比较小 |
| forward_test(input) | 以test模式来前向计算。不需要保留backward中要用到的中间计算结果。一般来说，test的过程是为了计算整个网络对于某数据集的accuracy，数据集可能比较大（比如MNIST训练集有60000张），所以需要使用不耗内存的方式来进行计算，且需要及时清理不需要用到的中间计算变量。 |
| backward(d_output) | 根据后一层传过来的误差项 d_output，来计算该层传播到前一层的误差项d_input |
| summary() | 以字符串形式返回该层的信息 |

### <span id="models_py">models.py</span>


