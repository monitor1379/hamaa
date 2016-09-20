# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: mnist_decoder.py
@time: 2016/8/16 20:03

对MNIST手写数字数据文件转换为bmp图片文件格式。
数据集下载地址为http://yann.lecun.com/exdb/mnist。
相关格式转换见官网以及代码注释。

========================
关于IDX文件格式的解析规则：
========================
THE IDX FILE FORMAT

the IDX file format is a simple format for vectors and multidimensional matrices of various numerical types.
The basic format is

magic number
size in dimension 0
size in dimension 1
size in dimension 2
.....
size in dimension N
data

The magic number is an integer (MSB first). The first 2 bytes are always 0.

The third byte codes the type of the data:
0x08: unsigned byte
0x09: signed byte
0x0B: short (2 bytes)
0x0C: int (4 bytes)
0x0D: float (4 bytes)
0x0E: double (8 bytes)

The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....

The sizes in each dimension are 4-byte integers (MSB first, high endian, like in most non-Intel processors).

The data is stored like in a C array, i.e. the index in the last dimension changes the fastest.
"""

import struct
import os
import numpy as np
import matplotlib.pyplot as plt

# 当前路径
module_path = os.path.dirname(__file__)
# 训练集文件
train_images_idx3_ubyte_file = module_path + '/bin/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = module_path + '/bin/train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = module_path + '/bin/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = module_path + '/bin/t10k-labels.idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file, num_data=-1):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :param num_data: 指定读取数据数量，-1表示读取全部
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、数据数量、每个数据的维度大小
    offset = 0
    fmt_header = '>iiii'
    magic_number, real_num_data, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    # print '魔数:%d, 数据数量: %d个, 数据维度: %d*%d' % (magic_number, real_num_data, num_rows, num_cols)

    # 获取指定数量的数据出来
    if num_data == -1:
        num_data = real_num_data
    else:
        num_data = min(num_data, real_num_data)

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_data, num_rows, num_cols))

    for i in range(num_data):
        # if (i + 1) % 10000 == 0:
        #     print '已解析 %d' % (i + 1) + '张'
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file, num_data=-1):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :param num_data: 指定读取数据数量，-1表示读取全部
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, real_num_data = struct.unpack_from(fmt_header, bin_data, offset)
    # print '魔数:%d, 数据数量: %d个' % (magic_number, real_num_data)

    # 获取指定数量的数据出来
    if num_data == -1:
        num_data = real_num_data
    else:
        num_data = min(num_data, real_num_data)

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_data, dtype=np.int)
    for i in range(num_data):
        # if (i + 1) % 10000 == 0:
        #     print '已解析 %d' % (i + 1) + '张'
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file, num_data=-1):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :param num_data: 指定读取数据数量，-1表示读取全部出来
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file, num_data)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file, num_data=-1):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :param num_data: 指定读取数据数量，-1表示读取全部出来
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file, num_data)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file, num_data=-1):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :param num_data: 指定读取数据数量，-1表示读取全部出来
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file, num_data)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file, num_data=-1):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :param num_data: 指定读取数据数量，-1表示读取全部出来
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file, num_data)



def run():
    train_images = load_train_images(num_data=1000)
    train_labels = load_train_labels(num_data=1000)

    # train_images = load_test_images(num_data=20000)
    # train_labels = load_test_labels(num_data=20000)

    print train_images.shape, train_labels.shape

    # for i in range(10):
    #     print train_labels[i]
    #     plt.imshow(train_images[i], cmap='gray')
    #     plt.show()
    print 'done'

if __name__ == '__main__':
    run()