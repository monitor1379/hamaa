# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: vis_utils.py
@time: 2016/11/1 21:19


"""

import matplotlib.pyplot as plt
import numpy as np


def visualize_convolution_kernel(conv_layer, title='convolution layer visualizing'):
    plt.figure(title)
    N, _, _, _ = conv_layer.w.shape
    row = int(np.ceil(np.sqrt(N)))
    col = row
    for i in range(N):
        plt.subplot(row, col, i + 1)
        plt.axis('off')
        plt.imshow(conv_layer.w[i][0], cmap='gray', interpolation='nearest')
    plt.show()
