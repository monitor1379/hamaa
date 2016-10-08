# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test10_plot_beautiful_map.py
@time: 2016/10/4 21:30


"""

import numpy as np
import matplotlib.pyplot as plt



def run():
    a = np.random.randn(1, 1, 5, 5)
    plt.axis('off')
    plt.imshow(a[0][0], cmap='gray_r',interpolation='None')

    plt.show()


if __name__ == '__main__':
    run()