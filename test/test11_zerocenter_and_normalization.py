# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test11_zerocenter_and_normalization.py
@time: 2016/9/28 19:39


"""

import numpy as np
import matplotlib.pyplot as plt


def test1():
    # c1 = np.random.randn(100, 2) + 3
    # c2 = np.random.randn(100, 2)
    # c = np.vstack((c1, c2))
    c = np.array([[3, 1], [3, 2], [3, 3]])



    plt.grid('on')
    plt.xlim([-7, 7])
    plt.ylim([-7, 7])
    plt.scatter(c[:, 0], c[:, 1], c='b', cmap=plt.cm.Spectral)

    a = c - np.mean(c, axis=0)
    a /= np.std(c, axis=0)
    plt.scatter(a[:, 0], a[:, 1], c='r', cmap=plt.cm.Spectral)

    b = c - np.mean(c, axis=0)
    b /= np.std(b)
    print b.min(), b.max()
    plt.scatter(b[:, 0], b[:, 1], c='g', cmap=plt.cm.Spectral)



    plt.show()




def run():
    test1()

if __name__ == '__main__':
    run()