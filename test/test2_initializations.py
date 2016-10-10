# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test2_initializations.py
@time: 2016/10/10 22:13


"""

from hamaa import initializations
import numpy as np
import matplotlib.pyplot as plt

def run():
    n = 10000
    nb_in = 100
    nb_out = 10
    x = np.random.normal(loc=0.0, scale=1.0, size=(n, nb_in))
    w = initializations.glorot_normal(shape=(nb_in, nb_out))
    z = np.dot(x, w)
    print z.mean(), z.std(), z.var()

    dz = np.random.normal(loc=0.0, scale=1.0, size=z.shape)
    dx = np.dot(dz, w.T)
    print dx.mean(), dx.std(), dx.var()





if __name__ == '__main__':
    run()