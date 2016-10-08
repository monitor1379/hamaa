# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test7_timer.py
@time: 2016/9/28 17:07


"""

from utils.time_utils import tic, toc
from line_profiler import LineProfiler


def test1():
    tic('fuck')
    a = []
    for i in range(10000000):
        pass
    toc()


def run():
    tic('run')
    test1()
    toc()

if __name__ == '__main__':
    run()