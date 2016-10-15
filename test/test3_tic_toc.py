# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test3_tic_toc.py
@time: 2016/10/11 15:26


"""

from hamaa.utils.time_utils import tic, toc
from datetime import datetime

def run():
    t0 = datetime.now()
    a = []
    for i in range(100000):
        a.append(i)
    t1 = datetime.now()
    t = t1 - t0

    sec = t.seconds + (t.microseconds / 1000000.0)
    print sec


if __name__ == '__main__':
    run()