# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: time_utils.py
@time: 2016/9/28 17:06

简易的性能分析工具
"""

import time


class T(object):
    events = []
    starts = []
    debug = True


def tic(event=None):
    if T.debug:
        T.events.append(event)
        T.starts.append(time.time())


def toc(verbose=True):
    if T.debug:
        end = time.time()
        event = T.events.pop()
        start = T.starts.pop()
        t = '%fs' % (end - start)
        if verbose:
            if event:
                print '@%s, time: %s' % t
            else:
                print 'time: %s' % t
        return t


