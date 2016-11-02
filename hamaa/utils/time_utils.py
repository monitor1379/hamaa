# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: time_utils.py
@time: 2016/9/28 17:06

简易的性能分析工具，以及控制台进度条工具ProgressBar类
"""

import sys
import time
import math


class T(object):
    events = []
    starts = []


def tic(event=None):
    T.events.append(event)
    T.starts.append(time.time())


def toc(verbose=False):
    end = time.time()
    event = T.events.pop()
    start = T.starts.pop()
    t = end - start
    if verbose:
        if event:
            print '@%s, time: %fs' % (event, t)
        else:
            print 'time: %fs' % t
    return t


class ProgressBar:
    """控制台进度条类"""

    def __init__(self, total=0, width=20):
        """
        :param total: 进度最大值，整数类型
        :param width: 进图条显示宽度，单位为一个字符占位格
        """
        # 当前轮数，类型为整数值
        self.current = 0
        self.total = total
        self.width = width
        self.timestamp = None
        self.one_move_time = 0

    def move(self, step=1):
        self.current += step
        if self.timestamp:
            self.one_move_time = (time.time() - self.timestamp) / step
        self.timestamp = time.time()

    def clear(self):
        """清空进度条"""
        sys.stdout.write('\r')
        sys.stdout.write(' ' * (100))
        sys.stdout.write('\r')
        sys.stdout.flush()

    def reset(self):
        """重置进度条进度"""
        self.current = 0

    def show(self, head='', message=''):
        self.clear()
        # 计算当前进度百分比
        percent = self.current * 1.0 / self.total
        # 根据百分比以及进度条总长度，计算当前进度的占位符宽度
        progress = int(math.ceil(self.width * percent))
        if progress == self.width:
            progress -= 1
        # 估计剩余耗时
        remaining_time = (self.total - self.current) * self.one_move_time
        # 打印进度条头
        sys.stdout.write(head + ',  {0:3}/{1}: '.format(self.current, self.total))
        # 打印进度条
        sys.stdout.write('[' + '='*progress + '>' + ' '*(self.width - progress - 1) + '] remaining time: %.2fs' % remaining_time)
        # 打印尾随信息
        sys.stdout.write(message)
        sys.stdout.flush()

        # 如果进度完成，则输出换行符（永久打印在控制台上）
        # if self.current == self.total:
        #     sys.stdout.write('\n')
        #     sys.stdout.flush()




