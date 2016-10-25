# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test6_progressbar.py
@time: 2016/10/24 19:25


"""



from hamaa.utils.time_utils import *

def run():
    total = 5
    bar = ProgressBar(total=total, width=20)

    bar.show('epoch:' + str(0))
    time.sleep(1)
    for i in range(total):
        bar.move()
        bar.show('epoch:' + str(i + 1))
        time.sleep(1)


if __name__ == '__main__':
    run()
