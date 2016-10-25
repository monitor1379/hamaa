# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: test3_MeanPooling2D.py
@time: 2016/10/21 16:53


"""

from hamaa.layers import *
from hamaa.models import Sequential
from hamaa.datasets import datasets
from hamaa.optimizers import SGD
from hamaa.utils.np_utils import *
from hamaa.utils.conv_utils import *
from hamaa.utils.time_utils import *


class MeanPooling2DGate:
    """均值池化计算单元"""

    @staticmethod
    def forward(im, pool_size):
        if pool_size[0] != pool_size[1]:
            raise RuntimeError('pool_size:{} is not equal!'.format(pool_size))
        H, W = im.shape
        KH, KW = pool_size
        CH, CW = H / KH, W / KW
        tmp = im2col_HW(im, KH, KW, KH)
        return np.mean(tmp, axis=0).reshape(im.shape[0] / pool_size[0], im.shape[1] / pool_size[1])

    @staticmethod
    def backward(pool_size, d_pim):
        return np.kron(d_pim, np.ones(pool_size)) / (pool_size[0] * pool_size[1])

def run():
    N, C, H, W = 10, 10, 40, 40
    pool_size = [2, 2]
    x = np.arange(N*C*H*W).reshape(N, C, H, W).astype(np.double)
    layer = MeanPooling2D(pooling_size=pool_size, input_shape=(C, H, W))
    layer.build()

    z = layer.forward(x)
    d_z = z

    itertimes = 100
    tic('hamaa 5')
    for times in xrange(itertimes):
        d_x = layer.backward(d_z)
    toc(True)


    tic('old')
    for times in xrange(itertimes):
        d_t = MeanPooling2DGate.backward(pool_size, d_z)
    toc(True)

    print sum_abs_err(d_t, d_x)



    return
    d_z = z
    d_x = layer.backward(d_z)
    grad_x = eval_numerical_gradient_array(layer.forward, x, d_z)
    print d_x
    print grad_x

    print sum_abs_err(d_x, grad_x)


if __name__ == '__main__':
    run()