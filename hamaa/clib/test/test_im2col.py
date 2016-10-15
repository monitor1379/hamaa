import sys
sys.path.append("..")

from clib import im2colutils
import numpy as np
from datetime import datetime

N, C, H, W = (2, 2, 3, 4)
KN, KC, KH, KW = (1, C, 3, 3)
stride = 1

x = np.arange(N * C * H * W).reshape(N, C, H, W).astype(np.double)
w = np.ones((KN, KC, KH, KW)).astype(np.double)

print x
print w

print im2colutils.im2col_NCHW_memcpy(x, w.shape[2], w.shape[3], stride)
