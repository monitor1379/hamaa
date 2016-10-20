import sys
sys.path.append("..")

from clib import im2colutils, im2rowutils, col2imutils
import numpy as np
from datetime import datetime


N, C, H, W = (2, 1, 3, 4)
KN, KC, KH, KW = (1, C, 3, 3)
stride = 1
CH = (H - KH) / stride + 1;
CW = (W - KW) / stride + 1;


x = np.arange(N * C * H * W).reshape(N, C, H, W).astype(np.double)
w = np.ones((KN, KC, KH, KW)).astype(np.double)

#x = x[0][0]
w = w[0][0]


print x
print w



columnize_x = im2colutils.im2col_NCHW(x, KH, KW, stride)
print columnize_x

stride = 1


re_x = col2imutils.col2im_NCHW(columnize_x, KH, KW, CH, CW, stride)
print re_x









