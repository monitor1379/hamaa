import numpy as np
import conv
from datetime import datetime

a = np.arange(10000).reshape(100, 100).astype(np.double)

ts = 0


#output = conv.im2col(a, 2, 2, 1, 99, 99)
for i in range(1000):
    t0 = datetime.now()
    output = conv.im2col(a, 2, 2, 1, 99, 99)
    t1 = datetime.now()
    dt = t1 - t0
#    print 'time: ', dt.seconds, 's, ', dt.microseconds / 1000.0, 'ms.'
    dtms = dt.seconds + dt.microseconds / 1000000.0
#    print 'time: %lfs' % dtms
    ts += dtms
print a
print output
print ts
