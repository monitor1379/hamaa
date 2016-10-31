#include "D:/Anaconda2/include/Python.h"
#include "D:/Anaconda2/Lib/site-packages/numpy/core/include/numpy/arrayobject.h"

#define SUCCESS 1
#define FAIL 0

// 检查numpy.ndarray对象的维度是否是ndim
int check_ndim(PyArrayObject *x, int ndim, PyObject *error);

// 检查numpy.ndarray对象的dtype是否为numpy.double
int check_dtype_is_double(PyArrayObject *x, PyObject *error);

// 检查大小为H*W的二维矩阵能否被大小为KH*KW的卷积核
// 以步长为stride进行完整的卷积
int check_can_be_convolved(int H, int W, int KH, int KW,
                              int stride, PyObject *error);


// 检查im2col_HW的结果columnize_x是否和卷积核的大小KH*KW与
// 卷积结果conv_x的大小CH*CW相符
int check_columnize_x_shape_HW(int OH, int OW, int KH, int KW,
                               int CH, int CW, PyObject *error);


// 检查im2col_NCHW的结果columnize_x是否和卷积核的大小KH*KW与
// 卷积结果conv_x的大小CH*CW相符
int check_columnize_x_shape_NCHW(int OH, int OW, int KH, int KW,
                                 int CH, int CW, PyObject *error);
