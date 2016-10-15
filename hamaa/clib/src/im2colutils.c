#include "D:/Anaconda2/include/Python.h"
#include "D:/Anaconda2/Lib/site-packages/numpy/core/include/numpy/arrayobject.h"

PyArrayObject *im2col_HW(PyArrayObject *x, int KH, int KW, int stride)
{
    assert(PyArray_TYPE(x) == NPY_DOUBLE);
    // 获取二维矩阵x的形状
    int H = PyArray_DIM(x, 0);
    int W = PyArray_DIM(x, 1);

    // 计算x与w进行卷积后的结果的形状
    int CH = (H - KH) / stride + 1;
    int CW = (W - KW) / stride + 1;

    // 将x进行2维im2col展开后的col_x的大小
    int IH = KH * KW;
    int IW = CH * CW;

    // 创建numpy.ndarray对象col_x，即im2col展开后的x
    int nd = 2;
    npy_intp dims[] = {IH, IW};
    PyArrayObject *col_x = PyArray_SimpleNew(nd, dims, NPY_DOUBLE);

    // 用来表示卷积结果中的第crow行第ccol列
    int crow, ccol;

    // 用来表示x中的第row行第col列
    int row, col;

    // 用来表示当w在x上作滑动卷积时，w的左上角在x中的位置
    int row_start, col_start;

    // 用来表示im2col展开x后col_x中的第irow行第icol列
    int irow, icol;

    // 当w在x上滑动到某一个位置时，用i与j来遍历x与w重叠区域的数据
    int i, j;

    // 提前进行类型转换，因为PyArray_DATA()与PyArray_GETPTR1/2/3/4()
    // 两个宏所获取的数据指针都是void *类型
    npy_double *x_data = PyArray_DATA(x);
    npy_double *col_x_data = PyArray_DATA(col_x);

    // 开始im2col过程
    icol = 0;
    for(crow = 0; crow < CH; ++crow)
    {
        for(ccol = 0; ccol < CW; ++ccol)
        {
            row_start = crow * stride;
            col_start = ccol * stride;
            irow = 0;
            for(i = row_start; i < row_start + KH; ++i)
            {
                for(j = col_start; j < col_start + KW; ++j)
                {
                    col_x_data[irow * IW + icol] = x_data[i * W + j];
                    // 如果不提前对所有数据进行显式数据类型转换
//                    *((npy_double *)(PyArray_GETPTR2(col_x, irow, icol))) = \
//                    *((npy_double *)(PyArray_GETPTR2(x, i, j)));
                    ++irow;
                }
            }
            ++icol;
        }
    }
    return col_x;
}

PyArrayObject *im2col_NCHW(PyArrayObject *x, int KH, int KW, int stride)
{
    assert(PyArray_TYPE(x) == NPY_DOUBLE);
    // 获取四维矩阵x的形状
    int N = PyArray_DIM(x, 0);
    int C = PyArray_DIM(x, 1);
    int H = PyArray_DIM(x, 2);
    int W = PyArray_DIM(x, 3);

    // 计算x与w进行卷积后的结果的形状
    int CH = (H - KH) / stride + 1;
    int CW = (W - KW) / stride + 1;

    // 将x进行2维im2col展开后的col_x的大小
    int IH = C * KH * KW;
    int IW = N * CH * CW;

    // 创建numpy.ndarray对象col_x，即im2col展开后的x
    int nd = 2;
    npy_intp dims[] = {IH, IW};
    PyArrayObject *col_x = PyArray_SimpleNew(nd, dims, NPY_DOUBLE);


    // 用来表示卷积结果中的第crow行第ccol列
    int crow, ccol;

    // 用来表示x中的第row行第col列
    int row, col;

    // 用来表示当w在x上作滑动卷积时，w的左上角在x中的位置
    int row_start, col_start;

    // 用来表示im2col展开x后col_x中的第irow行第icol列
    int irow, icol;

    // 用来表示当x[n][c]进行im2col展开时，保存在col_x中的左上角坐标
    int irow_start, icol_start;

    // 当w在x上滑动到某一个位置时，用i与j来遍历x与w重叠区域的数据
    int i, j;

    // 用来表示x中的第n个图片的第c个通道
    int n, c, n_start, c_start;

    // 提前进行类型转换，因为PyArray_DATA()与PyArray_GETPTR1/2/3/4()
    // 两个宏所获取的数据指针都是void *类型
    npy_double *x_data = PyArray_DATA(x);
    npy_double *col_x_data = PyArray_DATA(col_x);

    // 对x中的第n个图片的第c个通道进行im2col展开
    for(n = 0; n < N; ++n)
    {
        for(c = 0; c < C; ++c)
        {
            // 将x[n][c]展开后的im2col矩阵在col_x中的位置的左上角坐标
            irow_start = c * KH * KW;
            icol_start = n * CH * CW;

            // 开始对x[n][c]进行im2col过程
            // 在im2col_HW函数中，此处为icol = 0;
            icol = icol_start;
            for(crow = 0; crow < CH; ++crow)
            {
                for(ccol = 0; ccol < CW; ++ccol)
                {
                    // 在im2col_HW函数中，此处为irow = 0;
                    irow = irow_start;
                    row_start = crow * stride;
                    col_start = ccol * stride;
                    for(i = row_start; i < row_start + KH; ++i)
                    {
                        for(j = col_start; j < col_start + KW; ++j)
                        {
                            col_x_data[irow * IW + icol] = x_data[n*C*H*W + c*H*W + i*W+j];

                            // 如果不提前对所有数据进行显式数据类型转换
                            // 则需要在每次取数据时进行类型转换
//                            *((npy_double *)(PyArray_GETPTR2(col_x, irow, icol))) = \
//                            *((npy_double *)(PyArray_GETPTR4(x, n, c, i, j)));
                            ++irow;
                        }
                    }
                    ++icol;
                }
            }
        }
    }

    return col_x;
}



PyArrayObject *im2col_NCHW_memcpy(PyArrayObject *x, int KH, int KW, int stride)
{
    assert(PyArray_TYPE(x) == NPY_DOUBLE);
    // 获取四维矩阵x的形状
    int N = PyArray_DIM(x, 0);
    int C = PyArray_DIM(x, 1);
    int H = PyArray_DIM(x, 2);
    int W = PyArray_DIM(x, 3);

    // 计算x与w进行卷积后的结果的形状
    int CH = (H - KH) / stride + 1;
    int CW = (W - KW) / stride + 1;

    // 将x进行2维im2col展开后的row_x的大小
    int IH = N * CH * CW;
    int IW = C * KH * KW;

    // 创建numpy.ndarray对象row_x，即使用im2col展开x时col_x的转置
    int nd = 2;
    npy_intp dims[] = {IH, IW};
    PyArrayObject *row_x = PyArray_SimpleNew(nd, dims, NPY_DOUBLE);

    // 真正的im2col结果
    PyArrayObject *col_x;

    // 用来表示卷积结果中的第crow行第ccol列
    int crow, ccol;

    // 用来表示x中的第row行第col列
    int row, col;

    // 用来表示当w在x上作滑动卷积时，w的左上角在x中的位置
    int row_start, col_start;

    // 用来表示im2col展开x后row_x中的第irow行第icol列
    int irow, icol;

    // 用来表示当x[n][c]进行im2col展开时，保存在row_x中的左上角坐标
    int irow_start, icol_start;

    // 当w在x上滑动到某一个位置时，用i与j来遍历x与w重叠区域的数据
    int i, j;

    // 用来表示x中的第n个图片的第c个通道
    int n, c, n_start, c_start;

    // 提前进行类型转换，因为PyArray_DATA()与PyArray_GETPTR1/2/3/4()
    // 两个宏所获取的数据指针都是void *类型
    npy_double *x_data = PyArray_DATA(x);
    npy_double *row_x_data = PyArray_DATA(row_x);

    // 对x中的第n个图片的第c个通道进行im2col展开
    for(n = 0; n < N; ++n)
    {
        for(c = 0; c < C; ++c)
        {
            // 将x[n][c]展开后的im2col矩阵在row_x中的位置的左上角坐标
            irow_start = n * CH * CW;
            icol_start = c * KH * KW;

            // 开始对x[n][c]进行im2col过程
            irow = irow_start;
            for(crow = 0; crow < CH; ++crow)
            {
                for(ccol = 0; ccol < CW; ++ccol)
                {
                    // 在im2col_HW函数中，此处为irow = 0;
                    icol = icol_start;
                    row_start = crow * stride;
                    col_start = ccol * stride;
                    for(i = row_start; i < row_start + KH; ++i)
                    {
                        // 使用memcpy。这种方法在应对卷积核比较大的时候效果要比其他的要更快
                        memcpy(row_x_data+irow*IW+icol, x_data+n*C*H*W+c*H*W+i*W+col_start, KW*sizeof(npy_double));
                        icol += KW;
                    }
                    ++irow;
                }
            }
        }
    }
    col_x = PyArray_Transpose(row_x, NULL);
    Py_DECREF(row_x);
    return col_x;
}

PyObject *wrap_im2col_HW(PyObject *self, PyObject *args)
{
    PyArrayObject *x, *col_x;
    int KH, KW, stride;
    if(!PyArg_ParseTuple(args, "Oiii", &x, &KH, &KW, &stride))
        return NULL;
    col_x = im2col_HW(x, KH, KW, stride);
    return col_x;
}

PyObject *wrap_im2col_NCHW(PyObject *self, PyObject *args)
{
    PyArrayObject *x, *col_x;
    int KH, KW, stride;
    if(!PyArg_ParseTuple(args, "Oiii", &x, &KH, &KW, &stride))
        return NULL;
    col_x = im2col_NCHW(x, KH, KW, stride);
    return col_x;
}

PyObject *wrap_im2col_NCHW_memcpy(PyObject *self, PyObject *args)
{
    PyArrayObject *x, *col_x;
    int KH, KW, stride;
    if(!PyArg_ParseTuple(args, "Oiii", &x, &KH, &KW, &stride))
        return NULL;
    col_x = im2col_NCHW_memcpy(x, KH, KW, stride);
    return col_x;
}

static PyMethodDef methods[] = {
    {"im2col_HW", wrap_im2col_HW, METH_VARARGS, "no doc"},
    {"im2col_NCHW", wrap_im2col_NCHW, METH_VARARGS, "no doc"},
    {"im2col_NCHW_memcpy", wrap_im2col_NCHW_memcpy, METH_VARARGS, "no doc"},
    {NULL, NULL, 0, NULL},
};


PyMODINIT_FUNC initim2colutils()
{
    Py_InitModule("im2colutils", methods);
    import_array();
}

