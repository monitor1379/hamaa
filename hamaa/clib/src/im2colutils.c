#include "D:/Anaconda2/include/Python.h"
#include "D:/Anaconda2/Lib/site-packages/numpy/core/include/numpy/arrayobject.h"
#include "../include/checkutils.h"


// 模块中的Exception对象，用来抛出异常
static PyObject *error;


PyArrayObject *im2col_HW(PyArrayObject *x, int KH, int KW, int stride)
{
    if(!check_ndim(x, 2, error))
        return NULL;
    if(!check_dtype_is_double(x, error))
        return NULL;

    int H = PyArray_DIM(x, 0);
    int W = PyArray_DIM(x, 1);

    if(!check_can_be_convolved(H, W, KH, KW, stride, error))
        return NULL;

    // 计算卷积结果conv_x的形状
    int CH = (H - KH) / stride + 1;
    int CW = (W - KW) / stride + 1;

    // 计算im2col结果columnize_x的形状
    int OH = KH * KW;
    int OW = CH * CW;

    // 创建columnize_x对象
    int nd = 2;
    npy_intp dims[] = {OH, OW};
    PyArrayObject *columnize_x = PyArray_SimpleNew(nd, dims, NPY_DOUBLE);

    int row, col;
    int row_start, col_start;
    int crow, ccol;
    int orow, ocol;

    npy_double *x_data = PyArray_DATA(x);
    npy_double *columnize_x_data = PyArray_DATA(columnize_x);


    ocol = 0;
    for(crow = 0; crow < CH; ++crow)
    {
        for(ccol = 0; ccol < CW; ++ccol)
        {
            row_start = crow * stride;
            col_start = ccol * stride;
            orow = 0;
            for(row = row_start; row < row_start + KH; ++row)
            {
                for(col = col_start; col < col_start + KW; ++col)
                {
                    columnize_x_data[orow * OW + ocol] = x_data[row * W + col];
                    ++orow;
                }
            }
            ++ocol;
        }
    }
    return columnize_x;
}


PyArrayObject *im2col_NCHW(PyArrayObject *x, int KH, int KW, int stride)
{
    if(!check_ndim(x, 4, error))
        return NULL;
    if(!check_dtype_is_double(x, error))
        return NULL;

    int N = PyArray_DIM(x, 0);
    int C = PyArray_DIM(x, 1);
    int H = PyArray_DIM(x, 2);
    int W = PyArray_DIM(x, 3);

    if(!check_can_be_convolved(H, W, KH, KW, stride, error))
        return NULL;

    int CH = (H - KH) / stride + 1;
    int CW = (W - KW) / stride + 1;

    int OH = C * KH * KW;
    int OW = N * CH * CW;

    int nd = 2;
    npy_intp dims[] = {OH, OW};
    PyArrayObject *columnize_x = PyArray_SimpleNew(nd, dims, NPY_DOUBLE);

    int i, j;
    int row, col;
    int row_start, col_start;
    int crow, ccol;
    int orow, ocol;
    int orow_start, ocol_start;

    npy_double *x_data = PyArray_DATA(x);
    npy_double *columnize_x_data = PyArray_DATA(columnize_x);

    for(i = 0; i < N; ++i)
    {
        for(j = 0; j < C; ++j)
        {
            orow_start = j * KH * KW;
            ocol_start = i * CH * CW;

            ocol = ocol_start;
            for(crow = 0; crow < CH; ++crow)
            {
                for(ccol = 0; ccol < CW; ++ccol)
                {
                    orow = orow_start;
                    row_start = crow * stride;
                    col_start = ccol * stride;
                    for(row = row_start; row < row_start + KH; ++row)
                    {
                        for(col = col_start; col < col_start + KW; ++col)
                        {
                            columnize_x_data[orow * OW + ocol] = x_data[i*C*H*W + j*H*W + row*W + col];
                            ++orow;
                        }
                    }
                    ++ocol;
                }
            }
        }
    }
    return columnize_x;
}


PyObject *wrap_im2col_HW(PyObject *self, PyObject *args)
{
    PyArrayObject *x, *columnize_x;
    int KH, KW, stride;
    if(!PyArg_ParseTuple(args, "Oiii", &x, &KH, &KW, &stride))
        return NULL;
    columnize_x = im2col_HW(x, KH, KW, stride);
    return columnize_x;
}


PyObject *wrap_im2col_NCHW(PyObject *self, PyObject *args)
{
    PyArrayObject *x, *columnize_x;
    int KH, KW, stride;
    if(!PyArg_ParseTuple(args, "Oiii", &x, &KH, &KW, &stride))
        return NULL;
    columnize_x = im2col_NCHW(x, KH, KW, stride);
    return columnize_x;
}


static PyMethodDef methods[] =
{
    {"im2col_HW", wrap_im2col_HW, METH_VARARGS, "no doc"},
    {"im2col_NCHW", wrap_im2col_NCHW, METH_VARARGS, "no doc"},
    {NULL, NULL, 0, NULL},
};


PyMODINIT_FUNC initim2colutils()
{
    import_array();
    PyObject *m = Py_InitModule("im2colutils", methods);
    if(m == NULL)
        return ;
    error = PyErr_NewException("im2colutils.error", NULL, NULL);
    Py_INCREF(error);
    PyModule_AddObject(m, "error", error);
}

