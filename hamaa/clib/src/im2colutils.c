#include "D:/Anaconda2/include/Python.h"
#include "D:/Anaconda2/Lib/site-packages/numpy/core/include/numpy/arrayobject.h"


PyArrayObject *im2col_HW(PyArrayObject *x, int KH, int KW, int stride)
{
    assert(PyArray_TYPE(x) == NPY_DOUBLE);
    assert(PyArray_NDIM(x) == 2);

    int H = PyArray_DIM(x, 0);
    int W = PyArray_DIM(x, 1);

    assert((H - KH) % stride == 0);
    assert((W - KW) % stride == 0);

    int CH = (H - KH) / stride + 1;
    int CW = (W - KW) / stride + 1;

    int OH = KH * KW;
    int OW = CH * CW;

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
    assert(PyArray_TYPE(x) == NPY_DOUBLE);
    assert(PyArray_NDIM(x) == 4);

    int N = PyArray_DIM(x, 0);
    int C = PyArray_DIM(x, 1);
    int H = PyArray_DIM(x, 2);
    int W = PyArray_DIM(x, 3);

    assert ((H - KH) % stride == 0);
    assert ((W - KW) % stride == 0);

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
    Py_InitModule("im2colutils", methods);
    import_array();
}

