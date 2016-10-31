#include "Python.h"
#include "numpy/arrayobject.h"
#include "../include/checkutils.h"

static PyObject *error;

PyArrayObject *im2row_HW(PyArrayObject *x, int KH, int KW, int stride)
{
    if(!check_dtype_is_double(x, error))
        return NULL;
    if(!check_ndim(x, 2, error))
        return NULL;

    int H = PyArray_DIM(x, 0);
    int W = PyArray_DIM(x, 1);

    if(!check_can_be_convolved(H, W, KH, KW, stride, error))
        return NULL;

    int CH = (H - KH) / stride + 1;
    int CW = (W - KW) / stride + 1;

    int RH = CH * CW;
    int RW = KH * KW;

    int nd = 2;
    npy_intp dims[] = {RH, RW};
    PyArrayObject *rowing_x = PyArray_SimpleNew(nd, dims, NPY_DOUBLE);

    int i, j;
    int row, col;
    int row_start, col_start;
    int crow, ccol;
    int rrow, rcol;
    int rrow_start, rcol_start;

    npy_double *x_data = PyArray_DATA(x);
    npy_double *rowing_x_data = PyArray_DATA(rowing_x);


    rrow = 0;
    for(crow = 0; crow < CH; ++crow)
    {
        for(ccol = 0; ccol < CW; ++ccol)
        {
            row_start = crow * stride;
            col_start = ccol * stride;

            col = col_start;
            rcol = 0;
            for(row = row_start; row < row_start + KH; ++row)
            {
                // x[row][col:col+KH] <= rowing_x[rrow][rcol:rcol+KH]
                memcpy(rowing_x_data + rrow* RW + rcol,
                       x_data + row*W + col,
                       KH * sizeof(npy_double)
                       );
                rcol += KW;
            }
            ++rrow;
        }
    }

    return rowing_x;
}




PyArrayObject *im2row_NCHW(PyArrayObject *x, int KH, int KW, int stride)
{
    if(!check_dtype_is_double(x, error))
        return NULL;
    if(!check_ndim(x, 4, error))
        return NULL;

    int N = PyArray_DIM(x, 0);
    int C = PyArray_DIM(x, 1);
    int H = PyArray_DIM(x, 2);
    int W = PyArray_DIM(x, 3);

    if(!check_can_be_convolved(H, W, KH, KW, stride, error))
        return NULL;

    int CH = (H - KH) / stride + 1;
    int CW = (W - KW) / stride + 1;

    int RH = N * CH * CW;
    int RW = C * KH * KW;

    int nd = 2;
    npy_intp dims[] = {RH, RW};
    PyArrayObject *rowing_x = PyArray_SimpleNew(nd, dims, NPY_DOUBLE);

    int i, j;
    int row, col;
    int row_start, col_start;
    int crow, ccol;
    int rrow, rcol;
    int rrow_start, rcol_start;

    npy_double *x_data = PyArray_DATA(x);
    npy_double *rowing_x_data = PyArray_DATA(rowing_x);

    for(i = 0; i < N; ++i)
    {

        for(j = 0; j < C; ++j)
        {
            rrow_start = i * CH * CW;
            rcol_start = j * KH * KW;

            rrow = rrow_start;

            for(crow = 0; crow < CH; ++crow)
            {
                for(ccol = 0; ccol < CW; ++ccol)
                {
                    row_start = crow * stride;
                    col_start = ccol * stride;

                    col = col_start;
                    rcol = rcol_start;

                    for(row = row_start; row < row_start + KH; ++row)
                    {
                        memcpy(rowing_x_data + rrow*RW + rcol,
                               x_data + i*C*H*W + j*H*W + row*W + col,
                               KW * sizeof(npy_double));
                        rcol += KW;
                    }
                    ++rrow;
                }
            }

        }
    }
    return rowing_x;
}


PyObject *wrap_im2row_HW(PyObject *self, PyObject *args)
{
    PyArrayObject *x, *rowing_x;
    int KH, KW, stride;
    if(!PyArg_ParseTuple(args, "Oiii", &x, &KH, &KW, &stride))
        return NULL;
    rowing_x = im2row_HW(x, KH, KW, stride);
    return rowing_x;
}


PyObject *wrap_im2row_NCHW(PyObject *self, PyObject *args)
{
    PyArrayObject *x, *rowing_x;
    int KH, KW, stride;
    if(!PyArg_ParseTuple(args, "Oiii", &x, &KH, &KW, &stride))
        return NULL;
    rowing_x = im2row_NCHW(x, KH, KW, stride);
    return rowing_x;
}


static PyMethodDef methods[] =
{
    {"im2row_HW", wrap_im2row_HW, METH_VARARGS, "no doc"},
    {"im2row_NCHW", wrap_im2row_NCHW, METH_VARARGS, "no doc"},
    {NULL, NULL, 0, NULL},
};


PyMODINIT_FUNC initim2rowutils()
{
    import_array();
    PyObject *m = Py_InitModule("im2rowutils", methods);
    if(m == NULL)
        return ;
    error = PyErr_NewException("im2rowutils.error", NULL, NULL);
    Py_INCREF(error);
    PyModule_AddObject(m, "error", error);
}

