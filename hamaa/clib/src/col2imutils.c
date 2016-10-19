#include "D:/Anaconda2/include/Python.h"
#include "D:/Anaconda2/Lib/site-packages/numpy/core/include/numpy/arrayobject.h"

PyArrayObject *col2im_HW(PyArrayObject *columnize_x, int KH, int KW, int CH, int CW, int stride)
{
    assert(PyArray_TYPE(columnize_x) == NPY_DOUBLE);
    assert(PyArray_NDIM(columnize_x) == 2);

    int OH = PyArray_DIM(columnize_x, 0);
    int OW = PyArray_DIM(columnize_x, 1);

    assert(OH == KH * KW && OW == CH * CW);

    int H = (CH - 1) * stride + KH;
    int W = (CW - 1) * stride + KW;

    int nd = 2;
    npy_intp dims[] = {H, W};
    int is_f_order = 0;
    PyArrayObject *x = PyArray_ZEROS(nd, dims, NPY_DOUBLE, is_f_order);

    int col, row, col_start, row_start;
    int orow, ocol;

    npy_double *columnize_x_data = PyArray_DATA(columnize_x);
    npy_double *x_data = PyArray_DATA(x);

    for(ocol = 0; ocol < OW; ++ocol)
    {
        // (1): ccol * stride = col
        // (2): ccol * CW + crow = ocol
        col_start = ocol % CW * stride;
        row_start = ocol / CW * stride;
        
        for(orow = 0; orow < OH; ++orow)
        {
            col = col_start + orow % KW;
            row = row_start + orow / KW;
            x_data[row * W + col] = columnize_x_data[orow * OW + ocol];

        }
    }
    return x;
}


PyArrayObject *col2im_NCHW(PyArrayObject *columnize_x, int KH, int KW, int CH, int CW, int stride)
{
    assert(PyArray_TYPE(columnize_x) == NPY_DOUBLE);
    assert(PyArray_NDIM(columnize_x) == 2);

    int OH = PyArray_DIM(columnize_x, 0);
    int OW = PyArray_DIM(columnize_x, 1);

    assert(OH % (KH * KW) == 0 && OW % (CH * CW) == 0);

    int N = OW / (CH * CW);
    int C = OH / (KH * KW);
    int H = (CH - 1) * stride + KH;
    int W = (CW - 1) * stride + KW;

    int nd = 4;
    npy_intp dims[] = {N, C, H, W};
    int is_f_order = 0;
    PyArrayObject *x = PyArray_ZEROS(nd, dims, NPY_DOUBLE, is_f_order);

    int i, j;
    int col, row;
    int col_start, row_start;
    int orow, ocol;
    int orow_start, ocol_start;

    npy_double *columnize_x_data = PyArray_DATA(columnize_x);
    npy_double *x_data = PyArray_DATA(x);

    for(i = 0; i < N; ++i)
    {
        for(j = 0; j < C; ++j)
        {
            orow_start = j * KH * KW;
            ocol_start = i * CH * CW;

            for(ocol = ocol_start; ocol < ocol_start + CH * CW; ++ocol)
            {
                col_start = (ocol - ocol_start) % CW * stride;
                row_start = (ocol - ocol_start) / CW * stride;

                for(orow = orow_start; orow < orow_start + KH * KW; ++orow)
                {
                    col = (orow - orow_start) % KW + col_start;
                    row = (orow - orow_start) / KW + row_start;
                    x_data[i*C*H*W + j*H*W + row*W + col] = columnize_x_data[orow * OW + ocol];
                }
            }
        }
    }
    return x;
}


PyObject *wrap_col2im_HW(PyObject *self, PyObject *args)
{
    PyArrayObject *columnize_x, *x;
    int KH, KW, CH, CW, stride;
    if(!PyArg_ParseTuple(args, "Oiiiii", &columnize_x, &KH, &KW, &CH, &CW, &stride))
        return NULL;
    x = col2im_HW(columnize_x, KH, KW, CH, CW, stride);
    return x;
}


PyObject *wrap_col2im_NCHW(PyObject *self, PyObject *args)
{
    PyArrayObject *columnize_x, *x;
    int KH, KW, CH, CW, stride;
    if(!PyArg_ParseTuple(args, "Oiiiii", &columnize_x, &KH, &KW, &CH, &CW, &stride))
        return NULL;
    x = col2im_NCHW(columnize_x, KH, KW, CH, CW, stride);
    return x;
}


static PyMethodDef methods[] =
{
    {"col2im_HW", wrap_col2im_HW, METH_VARARGS, "no doc"},
    {"col2im_NCHW", wrap_col2im_NCHW, METH_VARARGS, "no doc"},
    {NULL, NULL, 0, NULL},
};


PyMODINIT_FUNC initcol2imutils()
{
    Py_InitModule("col2imutils", methods);
    import_array();
}


