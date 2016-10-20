#include "D:/Anaconda2/include/Python.h"
#include "D:/Anaconda2/Lib/site-packages/numpy/core/include/numpy/arrayobject.h"
#include "../include/checkutils.h"


int check_ndim(PyArrayObject *x, int ndim, PyObject *error)
{
    if (PyArray_NDIM(x) != ndim)
    {
        char message[80];
        sprintf(message, "Invalid shape: the input "
                "ndarray must be %dD!", ndim);
        PyErr_SetString(error, message);
        return FAIL;
    }
    else
        return SUCCESS;
}

int check_dtype_is_double(PyArrayObject *x, PyObject *error)
{
    if (PyArray_TYPE(x) != NPY_DOUBLE)
    {
        PyErr_SetString(error, "Invalid dtype: the input "
                        "ndarray's dtype must be numpy.double!");
        return FAIL;
    }
    else
        return SUCCESS;
}


int check_can_be_convolved(int H, int W, int KH, int KW,
                              int stride, PyObject *error)
{
    if ((H - KH) % stride != 0 || (W - KW) % stride != 0)
    {
        char message[100];
        sprintf(message, "Invalid shape: %d*%d x can't be "
                "convolved completely given %d*%d kernel "
                "and %d stride!", H, W, KH, KW, stride);
        PyErr_SetString(error, message);
        return FAIL;
    }
    else
        return SUCCESS;
}


int check_columnize_x_shape_HW(int OH, int OW, int KH, int KW,
                               int CH, int CW, PyObject *error)
{
    if (OH != KH * KW || OW != CH * CW)
    {
        char message[100];
        sprintf(message, "Invalid shape: %d*%d columnize_x is not "
                "fit with %d*%d kernel and %d*%d conv_x", OH, OW,
                KH, KW, CH, CW);
        PyErr_SetString(error, message);
        return FAIL;
    }
    else
        return SUCCESS;
}



int check_columnize_x_shape_NCHW(int OH, int OW, int KH, int KW,
                                 int CH, int CW, PyObject *error)
{
    if (OH % (KH * KW) != 0 && OW % (CH * CW) != 0)
    {
        char message[100];
        sprintf(message, "Invalid shape: %d*%d columnize_x is not "
                "fit with %d*%d kernel and %d*%d conv_x", OH, OW,
                KH, KW, CH, CW);
        PyErr_SetString(error, message);
        return FAIL;
    }
    else
        return SUCCESS;
}
