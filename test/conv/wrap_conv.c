#include "D:/Anaconda2/include/Python.h"
#include "D:/Anaconda2/Lib/site-packages/numpy/core/include/numpy/arrayobject.h"
#include <malloc.h>
#include "conv.h"

PyObject *wrap_im2col(PyObject *self, PyObject *args)
{
    import_array();
    PyArrayObject *in_arr;
    int kh, kw, stride, ch, cw;

    if(!PyArg_ParseTuple(args, "Oiiiii", &in_arr, &kh, &kw, &stride, &ch, &cw))
        return NULL;

    if(in_arr->descr->type_num != NPY_DOUBLE)
    {
        printf("ERROR:Input array's dtype must be DOUBLE!!!\n");
        return NULL;
    }

    Mat *in_mat = new_mat_without_data((int)in_arr->dimensions[0], (int)in_arr->dimensions[1]);
    in_mat->data = (dtype_t *)in_arr->data;

    Mat *out_mat = __im2col(in_mat, kh, kw, stride, ch, cw);

    int nd = 2;
    npy_intp dims[] = {out_mat->row, out_mat->col};
    PyArrayObject *out_arr = PyArray_SimpleNew(nd, dims, NPY_DOUBLE);

    memcpy(out_arr->data, out_mat->data,  out_mat->row * out_mat->col * sizeof(dtype_t));

    free_mat_without_data(in_mat);
    free_mat(out_mat);
    return out_arr;
}

static PyMethodDef methods[] = {
    {"im2col", wrap_im2col, METH_VARARGS, "No doc"},
    {NULL, NULL, 0, NULL}
};

void initconv()
{
    Py_InitModule("conv", methods);
}
