#include <stdio.h>
#include <malloc.h>
#include "conv.h"


Mat *new_mat(int row, int col)
{
    Mat *mat = (Mat *)malloc(sizeof(Mat));
    mat->data = (dtype_t *)malloc(sizeof(dtype_t) * row * col);
    mat->row = row;
    mat->col = col;
    int i;
    for(i = 0; i < col * row; ++i) mat->data[i] = i;
    return mat;
}

Mat *new_mat_without_data(int row, int col)
{
    Mat *mat = (Mat *)malloc(sizeof(Mat));
    mat->data = NULL;
    mat->row = row;
    mat->col = col;
    return mat;
}


void free_mat(Mat *mat)
{
    if(mat)
    {
        if(mat->data)
            free(mat->data);
        free(mat);
    }
}


void free_mat_without_data(Mat *mat)
{
    if(mat)
    {
        free(mat);
    }
}

void display(Mat *mat)
{
    if(!mat)
        return ;
    printf("row:%d, col:%d\n", mat->row, mat->col);
    int i, j;
    for(i = 0; i < mat->row; ++i)
    {
        for(j = 0; j < mat->col; ++j)
        {
            printf("%d, ", mat->data[i*mat->col+j]);
        }
        printf("\n");
    }
}



Mat *__im2col(Mat *im, int kh, int kw, int stride, int ch, int cw)
{
    Mat *output = new_mat(kh * kw, ch * cw);
    int crow, ccol, i, j, i_start, j_start, output_idx = 0, t = 0;
    for(crow = 0; crow < ch; ++crow)
    {
        for(ccol= 0; ccol < cw; ++ccol)
        {
            i_start = crow * stride;
            j_start = ccol * stride;
            t = 0;
            for(i = i_start; i < i_start + kh; ++i)
            {
                for(j = j_start; j < j_start + kw; ++j)
                {
                    output->data[t*output->col+output_idx] = im->data[i*im->col+j];
                    ++t;
                }
            }
            ++output_idx;
        }
    }
    return output;
}


int __get_conv_output_size(int ih, int kh, int stride, int ph)
{
    return (ih + 2 * ph - kh) / stride + 1;
}
