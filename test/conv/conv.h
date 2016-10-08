#ifndef CONV_H_INCLUDED
#define CONV_H_INCLUDED

typedef double dtype_t;

typedef struct Mat{
    dtype_t *data;
    int row;
    int col;
} Mat;


Mat *new_mat(int row, int col);
Mat *new_mat_without_data(int row, int col);

void free_mat(Mat *mat);
void free_mat_without_data(Mat *mat);

void display(Mat *mat);

Mat *__im2col(Mat *im, int kh, int kw, int stride, int ch, int cw);


int __get_conv_output_size(int ih, int kh, int stride, int ph);

#endif // CONV_H_INCLUDED
