#include <stdio.h>
#include <stdlib.h>
#include "conv.h"
#include <time.h>
#include <string.h>

void test1()
{
    int ih = 100, iw = 100, kh = 2, kw = 2, stride = 1, ph = 0, pw = 0, ch, cw;
    Mat *im = new_mat(ih, iw);
//    display(im);
    ch = __get_conv_output_size(ih, kh, stride, ph);
    cw = __get_conv_output_size(iw, kw, stride, pw);

    int i = 0;
    Mat *fuck;
    for(i = 0; i < 100; ++i)
    {
        clock_t t0 = clock();
        fuck = __im2col(im, kh, kw, stride, ch, cw);
        clock_t t1 = clock();
        free(fuck);
        printf("time: %lfs\n", (t1 - t0) * 1.0 / CLOCKS_PER_SEC);
    }

//    display(fuck);


    free_mat(im);
//    free_mat(fuck);

}

void test2()
{
    int a[] = {1, 2, 3};
    int *b = (int *)malloc(sizeof(a));
    memcpy(b, a, sizeof(a));
    int i;
    for(i = 0; i < 3; ++i)
        printf("%d, ", b[i]);
    free(b);
}

int main()
{
//    test1()
    test2();
    return 0;
}
