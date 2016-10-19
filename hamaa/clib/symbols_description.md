# 符号说明

## 格式

数据格式一共有两种: 

- `HW` : 表示一张单通道且高为`H`、宽为`W`个像素的图片 
- `NCHW` : 表示`N`张具有`C`个通道且高为`H`、宽为`W`个像素的图片集

## 数据

- `x/image/images` : 输入数据集，一般看成图片数据，支持`HW`格式与`NCHW`格式

- `w/kernel/kernels` : 卷积核，支持`HW`格式与`NCHW`格式

- `conv_x` : 使用`w`对`x`进行卷积后的结果

- `columnize_x` : 使用`im2col`方法对`x`进行二维展开后的结果，\
其中根据`x`的格式来选择调用`im2col_HW/im2col_NCHW`方法

- `rowing_x` : 在某些框架中卷积前的`im2col`操作并不是将`x`列化，\
而是行化得到`rowing_x`，即`columnize_x`的转置矩阵

- `patch` : 正在和`w`进行卷积的`x`上的一块和卷积核矩阵一样大小的二维区域


> **注意**: `x`与`w`必须同时为`HW`格式或者同时为`NCHW`格式。

## 形状

假设将`x`看成多通道图片数据集，则有 :

- `N` : `x`中图片的个数
- `C` : `x`中每张图片的通道数
- `H` : `x`中每张图片的高/行
- `W` : `x`中每张图片的宽/列


同样道理，对于卷积核`w/kernels`有 :

- `KN` : `kernel`的个数
- `KN` : 每个`kernel`的通道数
- `KH` : 每个`kernel`的高/行
- `KW` : 每个`kernel`的宽/列


对于`im2col`的输出（`output`）`columnize_x`有 :

- `OH` : `columnize_x`的高/行
- `OW` : `columnize_x`的宽/列

对于`im2row`的输出（`output`）`rowing_x`有 :

- `RH` : `rowing_x`的高/行
- `RW` : `rowing_x`的宽/列


## 下标

对于`x` : 
- `i` : 用于表示`x`中的第`i`张图片
- `j` : 用于表示`x`中某张图片的第`j`个通道
- `row` : 用于表示`x`中某个图片某个通道的某一行
- `col` : 用于表示`x`中某个图片某个通道的某一列


对于`w` : 
- `ki` : 用于表示`w`中的第`ki`个卷积核
- `kj` : 用于表示`w`中某一个卷积核的第`kj`个通道
- `krow` : 用于表示`w`中某个卷积核某个通道的某一行
- `kcol` : 用于表示`w`中某个卷积核某个通道的某一列


对于卷积输出结果`conv_x` : 
- `ci` : 用于表示`conv_x`中的第`i`个图片
- `cj` : 用于表示`conv_x`中某张图片的第`j`个通道
- `crow` : 用于表示`conv_x`中某个图片某个通道的某一行
- `ccol` : 用于表示`conv_x`中某个图片某个通道的某一列

对于`columnize_x` : 
- `orow` : `columnize_x`中的某一行 
- `ocol` : `columnize_x`中的某一列


对于`rowing_x` : 
- `rrow` : `rowing_x`中的某一行 
- `rcol` : `rowing_x`中的某一列



