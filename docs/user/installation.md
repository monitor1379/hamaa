# 安装 Installation

安装Hamaa所需的机器配置如下:

- **Platform:** Windows、Linux、macOS

- **Python Version:** Python 2.x (后续将支持Python 3.x版本)

## Step 1: Download Hamaa

如果你的电脑装有Git，可以通过Git下载源代码到本地:
```bash
$ git clone https://github.com/monitor1379/hamaa.git
```
或者通过浏览器在[官网首页](https://github.com/monitor1379/hamaa)点击右方的`Clone or download`按钮进行下载与解压。

---

## Step 2: Install dependencies

Hamaa主要依赖于4个库，它们的作用分别是:

- NumPy: 提供简便的矩阵运算、数值计算功能

- Matplotlib: 提供强大的绘图功能

- Nose: 负责单元测试与功能测试

- pillow: 供图像读写与图像处理功能

如果你的机器上装有`pip`（[Pyhton包管理器](https://pypi.python.org/pypi/pip)），则可通过以下方式进行安装:
```bash
$ pip install numpy matplotlib nose Pillow
```

或者进入Hamaa解压后的文件夹中输入:
```bash
$ pip install -r requirement.txt
```

---

## Step 3: Install Hamaa

继续在路径为Hamaa解压后的文件夹中打开shell输入:

编译 Python C Extension:
```bash
$ python setup.py build_ext
```


最后安装Hamaa: 
```bash
$ pip install .
```




