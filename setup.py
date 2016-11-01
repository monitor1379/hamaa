# -*- coding:utf-8 -*-
"""Hamaa: a simple and naive deep learning library for python.

Hamaa is built on the NumPy and use Python C API to accelerate,
which is not tough and simple enough for DL learner.
Hamaa are made by the purpose of making learning deep learning
more convenient, so may you have a good time. :D
All Hamaa wheels distributed from GitHub are GPLv3 licensed.
"""


from setuptools import setup, find_packages, Extension
import numpy as np
import os
import sys


DOCLINES = (__doc__ or '').split("\n")


# Python C Extension
mod_im2colutils = Extension(name="im2colutils",
                            sources=["src/im2colutils.c",
                                     "src/checkutils.c"],
                            include_dirs=[np.get_include()])

mod_col2imutils = Extension(name="col2imutils",
                            sources=["src/col2imutils.c",
                                     "src/checkutils.c"],
                            include_dirs=[np.get_include()])

mod_im2rowutils = Extension(name="im2rowutils",
                            sources=["src/im2rowutils.c",
                                     "src/checkutils.c"],
                            include_dirs=[np.get_include()])

# setup metadata
metadata = dict(
    name='hamaa',
    version='0.6.0',
    author='Zhenpeng Deng',
    author_email='yy4f5da2@hotmail.com',
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    url='https://github.com/monitor1379/hamaa',
    download_url='https://github.com/monitor1379/hamaa',
    license='GPLv3',
    platforms=['Windows'],
    packages=find_packages(),
    include_package_data=True,  # 使用MANIFEST.in文件自动导入非Py模块的文件/文件夹
)

if sys.argv[0] == 'setup.py':
    # >> python setup.py install
    if sys.argv[1] == 'install':
        raise Exception('please use: \n'
                        '>> python setup.py build_ext\n'
                        'and then \n'
                        '>> pip install .')

    # >> python setup.py build_ext [-i]
    elif sys.argv[1] == 'build_ext':
        metadata['ext_modules'] = [mod_im2colutils, mod_col2imutils, mod_im2rowutils]
        os.chdir('hamaa/ext')
        if '-i' not in sys.argv:
            sys.argv.append('-i')
        setup(**metadata)

# pip install .
else:
    setup(**metadata)

