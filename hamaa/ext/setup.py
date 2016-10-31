from distutils.core import Extension, setup
import numpy as np
import os

dir_path = os.path.dirname(__file__) + os.sep


mod_im2colutils = Extension(name="im2colutils",
                            sources=[dir_path + "src/im2colutils.c", dir_path + "src/checkutils.c"],
                            include_dirs=[np.get_include()])

mod_col2imutils = Extension(name="col2imutils",
                            sources=[dir_path + "src/col2imutils.c", dir_path + "src/checkutils.c"],
                            include_dirs=[np.get_include()])


mod_im2rowutils = Extension(name="im2rowutils",
                            sources=[dir_path + "src/im2rowutils.c", dir_path + "src/checkutils.c"],
                            include_dirs=[np.get_include()])

setup(ext_modules=[mod_im2colutils, mod_col2imutils, mod_im2rowutils],
      language='C')
