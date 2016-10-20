from distutils.core import Extension, setup

mod_im2colutils = Extension(name="im2colutils", sources=["src/im2colutils.c", "src/checkutils.c"])
mod_col2imutils = Extension(name="col2imutils", sources=["src/col2imutils.c", "src/checkutils.c"])
mod_im2rowutils = Extension(name="im2rowutils", sources=["src/im2rowutils.c", "src/checkutils.c"])

setup(ext_modules=[mod_im2colutils, mod_col2imutils, mod_im2rowutils])
