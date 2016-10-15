from distutils.core import Extension, setup

mod = Extension(name="im2colutils",
                sources=["src/im2colutils.c"])
setup(ext_modules=[mod])
