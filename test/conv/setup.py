from distutils.core import setup, Extension

module1 = Extension("conv", sources = ["wrap_conv.c", "conv.c"])
setup(ext_modules = [module1])
