from distutils.core import setup

setup(
    name='hamaa',
    version='0.5',
    packages=['test', 'hamaa', 'hamaa.clib', 'hamaa.clib.test', 'hamaa.utils', 'hamaa.datasets',
              'hamaa.datasets.mnist'],
    url='https://github.com/monitor1379/hamaa',
    license='GPL',
    author='monitor1379',
    author_email='yy4f5da2@hotmail.com',
    description='a Simple and Naive Deep Learning library'
)
