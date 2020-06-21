# -*- coding: utf-8 -*-

from setuptools import setup

exec(open('./SimLight/_version.py').read())

setup(
    name='SimLight',
    description='A tool helps your optical simulation.',
    license='MIT License',
    version=__version__,
    author='Miyoshichi',
    author_email='zhou.x.ae@m.titech.ac.jp',
    packages=['SimLight'],
    install_requires=['numpy', 'matplotlib'],
    classifiers=[
        'Development Stauts :: 3 - Alpha',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Utilities'
    ],
    url='https://github.com/Miyoshichi/SimLight'
)
