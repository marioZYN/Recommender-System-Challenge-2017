#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16/07/2017

@author: Maurizio Ferrari Dacrema
"""

from distutils.core import setup,  Extension
from Cython.Build import cythonize
import numpy
# setup(
#     ext_modules=[
#         Extension("my_module", ["my_module.c"],
#                   include_dirs=[numpy.get_include()]),
#     ],
# )

setup(
    name = "My hello app",
    ext_modules = cythonize('SLIM_BPR_Cython_Epoch.pyx'), 
    include_dirs=[numpy.get_include()]
     # accepts a glob pattern
    )
