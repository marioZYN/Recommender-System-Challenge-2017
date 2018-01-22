from distutils.core import setup
from Cython.Build import cythonize
import numpy

#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"

setup(
  name = 'CosineSim',
  ext_modules = cythonize("CosineSim.pyx"),
  include_dirs=[numpy.get_include()]
)

# to compile --python setup.py build_ext --inplace
