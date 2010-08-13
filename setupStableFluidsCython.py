#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

sourcefiles = ['StableFluidsCython.pyx','solver.c']
ext_modules = [Extension('StableFluidsCython',sourcefiles,
        include_dirs = [numpy.get_include(),'.'])]

setup(
  name = "StableFluidsCython",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules
)