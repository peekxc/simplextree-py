# -*- coding: utf-8 -*-
import os 
import sys
import sysconfig
from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

base_path = os.path.dirname(__file__)
# compile_flags = list(dict.fromkeys(sysconfig.get_config_var('CFLAGS').split()))
# compile_flags += list(dict.fromkeys(sysconfig.get_config_var('CPPFLAGS').split()))
if sys.platform.startswith("win"):
  cpp_version = "c++11"
  compile_flags = ["/O2", "/std:c++11"]
else: 
  cpp_version = "c++17"
  compile_flags = ["-std=c++17", "-Wall", "-Wextra", "-O2", "-Wno-unused-parameter"]
print(f"COMPILER FLAGS: { str(compile_flags) }")

ext_modules = [
  Pybind11Extension(
    'simplextree._simplextree', 
    sources = ['simplextree/_simplextree.cpp'], 
    include_dirs=[
      'extern/pybind11/include',
      'include'
    ], 
    extra_compile_args=compile_flags,
    language=cpp_version
  ), 
   Pybind11Extension(
    'simplextree._union_find', 
    sources = ['simplextree/UnionFind.cpp'], 
    include_dirs=[
      'extern/pybind11/include', 
      'include'
    ], 
    extra_compile_args=compile_flags,
    language=cpp_version
  )
]

from setuptools.command.build_ext import build_ext

setup(
  name="simplextree",
  author="Matt Piekenbrock",
  # version=__version__,
  author_email="matt.piekenbrock@gmail.com",
  description="Package for manipulating simplicial complexes",
  long_description="",
  ext_modules=ext_modules,
  cmdclass={'build_ext': build_ext},
  zip_safe=False, # needed for platform-specific wheel 
  python_requires=">=3.8",
  package_dir={'': 'src'}, # < root >/src/* contains packages
  packages=['simplextree']
)

