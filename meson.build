project('simplextree', 'cpp')

py_mod = import('python')
py = py_mod.find_installation(pure: false)

incdir = include_directories('include')
pybind11dir = include_directories('extern/pybind11/include')

py.extension_module(
  '_simplextree',
  'src/simplextree/_simplextree.cpp',
  install: true,
  include_directories: [incdir, pybind11dir]
)
py.extension_module(
  '_unionfind',
  'src/simplextree/_unionfind.cpp',
  install: true,
  include_directories: [incdir, pybind11dir]
)