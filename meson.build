project('simplextree', 'cpp', default_options : ['cpp_std=c++17'])

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()
message(py.path())
message(run_command(py, ['-c', 'import numpy; print(numpy.__version__)']).stdout().strip())

if py.language_version().version_compare('< 3.8')
  error('Invalid Python version, only >= 3.8 is supported.')
endif

## Message the C++ compiler location for debugging
compiler = meson.get_compiler('cpp')
message(compiler.get_id())

## Numpy Include 
# incdir_numpy = run_command(py3,
#   ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
#   check : true
# ).stdout().strip()
# incdir = include_directories(join_paths(meson.source_root(), 'include'))
# pybind11dir = include_directories(join_paths(meson.source_root(), 'extern', 'pybind11', 'include'))

incdir = include_directories('include')
pybind11dir = include_directories('extern' / 'pybind11' / 'include')

## Install the package
install_subdir('src' / 'simplextree', install_dir: py.get_install_dir(pure: false))

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

# py.install_sources(
#   'src/simplextree/__init__.py',
#   subdir: 'src/simplextree',
# )
