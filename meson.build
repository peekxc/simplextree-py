project('simplextree', 'cpp', default_options : ['cpp_std=c++17'])

py_mod = import('python')
py = py_mod.find_installation(pure: true)
py_dep = py.dependency()
message(py.path())
message(run_command(py, ['-c', 'import numpy; print(numpy.__version__)']).stdout().strip())

if py.language_version().version_compare('< 3.8')
  error('Invalid Python version, only >= 3.8 is supported.')
endif

## Message the C++ compiler location for debugging
compiler = meson.get_compiler('cpp')
message('Compiler: ')
message(compiler.get_id())

## Header includes
inc_pybind11 = include_directories('extern' / 'pybind11' / 'include')
inc_local = include_directories('include')
incdir_numpy = run_command(py, 
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()
inc_np = include_directories(incdir_numpy)

## Platform detection
is_windows = host_machine.system() == 'windows'

subdir('simplextree')

## Install the package
# install_subdir('simplextree', install_dir: py.get_install_dir(pure: false))

# py.extension_module(
#   '_simplextree',
#   'simplextree' / '_simplextree.cpp',
#   install: true,
#   include_directories: [inc_local, incdir_numpy, inc_pybind11]
# )
# py.extension_module(
#   '_unionfind',
#   'simplextree' / '_unionfind.cpp',
#   install: true,
#   include_directories: [inc_local, incdir_numpy, inc_pybind11]
# )

# py.install_sources(
#   'src/simplextree/__init__.py',
#   subdir: 'src/simplextree',
# )
