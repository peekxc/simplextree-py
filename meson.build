project(
  'simplextree', 'cpp', 
  license: 'Apache-2.0',
  meson_version: '>= 0.64.0',
  default_options : [
    'buildtype=release', 
    'b_ndebug=if-release',
    'cpp_std=c++17', 
    'warning_level=2', 
    'pkgconfig.relocatable=true'
  ]
)

py_mod = import('python')
py = py_mod.find_installation(pure: true)
py_dep = py.dependency()
message('Python path =' + py.full_path())
message('Numpy version =' + run_command(py, ['-c', 'import numpy; print(numpy.__version__)'], check: true).stdout().strip())

## Check the python version
if py.language_version().version_compare('< 3.8')
  error('Invalid Python version, only >= 3.8 is supported.')
endif

## Header includes
inc_pybind11 = include_directories('extern' / 'pybind11' / 'include')
inc_local = include_directories('include')
incdir_numpy = run_command(py, ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'], check : true).stdout().strip()
inc_np = include_directories(incdir_numpy)

## Begin compiler arguments
compiler = meson.get_compiler('cpp')
message('Compiler = '+compiler.get_id())
_cpp_args = []
if get_option('buildtype') == 'debugoptimized'
  _cpp_args += compiler.get_supported_arguments(
    '-O2',
    # '-fsanitize=address', 
    # '-fno-omit-frame-pointer',
    '-g',
    '-Wall'
  )
else
  ## Release
  _cpp_args += compiler.get_supported_arguments(
    '-flto=thin',   # think LTO
    # '-flto',      # monolithic LTO
    '-O3',          # full optimizations
    '-DNDEBUG',     # remove assertions
    '-Wl,-s',       # strip symbols to reduce binary size
    # '-march=native' # either this or lto seems to not work on multiple builds
  )
endif




## Compile the package directory
subdir('simplextree')

## Install the package
# install_subdir('simplextree', install_dir: py.get_install_dir(pure: false))

# py.install_sources(
#   'src/simplextree/__init__.py',
#   subdir: 'src/simplextree',
# )
