## Message the C++ compiler location for debugging
compiler = meson.get_compiler('cpp')
message('Compiler = '+compiler.get_id())

## Platform detection
cpp_std = 'c++17'
is_windows = host_machine.system() == 'windows'

## Windows adjustments
is_mingw = is_windows and compiler.get_id() == 'gcc'
if is_windows 
  cpp_std = 'c++11'
  if is_mingw
    add_project_link_arguments(['-lucrt', '-static'], language: ['cpp']) # For mingw-w64, link statically against the UCRT.
    add_project_arguments('-D__USE_MINGW_ANSI_STDIO=1', language: ['cpp']) # https://github.com/rgommers/scipy/issues/118
  endif 
endif   

## Extension modules
py.extension_module(
  '_simplextree',
  '_simplextree.cpp',
  install: true,
  include_directories: [inc_local, incdir_numpy, inc_pybind11], 
  cpp_args: ['-std=' + cpp_std]
)
py.extension_module(
  '_unionfind',
  '_unionfind.cpp',
  install: true,
  include_directories: [inc_local, incdir_numpy, inc_pybind11], 
  cpp_args: ['-std=' + cpp_std]
)

python_sources = [
  '__init__.py',
  'SimplexTree.py', 
  'UnionFind.py'
]

## Print install directory
message('Source install dir = '+py.get_install_dir())
py.install_sources(
  python_sources,
  subdir: 'simplextree'
)