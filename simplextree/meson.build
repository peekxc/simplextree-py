## Message the C++ compiler location for debugging
compiler = meson.get_compiler('cpp')
message('Compiler = '+compiler.get_id())

## Compiler flags
_cpp_args = compiler.get_supported_arguments(
  '-Wno-unused-parameter', 
  '-Wno-reserved-identifier', 
  '/Wc++11-extensions',
  '-Wno-c++98-compat-pedantic', 
  '-Wno-old-style-cast', 
  '-Wno-c++98-compat'
)
add_project_arguments(_cpp_args, language: ['cpp'])
message('Compiler args:')
message(_cpp_args)

## Platform detection
is_windows = host_machine.system() == 'windows'

## Windows adjustments
is_mingw = is_windows and compiler.get_id() == 'gcc'
if is_windows and is_mingw
  add_project_link_arguments(['-lucrt', '-static'], language: ['cpp']) # For mingw-w64, link statically against the UCRT.
  add_project_arguments('-D__USE_MINGW_ANSI_STDIO=1', language: ['cpp']) # https://github.com/rgommers/scipy/issues/118
  add_project_arguments('-DMS_WIN64', language: ['cpp'])
endif   

## Set C++ version 
override_args = ['cpp_std=c++17']
if is_windows
  override_args = ['cpp_std=c++14']
endif 
# if is_windows
#   cpp_std_args = ['-std=c++17']
# else
#   cpp_std_args = ['/std:c++11']
# endif 

## Extension modules
py.extension_module(
  '_simplextree',
  '_simplextree.cpp',
  install: true,
  include_directories: [inc_local, incdir_numpy, inc_pybind11], 
  cpp_args: _cpp_args,
  override_options: override_args
)
py.extension_module(
  '_unionfind',
  '_unionfind.cpp',
  install: true,
  include_directories: [inc_local, incdir_numpy, inc_pybind11], 
  cpp_args: _cpp_args,
  override_options: override_args
)
## Shouldn't add C++20 to the list, make a new package first
# py.extension_module(
#   '_combinatorial',
#   '_combinatorial.cpp',
#   install: true,
#   include_directories: [inc_local, incdir_numpy, inc_pybind11], 
#   cpp_args: _cpp_args,
#   override_options: override_args
# )

python_sources = [
  '__init__.py',
  'SimplexTree.py', 
  'UnionFind.py', 
  'combinatorial.py', 
  'predicates.py'
]

## Print install directory
message('Source install dir = '+py.get_install_dir())
py.install_sources(
  python_sources,
  subdir: 'simplextree'
)