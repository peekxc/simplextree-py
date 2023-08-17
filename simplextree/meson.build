
## Windows adjustments
is_mingw = is_windows and cc.get_id() == 'gcc'
if is_windows
  gcc_link_args = ['-lucrt', '-static'] # For mingw-w64, link statically against the UCRT.
  if is_mingw
    add_project_link_arguments(gcc_link_args, language: ['cpp'])
    add_project_arguments('-D__USE_MINGW_ANSI_STDIO=1', language: ['cpp']) # https://github.com/rgommers/scipy/issues/118
  endif
endif 

## Extension modules
py.extension_module(
  '_simplextree',
  '_simplextree.cpp',
  install: true,
  include_directories: [inc_local, incdir_numpy, inc_pybind11]
)
py.extension_module(
  '_unionfind',
  '_unionfind.cpp',
  install: true,
  include_directories: [inc_local, incdir_numpy, inc_pybind11]
)

python_sources = [
  '__init__.py',
  'SimplexTree.py', 
  'UnionFind.py'
]

## Print install directory
message('Install directory for sources: ')
message(py.get_install_dir())

py.install_sources(
  python_sources,
  subdir: 'simplextree'
)