# simplextree
[![Tests](https://github.com/peekxc/simplextree-py/actions/workflows/package.yml/badge.svg)](https://github.com/peekxc/splex/actions/workflows/package.yml)
[![coverage_badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/peekxc/ef42349965f40edf4232737026690c5f/raw/coverage_info.json)](https://coveralls.io/github/peekxc/simplextree-py)
[![python_badge](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://github.com/peekxc/splex/actions/workflows/python-package.yml)
[![coverage_badge](https://img.shields.io/github/actions/workflow/status/peekxc/simplextree-py/build-macos.yml?logo=apple&logoColor=white)](https://github.com/peekxc/splex/actions/workflows/build-macos.yml)
[![coverage_badge](https://img.shields.io/github/actions/workflow/status/peekxc/simplextree-py/build-windows.yml?logo=windows&logoColor=white)](https://github.com/peekxc/splex/actions/workflows/build-windows.yml)
[![coverage_badge](https://img.shields.io/github/actions/workflow/status/peekxc/simplextree-py/build-linux.yml?logo=linux&logoColor=white)](https://github.com/peekxc/splex/actions/workflows/build-linux.yml)

`simplextree` is an Python package that simplifies computation for general [simplicial complexes](https://en.wikipedia.org/wiki/Simplicial_complex) of any dimension by providing [pybind11](https://github.com/pybind/pybind11) bindings to a _Simplex Tree_ data structure implemented in modern C++17. 

A _Simplex Tree_ is an ordered, [trie](https://en.wikipedia.org/wiki/Trie)-like structure whose nodes are in bijection with the faces of the complex. Here's a picture of a simplicial 3-complex (left) and its corresponding Simplex Tree (right):

![simplex tree picture](./docs/pages/static/simplextree_pic.png)

The _Simplex Tree_ was originally introduced in the following paper:

> Boissonnat, Jean-Daniel, and Clément Maria. "The simplex tree: An efficient data structure for general simplicial complexes." Algorithmica 70.3 (2014): 406-427.

The `SimplexTree` class exported by the package includes support for many tree operations, e.g. insertions, removals, expansions, collapses, star/link enumerations, and other traversals. 

## Install 

The easiest way to install the package is via the platform-specific [wheels](https://pythonwheels.com/) on [pypi](https://pypi.org/project/simplextree/). 

```bash 
python -m pip install simplextree 
```

You can also `pip install` the package manually by downloading the appropriate wheel (or sdist) from the [releases](https://github.com/peekxc/simplextree-py/releases/). 

For installation instructions for developers looking to extend the package, see [Building & Developing](#building--developing). 

## Quickstart

```python
## The SimplexTree class provides light wrapper around the extension module
from simplextree import SimplexTree 
st = SimplexTree([[0,1,2], [0,1], [4,5]]) 
print(st) 
# Simplex Tree with (5, 4, 1) (0, 1, 2)-simplices

## Batch insertion, removal, and membership queries are supported
st.insert([[1,4], [1,5], [6]])
# Simplex Tree with (6, 6, 1) (0, 1, 2)-simplices 

st.remove([[6]])
# Simplex Tree with (5, 6, 1) (0, 1, 2)-simplices

st.find([[6], [0,1]])
# array([False,  True])

## Collections of simplices are returned as simple lists-of-lists
print(st.simplices())
# [[0],[1],[2],[4],[5], [0,1],[0,2],[1,2],[1,4],[1,5],[4,5],[0,1,2]])

print(st.skeleton(1)) 
# [[0],[1],[2],[4],[5], [0,1],[0,2],[1,2],[1,4],[1,5],[4,5]])

## Familiar Pythonic collection semantics are supported, including contains and iteration
[0,1,2] in st
# True 

[len(simplex) for simplex in st]
# [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3]

## Various subsets of the complex can be enumerated
st.cofaces([1])
# [[1], [0,1], [1,2], [1,4], [1,5], [0,1,2], [1,4,5]]

st.maximal()
# [[0, 1, 2], [1, 4], [1, 5], [4, 5]]

## Basic properties are also available as attributes 
st.connected_components 
# [1,1,1,1,1]

st.vertices
# [0,1,2,4,5]

st.n_simplices, st.dimension
# [5, 6, 1], 2

## Interoperability with numpy is provided whenever possible
all(np.all(st.triangles == np.array(st.simplices(p=2)), axis=0))
# True 

## Other complex-wide operations are supported, like k-expansions 
st.insert([[1,4]]) 
st.expand(2)       
# Simplex Tree with (6, 6, 2) (0, 1, 2)-simplices

## The trie-structure can also be inspected on the python side 
st.print_tree()
# 0 (h = 2): .( 1 2 )..( 2 )
# 1 (h = 1): .( 2 4 5 )
# 2 (h = 0): 
# 4 (h = 1): .( 5 )
# 5 (h = 0): 

st.print_cousins()
# (last=1, depth=2): { 0 1 } 
# (last=2, depth=2): { 0 2 } { 1 2 } 
# (last=4, depth=2): { 1 4 } 
# (last=5, depth=2): { 4 5 } { 1 5 } 
# (last=2, depth=3): { 0 1 2 } 
```

## Building & Developing 

If you would like to build the package yourself for development reasons, a typical workflow is to install the [build-time dependencies](https://pip.pypa.io/en/stable/reference/build-system/pyproject-toml/#build-time-dependencies) first: 

```bash
python -m pip install meson-python ninja pybind11 numpy
```

Then, build and install the package in [editable mode](https://peps.python.org/pep-0660/) (see also [meson-python notes](https://meson-python.readthedocs.io/en/latest/how-to-guides/editable-installs.html)), optionally without build isolation for speed:

```bash
python -m pip install --no-build-isolation --editable .
```

Unit testing is handled with [pytest](https://docs.pytest.org/en/7.4.x/). See the [gh-workflows](https://github.com/peekxc/simplextree-py/actions) for platform-specific configuration. 

## Native Extensions 

The underlying C++ library is [header-only](https://en.wikipedia.org/wiki/Header-only) and may be included as a dependency by [extension modules](https://docs.python.org/3/extending/extending.html) in other Python packages.

Thus, to modify or extending the complex in C++, it is sufficient to add the package as a build-time dependency and append the include directory to the compilation target.