# Changelog

All notable changes to this project will be documented in this file.

This project loosely adheres to [semantic versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased [v0.1.3] - 2024-07-16

## [v0.1.3](https://github.com/peekxc/simplextree-py/releases/tag/v0.1.3) - 2024-07-16

## [v0.1.1](https://github.com/peekxc/simplextree-py/releases/tag/v0.1.1) - 2024-07-16

- Added ability to change the default simplex output type via the `s_type` Callable
- Minor performance improvement across the board by switching to vector-based node children 

## [v0.1.2]

- Exported underlying `st.reindex()` function to allow node relabeling
- Re-added vertex/edge contraction to the API

## [v0.1.1](https://github.com/peekxc/simplextree-py/releases/tag/v0.1.1) - 2023-08-20

- Added Conditional k-expansion now available via `expand(k=k, f=...)` supporting arbitrary predicates `f`
- Online docs now available on GH pages with quartodoc

## [v0.1.0](https://github.com/peekxc/simplextree-py/releases/tag/v0.1.0) - 2023-08-17

- Initial tested release to pypi
- Added GH workflows + cibuildwheels 