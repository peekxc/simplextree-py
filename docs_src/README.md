
Reference documentation is built using the internal API of quartodoc. 

First, install `simplextree` into the current environment---NOT IN EDITABLE MODE (griffe limitation)

Then, relative to the `docs_src` directory, build the docs with:

> rm pages/reference/* & python gen_api.py & quarto render 

Optionally preview with: 

> quarto preview

or, for fast preview without the re-render, use [http-server](https://github.com/http-party/http-server): 

> http-server ../docs