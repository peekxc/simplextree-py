
Reference documentation is built using the internal API of quartodoc. 

Build the docs with:

> rm pages/reference/* & python gen_api.py & quarto render 

Optionally preview with: 

> quarto preview

or, for fast preview without the re-render, use [http-server](https://github.com/http-party/http-server): 

> http-server ../docs