import yaml
import quartodoc
from quartodoc import Builder, preview, blueprint, collect, MdRenderer
from quartodoc.builder.blueprint import BlueprintTransformer
from _renderer import Renderer as NumpyRenderer

## Configure builder 
cfg = yaml.safe_load(open("_quarto.yml", "r"))
builder = Builder.from_quarto_config(cfg)
# builder.renderer = NumpyRenderer()

# builder.renderer = MdRenderer(show_signature=True, show_signature_annotations=True, display_name="name")
# builder.renderer.display_name = 'name'
# builder.renderer.show_signature_annotations = True 

## Blueprint + collect  using google style parsing 
trans = BlueprintTransformer(parser="google")
bp = trans.visit(builder.layout)
pages, items = collect(bp, builder.dir)

builder.layout.sections[0].package = 'simplextree'

## Write the doc pages + the index  
builder.write_doc_pages(pages, "*")
builder.write_index(bp)
# builder.write_sidebar(bp)

# ## Simplex Tree
# preview(builder.layout.sections[0])

# ## The markdown text of each individual page...
# [renderer.render(p) for p in pages]

# preview(bp.sections[0], max_depth=8, compact=True)
# preview(bp.sections[0].contents[0].contents[0])

# ## Rendering individual docstrings 
# from quartodoc import get_object
# st_doc = get_object("splex.complexes", "SimplexTree.SimplexTree", parser="google")
# preview(st_doc.members['insert'], compact=True)

# from quartodoc.renderers import MdRenderer
# renderer = MdRenderer(header_level=2)
# print(renderer.render(st_doc.members['insert']))