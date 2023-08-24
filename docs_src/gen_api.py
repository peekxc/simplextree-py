## Ensure at the doc root
import os
doc_root = '/Users/mpiekenbrock/simplextree-py/docs_src'
os.chdir(doc_root)

import yaml
import quartodoc
from more_itertools import unique_everseen
from quartodoc import Builder, preview, blueprint, collect, MdRenderer
from quartodoc.builder.blueprint import BlueprintTransformer
from _renderer import Renderer as NumpyRenderer

## Configure builder 
cfg = yaml.safe_load(open("_quarto.yml", "r"))
builder = Builder.from_quarto_config(cfg)
builder.renderer = NumpyRenderer()
# builder.renderer = MdRenderer(show_signature=True, show_signature_annotations=True, display_name="name")
# builder.renderer.display_name = 'name'
# builder.renderer.show_signature_annotations = True 

## Preview the section layout
preview(builder.layout)

## Transform 
blueprint = BlueprintTransformer(parser="google").visit(builder.layout)
pages, items = collect(blueprint, builder.dir)

## Tree preview 
# preview(pages)

## Write the doc pages + the index  
builder.write_doc_pages(pages, "*")
builder.write_index(blueprint)
builder.write_sidebar(blueprint)

## Look at the insert function
insert_obj = pages[0].contents[0].members[9]
preview(insert_obj)

import quarto
from griffe.docstrings.dataclasses import DocstringSectionExamples

for ds_section in insert_obj.obj.docstring.parsed:
  if type(ds_section) == DocstringSectionExamples:
    for i, example_sec in enumerate(ds_section):
      ## Extract text as-is
      ds_txt_kind, ex_text = example_sec.value[0]    

      ## Option (1): put all the text within an executable cell
      quarto_cells = '```{python}\n' + ex_text + '```' 

      ## Option (2): split the text using some delimiter to separate outputs, e.g. two newlines 
      as_exec_cell = lambda text: '```{python}\n' + text + '\n```'
      quarto_cells = ''.join([as_exec_cell(example) for example in ex_text.split("\n\n")])

      ## Re-assign the text
      ds_section[i].value[1] = quarto_cells            

func_path = insert_obj.anchor.split('.')
# 'from ' + '.'.join(func_path[:-1]) + ' import ' + func_path[-1]
# builder.package

# import os 
# import tempfile
# tmp_file, filename = tempfile.mkstemp()
# with open("temp_file.py", "w+") as tmp_file: 
#   tmp_file.write(ex_text)
#   os.write(tmp_file, ex_text.encode('utf-8'))
# os.close(tmp_file)

# quarto.render("temp_file.qmd", output_format="html", output_file=, execute=True)


# items[9].obj.docstring.parsed
# preview(builder.renderer._UNHANDLED[0])

## Debugging -- NOTE: this only works with NumpyDoc parsing! 
# from quartodoc import get_object
# f_obj = get_object('simplextree', 'SimplexTree.insert')
# # print(f_obj.docstring.value)
# preview(f_obj.docstring.parsed)
# preview(f_obj.parameters)




## Settle unhandled with: builder.renderer._UNHANDLED[0]



# sys.path = list(unique_everseen([doc_root] + sys.path))

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