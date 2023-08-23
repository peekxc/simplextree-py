from quartodoc import MdRenderer
from griffe import dataclasses as dc
from griffe.docstrings import dataclasses as ds
from plum import dispatch
from tabulate import tabulate
from typing import *
from griffe.expressions import Name, Expression
from quartodoc.renderers import *
from quartodoc.ast import ExampleCode, ExampleText

Name, Expression = Name, Expression

class Renderer(MdRenderer):
  style = "markdown_numpy"
  display_name = 'name'
  show_signature_annotations = True 

  def __init__(self, *args, **kwargs):
    super().__init__()
    self._UNHANDLED = []

  # def _fetch_object_dispname(self, el: "dc.Alias | dc.Object"):
  #   if self.display_name == "parent":
  #     return el.parent.name + "." + el.name
  #   else: 
  #     return super(Renderer, self).render(el)
  
  # Keep admonition at the top here ----
  @dispatch
  def render(self, el: ds.DocstringSectionAdmonition) -> str:
    self._UNHANDLED.append(el)
    return "UNHANDLED ADMONITION"


  # Parameters ----
  @dispatch
  def render(self, el: ds.DocstringSectionParameters) -> str:
    params_str = []
    for ds_param in el.value:
      d = ds_param.as_dict()
      pn, pa, pd = [d.get(k) for k in ("name", "annotation", "description")]
      sec_md = f"**{pn}** : "
      if isinstance(pa, Name) or isinstance(pa, Expression):
        sec_md += f"<span class='type_annotation'> {pa.full}, </span>"
      else: 
        sec_md += "" if pa is None or len(str(pa)) == 0 else str(pa)+", "
      sec_md += f"optional (default={ d.get('value') })" if "value" in d.keys() else "required"
      sec_md += f"<p> {pd} </p>" 
      params_str.append(sec_md)
    return "\n\n".join(params_str)

  # @dispatch
  # def render(self, el: dc.Parameters):
  #   print(el)
  #   return "RENDER PARAMETERS"
    # return super(Renderer, self).render(el)

  # @dispatch
  # def render(self, el: dc.Parameter):
  #   return super(Renderer, self).render(el)

  # Returns ----
  @dispatch
  def render(self, el: Union[ds.DocstringSectionReturns, ds.DocstringSectionRaises]) -> str:
    params_str = []
    for ds_param in el.value:
      d = ds_param.as_dict()
      pn, pa, pd = [d.get(k) for k in ("name", "annotation", "description")]
      sec_md = f"**{pn}** : "
      if isinstance(pa, Name) or isinstance(pa, Expression):
        sec_md += f"<span class='type_annotation'> {pa.full}, </span>"
      else: 
        sec_md += "" if pa is None or len(str(pa)) == 0 else str(pa)+", "
      sec_md += f"<p> {pd} </p>" #style='margin-top: 10px;margin-left: 2.5em;
      params_str.append(sec_md)
    return "\n\n".join(params_str)

  # @dispatch
  # def render(self, el: dc.Parameter):
  #   splats = {dc.ParameterKind.var_keyword, dc.ParameterKind.var_positional}
  #   has_default = el.default and el.kind not in splats

  #   if el.kind == dc.ParameterKind.var_keyword:
  #       glob = "**"
  #   elif el.kind == dc.ParameterKind.var_positional:
  #       glob = "*"
  #   else:
  #       glob = ""

  #   annotation = self.render_annotation(el.annotation)
  #   if self.show_signature_annotations:
  #       if annotation and has_default:
  #           res = f"{glob}{el.name}: {el.annotation} = {el.default}"
  #       elif annotation:
  #           res = f"{glob}{el.name}: {el.annotation}"
  #   elif has_default:
  #       res = f"{glob}{el.name}={el.default}"
  #   else:
  #       res = f"{glob}{el.name}"

  #   return res

  # ## This shouldn't be triggered 
  # @dispatch
  # def render(self, el: Union[ds.DocstringReturn, ds.DocstringRaise]):
  #   _UNHANDLED.append(el)
  #   return "UNHANDLED RETURN"

  # --- Attributes
  @dispatch
  def render(self, el: ds.DocstringAttribute) -> str :
    self._UNHANDLED.append(el)
    for ds_attr in el.value:
      d = ds_attr.as_dict()
      pn, pa, pd = [d.get(k) for k in ("name", "annotation", "description")]
      print(pn)
    # return [pn, self._render_annotation(pa), pd]
    return "UNHANDLED ATTRIBUTE" 

  @dispatch
  def render(self, el: ds.DocstringSectionAttributes):
    header = ["Name", "Type", "Description"]
    rows = []
    for ds_attr in el.value:
      d = ds_attr.as_dict()
      pn, pa, pd = [d.get(k) for k in ("name", "annotation", "description")]
      rows.append([pn, self.render_annotation(pa), pd])
    return tabulate(rows, header, tablefmt="github")

  ## examples ----
  # @dispatch
  # def render(self, el: ds.DocstringSectionExamples) -> str:
  #   return super(Renderer, self).render(el)
  
  # @dispatch
  # def render(self, el: ExampleText) -> str:
  #   return super(Renderer, self).render(el)

  # ## Sections ---   
  # @dispatch
  # def render(self, el: ds.DocstringSectionText):
  #   return super(Renderer, self).render(el)

  # @dispatch
  # def render(self, el: ds.DocstringSection):
  #   _UNHANDLED.append(el)
  #   return "UNHANDLED SECTION"

  # @dispatch
  # def render(self, el: ExampleCode):
  #   return "```{python}\n" + el.value + "\n```"

  # @dispatch
  # def render(self, el) -> str:
  #   #raise NotImplementedError(f"Unsupported type of: {type(el)}")
  #   _UNHANDLED.append(el)
  #   import warnings
  #   warnings.warn(f"Unsupported type of: {type(el)}")
  #   return ""


