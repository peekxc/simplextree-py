# ruff: noqa: PLR0904
from __future__ import annotations

import builtins
from numbers import Integral
from typing import Callable, Collection, Iterable, Iterator, Optional, Sequence, Union

import numpy as np
from _simplextree import SimplexTree as SimplexTreeCpp
from numpy.typing import ArrayLike


class SimplexTree(SimplexTreeCpp):
	"""SimplexTree provides lightweight wrapper around a Simplex Tree data structure.

	This class exposes a native extension module wrapping a simplex tree implemented with modern C++.

	The Simplex Tree was originally introduced in the paper:
	> Boissonnat, Jean-Daniel, and Clément Maria. "The simplex tree: An efficient data structure for general simplicial complexes." Algorithmica 70.3 (2014): 406-427.

	Attributes:
		n_simplices: number of simplices
		dimension: maximal dimension of the complex
		id_policy: policy for generating new vertex ids
		vertices: 0-simplices in the complex.
		edges: 1-simplices in the complex.
		triangles: 2-simplices in the complex.
		quads: 3-simplices in the complex.
		connected_components: connected component ids.
	"""

	def __init__(self, simplices: Optional[Iterable[Collection]] = None, s_type: Callable = tuple) -> None:
		assert callable(s_type), "Simplex type must have a constructor"
		SimplexTreeCpp.__init__(self)
		if simplices is not None:
			self.insert(simplices)
		self.s_type = s_type

	def insert(self, simplices: Iterable[Collection]) -> None:
		"""Inserts simplices into the Simplex Tree.

		By definition, inserting a simplex also inserts all of its faces. If the simplex already exists in the complex, the tree is not modified.

		Parameters:
			simplices: Iterable of simplices to insert (each of which are SimplexLike)

		::: {.callout-note}
		  If the iterable is an 2-dim np.ndarray, then a p-simplex is inserted along each contiguous p+1 stride.
			Otherwise, each element of the iterable to casted to a Simplex and then inserted into the tree.
		:::

		Examples:
			```{python}
			from simplextree import SimplexTree
			st = SimplexTree([range(3)])
			print(st)
			```
			```{python}
			st.insert([[0,1]])
			print(st)
			```
		"""
		if isinstance(simplices, np.ndarray):
			simplices = np.sort(simplices, axis=1).astype(np.uint16)
			assert simplices.ndim in {1, 2}, "dimensions should be 1 or 2"
			self._insert(simplices)
		else:
			assert isinstance(simplices, Iterable), f"Invalid simplices type '{type(simplices)}' given"
			self._insert_list(list(map(lambda x: np.asarray(x, dtype=np.uint16), simplices)))

	def remove(self, simplices: Iterable[Collection]):
		"""Removes simplices into the Simplex Tree.

		By definition, removing a face also removes all of its cofaces. If the simplex does not exist in the complex, the tree is not modified.

		Parameters:
			simplices: Iterable of simplices to insert (each of which are SimplexLike).

		::: {.callout-note}
			If the iterable is an 2-dim np.ndarray, then a p-simplex is removed along each contiguous p+1 stride.
			Otherwise, each element of the iterable to casted to a Simplex and then removed from the tree.
		:::

		Examples:
			```{python}
			from simplextree import SimplexTree
			st = SimplexTree([range(3)])
			print(st)
			```
			```{python}
			st.remove([[0,1]])
			print(st)
			```
		"""
		if isinstance(simplices, np.ndarray):
			simplices = np.sort(simplices, axis=1).astype(np.uint16)
			assert simplices.ndim in {1, 2}, "dimensions should be 1 or 2"
			self._remove(simplices)
		elif isinstance(simplices, Iterable):
			self._remove_list(list(map(lambda x: np.asarray(x, dtype=np.uint16), simplices)))
		else:
			raise ValueError("Invalid type given")

	def find(self, simplices: Iterable[Collection]) -> np.ndarray:
		"""Finds whether simplices exist in Simplex Tree.

		Parameters:
			simplices: Iterable of simplices to insert (each of which are SimplexLike)

		Returns:
			found: boolean array indicating whether each simplex was found in the complex

		::: {.callout-note}
			If the iterable is an 2-dim np.ndarray, then the p-simplex to find is given by each contiguous p+1 stride.
			Otherwise, each element of the iterable to casted to a Simplex and then searched for in the tree.
		:::
		"""
		if isinstance(simplices, np.ndarray):
			simplices = np.array(simplices, dtype=np.int16)
			assert simplices.ndim in {1, 2}, "dimensions should be 1 or 2"
			return self._find(simplices)
		elif isinstance(simplices, Iterable):
			return self._find_list([tuple(s) for s in simplices])
		else:
			raise ValueError("Invalid type given")

	def adjacent(self, vertices: Iterable[Collection]):
		"""Finds adjacent vertices of a collection of vertices."""
		return self._adjacent(list(map(lambda x: np.asarray(x, dtype=np.uint16), vertices)))

	def collapse(self, sigma: Collection, tau: Collection) -> bool:
		r"""Performs an elementary collapse on two given simplices.

		Checks whether its possible to collapse $\sigma$ through $\tau$, and if so, both simplices are removed.
		A simplex $\sigma$ is said to be collapsible through one of its faces $\tau$ if $\sigma$ is the only coface of $\tau$ (excluding $\tau$ itself).

		Parameters:
			sigma: maximal simplex to collapse
			tau: face of sigma to collapse

		Returns:
			whether the pair was collapsed

		Examples:
			```{python}
			from simplextree import SimplexTree
			st = SimplexTree([[0,1,2]])
			print(st)
			```
			```{python}
			st.collapse([0,1,2], [1,2])
			print(st)
			```
		"""
		if len(sigma) != (len(tau) + 1):  # , f"Simplex {tau} not in the boundary of simplex {sigma}"
			return False
		success = self._collapse(tau, sigma)
		return success

	def contract(self, pair: Collection) -> None:
		r"""Performs an pair contraction.

		This function performs an pair contraction: given a pair of vertices $(va, vb)$, vertex $vb$ is said to *contract*
		to $va$ if $vb$ is removed from the complex and the link of $va$ is augmented with the link of $vb$.

		Some notes about `pair` are in order:
			- `pair` is **not** sorted like other simplex inputs
			- The second vertex is always contracted to the first
			- `pair` need not be an existing simplex (edge) in the complex
			- Contraction is not symmetric.

		Parameters:
			pair: pair to contract

		Returns:
			whether the pair was contracted

		Examples:
			```{python}
			from simplextree import SimplexTree
			st = SimplexTree([[0,1,2]])
			st.print()
			```
			```{python}
			st.contract([0,2])
			st.print()
			```
		"""
		success = self._contract(pair)
		return success

	def vertex_collapse(self, u: int, v: int, w: int) -> bool:
		"""Maps a pair of vertices into a single vertex.

		Parameters:
			u: the first vertex in the free pair.
			v: the second vertex in the free pair.
			w: the target vertex to collapse to.

		Returns:
			collapsed: whether the collapse was performed.
		"""
		u, v, w = int(u), int(v), int(w)
		assert all([isinstance(e, Integral) for e in [u, v, w]]), "Unknown vertex types given; must be integral"
		return self._vertex_collapse(u, v, w)

	def degree(self, vertices: Optional[ArrayLike] = None) -> Union[ArrayLike, int]:
		"""Computes the degree of select vertices in the trie.

		Parameters:
			vertices: If no vertices are specified, all degrees are computed. Non-existing vertices by default have degree 0.

		Returns:
			degree of each vertex id given in 'vertices'.
		"""
		if vertices is None:
			return self._degree_default()
		elif isinstance(vertices, Iterable):
			vertices = np.fromiter(iter(vertices), dtype=np.int16)
			assert vertices.ndim == 1, "Invalid shape given; Must be flattened array of vertex ids"
			return self._degree(vertices)
		else:
			raise ValueError(f"Invalid type {type(vertices)} given")

	# PREORDER = 0, LEVEL_ORDER = 1, FACES = 2, COFACES = 3, COFACE_ROOTS = 4,
	# K_SKELETON = 5, K_SIMPLICES = 6, MAXIMAL = 7, LINK = 8
	def traverse(self, order: str = "preorder", f: Callable = builtins.print, sigma: Collection = [], p: int = 0) -> None:
		"""Traverses the simplex tree in the specified order, calling `f` on each simplex encountered.

		Supported traversals include breadth-first / level order ("bfs", "levelorder"), depth-first / prefix ("dfs", "preorder").
		faces, cofaces, coface roots ("coface_roots"), p-skeleton, p-simplices, maximal simplices ("maximal"), and link.

		Where applicable, each traversal begins its traversal `sigma`, which defaults to the empty set (root node).

		Parameters:
			order: the type of traversal of the simplex tree to execute.
			f: a function to evaluate on every simplex in the traversal. Defaults to print.
			sigma: simplex to start the traversal at, where applicable. Defaults to the root node (empty set).
			p: dimension of simplices to restrict to, where applicable. Defaults to 0.

		Returns:
			None
		"""
		# todo: handle kwargs
		assert isinstance(order, str)
		order = order.lower()
		if order in {"dfs", "preorder"}:
			order = 0
		elif order in {"bfs", "level_order", "levelorder"}:
			order = 1
		elif order == "faces":
			order = 2
		elif order == "cofaces":
			order = 3
		elif order == "coface_roots":
			order = 4
		elif order == "p-skeleton":
			order = 5
		elif order == "p-simplices":
			order = 6
		elif order == "maximal":
			order = 7
		elif order == "link":
			order = 8
		else:
			raise ValueError(f"Unknown order '{order}' specified")
		assert isinstance(int(p), int), f"Invalid argument type '{type(p)}', must be integral"
		sigma = [] if sigma is None else sigma
		if p >= 0:
			self._traverse(order, f, sigma, p)  # order, f, init, k

	def cofaces(self, sigma: Collection = []) -> list[Collection]:
		"""Returns the cofaces of `sigma`.

		Note, by definition, `sigma` is defined as a coface of itself.

		Parameters:
			sigma: the simplex to obtain cofaces of.

		Returns:
			cofaces: the cofaces of `sigma`.
		"""
		if sigma == [] or len(sigma) == 0:
			return self.simplices()
		F = []
		self._traverse(3, lambda s: F.append(self.s_type(s)), sigma, 0)  # order, f, init, k
		return F

	def coface_roots(self, sigma: Collection = []) -> list[Collection]:
		"""Returns the roots whose subtrees span the cofaces of `sigma`.

		Note that `sigma` itself is included in the set of its cofaces.

		Parameters:
			sigma: the simplex to obtain cofaces of. Defaults to the empty set (root node).

		Returns:
			coface_roots: the coface roots of `sigma`.
		"""
		F = []
		self._traverse(4, lambda s: F.append(self.s_type(s)), sigma, 0)  # order, f, init, k
		return F

	def skeleton(self, p: Optional[int] = None, sigma: Collection = []) -> Iterable[Collection]:
		"""Returns the simplices in the p-skeleton of `sigma`.

		Note that, when dim(`sigma`) <= `p`, `sigma` is included in the skeleton.

		Parameters:
			p: the dimension of the skeleton.
			sigma: the simplex to obtain cofaces of. Defaults to the empty set (root node).

		Returns:
			list: the simplices in the p-skeleton of `sigma`.
		"""
		p = self.dim() if p is None else p
		assert isinstance(int(p), int), f"Invalid argument type '{type(p)}', must be integral"
		F = []
		if p >= 0:
			self._traverse(5, lambda s: F.append(self.s_type(s)), sigma, p)
		return F

	def simplices(self, p: Optional[int] = None) -> Iterable[Collection]:
		"""Returns the p-simplices in the complex."""
		## NOTE: in either case, to traverse all simplices, we use sigma = []
		F = []
		if p is None:
			self.traverse("p-skeleton", lambda s: F.append(self.s_type(s)), sigma=[], p=self.dimension)
		else:
			self.traverse("p-simplices", lambda s: F.append(self.s_type(s)), sigma=[], p=int(p))
		return F

	def faces(self, p: Optional[int] = None, sigma: Optional[Collection] = None) -> Iterable[Collection]:
		"""Returns the p-faces of a given simplex."""
		if sigma is None:
			return self.simplices(p=p)
		else:
			F = []
			self.traverse("faces", lambda s: F.append(self.s_type(s)), sigma=sigma)
			if p is not None:
				F = list(filter(lambda s: len(s) == (p + 1), F))
			return F

	def maximal(self) -> Iterable[Collection]:
		"""Returns the maximal simplices in the complex."""
		F = []
		self._traverse(7, lambda s: F.append(self.s_type(s)), [], 0)
		return F

	def link(self, sigma: Collection = []) -> Iterable[Collection]:
		"""Returns the simplices in the link of `sigma`."""
		F = []
		self._traverse(8, lambda s: F.append(self.s_type(s)), sigma, 0)
		return F

	def expand(self, k: int, f: Optional[Callable[[Collection], bool]] = None) -> None:
		"""Performs a k-expansion of the complex.

		This function is particularly useful for expanding clique complexes beyond their 1-skeleton.

		Parameters:
			k: maximum dimension to expand to.
			f: boolean predicate which returns whether a simplex should added to the complex (and further expanded).

		Examples:
			```{python}
			from simplextree import SimplexTree
			from itertools import combinations
			st = SimplexTree(combinations(range(8), 2))
			print(st)
			```
			```{python}
			st.expand(2, lambda s: 2 in s)  # Expand only triangles containing 2 as a vertex
			print(st)
			```
			```{python}
			st.expand(2) # Expand all 2-cliques
			print(st)
			```
		"""
		assert int(k) >= 0, f"Invalid expansion dimension k={k} given"
		if f is None:
			self._expand(int(k))
		else:
			assert isinstance(f([]), bool), "Supplied callable must be boolean-valued"
			self._expand_f(int(k), f)

	def reindex(self, labels: Optional[Sequence] = None) -> None:
		"""Reindexes the vertex labels of the complex."""
		if len(self.n_simplices) == 0:
			return
		labels = list(range(self.n_simplices[0])) if labels is None else list(labels)
		assert len(labels) == self.n_simplices[0]
		self._reindex(labels)

	def __repr__(self) -> str:
		if len(self.n_simplices) == 0:
			return "< Empty simplex tree >"
		return f"Simplex Tree with {tuple(map(int, self.n_simplices))} {tuple(range(0,self.dimension+1))}-simplices"

	def __iter__(self) -> Iterator[Collection]:
		yield from self.simplices()

	def __contains__(self, s: Collection) -> bool:
		return bool(self.find([s])[0])

	def __len__(self) -> int:
		return int(sum(self.n_simplices))

	def dim(self) -> int:
		return len(self.n_simplices) - 1

	def card(self, p: Optional[int] = None) -> Union[int, tuple]:
		"""Returns the cardinality of various skeleta of the complex.

		Parameters:
			p: dimension parameter. Defaults to None.

		Returns:
			cardinalities: if p is an integer, the number of p-simplices in the complex. Otherwise a tuple indicating the number of simplices of all dimensions.
		"""
		if p is None:
			return tuple(self.n_simplices)
		else:
			assert isinstance(int(p), int), f"Invalid argument type '{type(p)}', must be integral"
			return 0 if p < 0 or p >= len(self.n_simplices) else self.n_simplices[p]

	def print(self, **kwargs):
		ST = np.zeros(shape=(np.sum(self.n_simplices), self.dimension + 1), dtype="<U15")
		ST.fill(" ")
		for i, s in enumerate(self):
			ST[i, : len(s)] = [str(si) for si in s]
		SC = np.apply_along_axis(lambda x: " ".join(x), axis=0, arr=ST)
		for i, s in enumerate(SC):
			ending = "\n" if i != (len(SC) - 1) else ""
			builtins.print(s, sep="", end=ending, **kwargs)
