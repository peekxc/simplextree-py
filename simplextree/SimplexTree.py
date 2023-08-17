from __future__ import annotations
from typing import *
from numbers import Integral
from numpy.typing import ArrayLike

import numpy as np 
from _simplextree import SimplexTree as SimplexTreeCpp

class SimplexTree(SimplexTreeCpp):
	""" 
	SimplexTree provides lightweight wrapper around a Simplex Tree data structure: an ordered, trie-like structure whose nodes are in bijection with the faces of the complex. 
	This class exposes a native extension module wrapping a simplex tree implemented with modern C++.

	The Simplex Tree was originally introduced in the paper
	> Boissonnat, Jean-Daniel, and Clément Maria. "The simplex tree: An efficient data structure for general simplicial complexes." Algorithmica 70.3 (2014): 406-427.

	Attributes:
		n_simplices (ndarray): number of simplices 
		dimension (int): maximal dimension of the complex
		id_policy (str): policy for generating new vertex ids 

	Attributes: Properties:
		vertices (ndarray): vertices of the complex
	"""    
	def __init__(self, simplices: Iterable[Collection] = None) -> None:
		SimplexTreeCpp.__init__(self)
		if simplices is not None: 
			self.insert(simplices)
		return None

	def insert(self, simplices: Iterable[Collection]) -> None:
		"""
		Inserts simplices into the Simplex Tree. 

		By definition, inserting a simplex also inserts all of its faces. If the simplex already exists in the complex, the tree is not modified. 

		Parameters:
			simplices: Iterable of simplices to insert (each of which are SimplexLike)

		::: {.callout-note}
		 	If the iterable is an 2-dim np.ndarray, then a p-simplex is inserted along each contiguous p+1 stride.
			Otherwise, each element of the iterable to casted to a Simplex and then inserted into the tree. 
		:::
		"""
		if isinstance(simplices, np.ndarray):
			simplices = np.sort(simplices, axis=1).astype(np.uint16)
			assert simplices.ndim in [1,2], "dimensions should be 1 or 2"
			self._insert(simplices)
		elif isinstance(simplices, Iterable): 
			self._insert_list(list(map(lambda x: np.asarray(x, dtype=np.uint16), simplices)))
		else: 
			raise ValueError("Invalid type given")

	def remove(self, simplices: Iterable[Collection]):
		"""
		Removes simplices into the Simplex Tree. 

		By definition, removing a face also removes all of its cofaces. If the simplex does not exist in the complex, the tree is not modified. 

		Parameters:
			simplices: 
				Iterable of simplices to insert (each of which are SimplexLike).		
		
		::: {.callout-note}
		 	If the iterable is an 2-dim np.ndarray, then a p-simplex is removed along each contiguous p+1 stride.
			Otherwise, each element of the iterable to casted to a Simplex and then removed from the tree. 
		:::

		Examples:
			st = SimplexTree([range(3)])
			print(st)
			st.remove([[0,1]])
			print(st)
		"""
		if isinstance(simplices, np.ndarray):
			simplices = np.sort(simplices, axis=1).astype(np.uint16)
			assert simplices.ndim in [1,2], "dimensions should be 1 or 2"
			self._remove(simplices)
		elif isinstance(simplices, Iterable): 
			self._remove_list(list(map(lambda x: np.asarray(x, dtype=np.uint16), simplices)))
		else: 
			raise ValueError("Invalid type given")

	def find(self, simplices: Iterable[Collection]):
		"""
		Finds whether simplices exist in Simplex Tree. 
		
		Parameters:
			simplices: Iterable of simplices to insert (each of which are SimplexLike)

		Returns:
			found (ndarray) : boolean array indicating whether each simplex was found in the complex

		::: {.callout-note}
		 	If the iterable is an 2-dim np.ndarray, then the p-simplex to find is given by each contiguous p+1 stride.
			Otherwise, each element of the iterable to casted to a Simplex and then searched for in the tree. 
		:::
		"""
		if isinstance(simplices, np.ndarray):
			simplices = np.array(simplices, dtype=np.int16)
			assert simplices.ndim in [1,2], "dimensions should be 1 or 2"
			return self._find(simplices)
		elif isinstance(simplices, Iterable): 
			return self._find_list([tuple(s) for s in simplices])
		else: 
			raise ValueError("Invalid type given")

	def adjacent(self, simplices: Iterable[Collection]):
		"""Checks for adjacencies between simplices."""
		return self._adjacent(list(map(lambda x: np.asarray(x, dtype=np.uint16), simplices)))

	def collapse(self, tau: Collection, sigma: Collection) -> None:
		"""Performs an elementary collapse on two given simplices. 

		Checks whether its possible to collapse $\\sigma$ through $\\tau$, and if so, both simplices are removed. 
		A simplex $\\sigma$ is said to be collapsible through one of its faces $\\tau$ if $\\sigma$ is the only coface of $\\tau$ (excluding $\\tau$ itself). 
		
		Parameters:
			sigma : maximal simplex to collapse
			tau : face of sigma to collapse 

		Returns:
			bool: whether the pair was collapsed

		Examples:

			from splex import SimplexTree 
			st = SimplexTree([[0,1,2]])
			print(st)

			st.collapse([1,2], [0,1,2])

			print(st)
		"""
		# assert tau in boundary(sigma), f"Simplex {tau} is not in the boundary of simplex {sigma}"
		success = self._collapse(tau, sigma) 
		return success 

	def vertex_collapse(self, u: int, v: int, w: int):
		"""Maps a pair of vertices into a single vertex. 
		
		Parameters:
			u (int): the first vertex in the free pair.
			v (int): the second vertex in the free pair. 
			w (int): the target vertex to collapse to.
		"""
		u,v,w = int(u), int(v), int(w)
		assert all([isinstance(e, Integral) for e in [u,v,w]]), f"Unknown vertex types given; must be integral"
		self._vertex_collapse(u,v,w)

	def degree(self, vertices: Optional[ArrayLike] = None) -> Union[ArrayLike, int]:
		"""Computes the degree of select vertices in the trie.

		Parameters:
			vertices (ArrayLike): Retrieves vertex degrees
				If no vertices are specified, all degrees are computed. Non-existing vertices by default have degree 0. 
		
		Returns:
			list: degree of each vertex id given in 'vertices'
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
	def traverse(self, order: str = "preorder", f: Callable = print, sigma: Collection = [], p: int = 0) -> None:
		"""Traverses the simplex tree in the specified order, calling 'f' on each simplex encountered.

		Supported traversals: 
			- breadth-first / level order ("bfs", "levelorder") 
			- depth-first / prefix ("dfs", "preorder")
			- faces 
			- cofaces
			- coface roots ("coface_roots")
			- p-skeleton
			- p-simplices 
			- maximal simplices ("maximal")
			- link 
		To select one of these options, set order to one of ["bfs", "levelorder", "dfs", "preorder"]

		Parameters:
			order: the type of traversal of the simplex tree to execute.
			f: a function to evaluate on every simplex in the traversal. Defaults to print. 
			sigma: simplex to start the traversal at, where applicable. Defaults to the root node (empty set).
			p: dimension of simplices to restrict to, where applicable. Defaults to 0. 
		"""
		# todo: handle kwargs
		assert isinstance(order, str)
		order = order.lower() 
		if order in ["dfs", "preorder"]:
			order = 0
		elif order in ["bfs", "level_order", "levelorder"]:
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
		self._traverse(order, lambda s: f(s), sigma, p) # order, f, init, k

	def cofaces(self, sigma: Collection = []) -> list[Collection]:
		"""Returns the p-cofaces of a given simplex.

		Parameters:
			p : coface dimension to restrict to 
			sigma : the simplex to obtain cofaces of

		Returns:
			list: the p-cofaces of sigma
		"""
		if sigma == [] or len(sigma) == 0:
			return self.simplices()
		F = []
		self._traverse(3, lambda s: F.append(s), sigma, 0) # order, f, init, k
		return F
	
	def coface_roots(self, sigma: Collection = []) -> Iterable[Collection]:
		"""Returns the simplex 'roots' of a given simplex whose subtrees generate its cofaces."""
		F = []
		self._traverse(4, lambda s: F.append(s), sigma, 0) # order, f, init, k
		return F

	def skeleton(self, p: int = None, sigma: Collection = []) -> Iterable[Collection]:
		"""Returns the simplices in the p-skeleton of the complex."""
		F = []
		self._traverse(5, lambda s: F.append(s), sigma, p)
		return F 

	def simplices(self, p: int = None) -> Iterable[Collection]:
		"""Returns the p-simplices in the complex."""
		F = []
		if p is None: 
			self._traverse(1, lambda s: F.append(s), [], 0) # order, f, init, k
		else: 
			assert isinstance(int(p), int), f"Invalid argument type '{type(p)}', must be integral"
			self._traverse(6, lambda s: F.append(s), [], p) # order, f, init, k
		return F
	
	def faces(self, p: int = None, sigma: Collection = [], **kwargs) -> Iterable[Collection]:
		"""Wrapper for simplices function."""
		return self.skeleton(p, sigma)
	
	def maximal(self) -> Iterable[Collection]:
		"""Returns the maximal simplices in the complex."""
		F = []
		self._traverse(7, lambda s: F.append(s), [], 0)
		return F

	def link(self, sigma: Collection = []) -> Iterable[Collection]:
		"""Returns all simplices in the link of a given simplex."""
		F = []
		self._traverse(8, lambda s: F.append(s), sigma, 0)
		return F

	def expand(self, k: int) -> None:
		"""Performs a k-expansion of the complex.

		This function is particularly useful for expanding clique complexes beyond their 1-skeleton. 
		
		Parameters:
			k : maximum dimension to expand to. 

		Examples:

			from simplextree import SimplexTree 
			from itertools import combinations 
			st = SimplexTree(combinations(range(8), 2))
			print(st)
			
			st.expand(k=2)
			print(st)
		"""
		assert int(k) >= 0, f"Invalid expansion dimension k={k} given"
		self._expand(int(k))

	def __repr__(self) -> str:
		if len(self.n_simplices) == 0:
			return "< Empty simplex tree >"
		return f"Simplex Tree with {tuple(self.n_simplices)} {tuple(range(0,self.dimension+1))}-simplices"

	def __iter__(self) -> Iterator[Collection]:
		yield from self.simplices()

	def __contains__(self, s: Collection) -> bool:
		return bool(self.find([s])[0])

	def __len__(self) -> int:
		return int(sum(self.n_simplices))

	def card(self, p: int = None):
		"""Returns the cardinality of various skeleta of the complex."""
		if p is None: 
			return tuple(self.n_simplices)
		else: 
			assert isinstance(p, int), "Invalid p"
			return 0 if p < 0 or p >= len(self.n_simplices) else self.n_simplices[p]
