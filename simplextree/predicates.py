import numpy as np 
from typing import Iterable, Any
from numpy.typing import ArrayLike 
from .combinatorial import inverse_choose

def is_repeatable(x: Iterable) -> bool:
	"""Checks whether _x_ is Iterable and repeateable as an Iterable (generators fail this test)."""
	return iter(x) is not x

def is_array_convertible(x: Any) -> bool:
	return hasattr(x, "__array__")

def is_distance_matrix(x: ArrayLike) -> bool:
	"""Checks whether _x_ is a distance matrix, i.e. is square, symmetric, and that the diagonal is all 0."""
	x = np.array(x, copy=False)
	is_square = x.ndim == 2	and (x.shape[0] == x.shape[1])
	return(False if not(is_square) else np.all(np.diag(x) == 0))

def is_pairwise_distances(x: ArrayLike) -> bool:
	"""Checks whether 'x' is a 1-d array of pairwise distances."""
	x = np.array(x, copy=False) # don't use asanyarray here
	if x.ndim > 1: 
		return(False)
	n = inverse_choose(len(x), 2)
	return(x.ndim == 1 and n == int(n))

def is_point_cloud(x: ArrayLike) -> bool: 
	"""Checks whether 'x' is a 2-d array of points"""
	return(isinstance(x, np.ndarray) and x.ndim == 2)

def is_dist_like(x: ArrayLike):
	"""Checks whether _x_ is any recognizable distance object."""
	return(is_distance_matrix(x) or is_pairwise_distances(x))