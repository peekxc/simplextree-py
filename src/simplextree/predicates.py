import numpy as np 
from typing import *
from numpy.typing import ArrayLike 

from math import comb, factorial
# from .combinatorial import * 

def inverse_choose(x: int, k: int):
  """Inverse binomial coefficient (approximately). 

  This function *attempts* to find the integer _n_ such that binom(n,k) = x, where _binom_ is the binomial coefficient: 

  binom(n,k) := n!/(k! * (n-k)!)

  For k <= 2, an efficient iterative approach is used and the result is exact. For k > 2, the same approach is 
  used if x > 10e7; otherwise, an approximation is used based on the formula from this stack exchange post: 

  https://math.stackexchange.com/questions/103377/how-to-reverse-the-n-choose-k-formula
  """
  assert k >= 1, "k must be >= 1" 
  if k == 1: return(x)
  if k == 2:
    rng = np.array(list(range(int(np.floor(np.sqrt(2*x))), int(np.ceil(np.sqrt(2*x)+2) + 1))))
    final_n = rng[np.nonzero(np.array([comb(n, 2) for n in rng]) == x)[0].item()]
  else:
    # From: https://math.stackexchange.com/questions/103377/how-to-reverse-the-n-choose-k-formula
    if x < 10**7:
      lb = (factorial(k)*x)**(1/k)
      potential_n = np.array(list(range(int(np.floor(lb)), int(np.ceil(lb+k)+1))))
      idx = np.nonzero(np.array([comb(n, k) for n in potential_n]) == x)[0].item()
      final_n = potential_n[idx]
    else:
      lb = np.floor((4**k)/(2*k + 1))
      C, n = factorial(k)*x, 1
      while n**k < C: n = n*2
      m = (np.nonzero( np.array(list(range(1, n+1)))**k >= C )[0])[0].item()
      potential_n = np.array(list(range(int(np.max([m, 2*k])), int(m+k+1))))
      if len(potential_n) == 0: 
        raise ValueError(f"Failed to invert C(n,{k}) = {x}")
      ind = np.nonzero(np.array([comb(n, k) for n in potential_n]) == x)[0].item()
      final_n = potential_n[ind]
  return(final_n)

def is_repeatable(x: Iterable) -> bool:
	"""Checks whether _x_ is Iterable and repeateable as an Iterable (generators fail this test)."""
	return not(iter(x) is x)

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
	if x.ndim > 1: return(False)
	n = inverse_choose(len(x), 2)
	return(x.ndim == 1 and n == int(n))

def is_point_cloud(x: ArrayLike) -> bool: 
	"""Checks whether 'x' is a 2-d array of points"""
	return(isinstance(x, np.ndarray) and x.ndim == 2)

def is_dist_like(x: ArrayLike):
	"""Checks whether _x_ is any recognizable distance object."""
	return(is_distance_matrix(x) or is_pairwise_distances(x))