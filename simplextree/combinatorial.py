import numpy as np 
from typing import * 
from itertools import * 
from numbers import Integral
from math import floor, ceil, comb, factorial
# import _combinatorial as comb_mod
from more_itertools import collapse, spy, first_true

## On the naming convention:
## SimplexWrapper is a 
## Also: https://stackoverflow.com/questions/1942328/add-a-member-variable-method-to-a-python-generator
## See: https://stackoverflow.com/questions/48349929/numpy-convertible-class-that-correctly-converts-to-ndarray-from-inside-a-sequenc
class SimplexWrapper:
  ## Precondition: Generator contains containers all of equal length (d)
  def __init__(self, g: Generator, d: int, dtype = None):
    self.simplices = g 
    if d == 0:
      self.s_dtype = np.uint16 if dtype is None else dtype
    else:
      self.s_dtype = (np.uint16, d+1) if dtype is None else (dtype, d+1)
  
  def __iter__(self) -> Iterator:
    # seq = self.simplices if isinstance(self.s_dtype, tuple) else collapse(self.simplices)
    return iter(self.simplices)
    # if isinstance(self,self.s_dtype):
    #   return map(lambda x: np.asarray(x, dtype=self.s_dtype), self.simplices)
    # else:
    #   return iter(np.fromiter(collapse(self.simplices), dtype=self.s_dtype))

  def __array__(self) -> np.ndarray:
    seq = iter(self) if isinstance(self.s_dtype, tuple) else collapse(iter(self))
    return np.fromiter(seq, dtype=self.s_dtype)

def c2_lex_rank(i: int, j: int, n: int) -> int:
  i, j = (j, i) if j < i else (i, j)
  return(int(n*i - i*(i+1)/2 + j - i - 1))

def c2_lex_unrank(x: int, n: int) -> tuple:
  i = int(n - 2 - np.floor(np.sqrt(-8*x + 4*n*(n-1)-7)/2.0 - 0.5))
  j = int(x + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2)
  return(i,j) 

def comb_unrank_lex(r: int, k: int, n: int):
  result = [0]*k
  x = 1
  for i in range(1, k+1):
    while(r >= comb(n-x, k-i)):
      r -= comb(n-x, k-i)
      x += 1
    result[i-1] = (x - 1)
    x += 1
  return tuple(result)

def comb_rank_lex(c: Iterable, n: int) -> int:
  c = tuple(sorted(c))
  k = len(c)
  index = sum([comb(int(n-ci-1),int(k-i)) for i,ci in enumerate(c)])
  #index = sum([comb((n-1)-cc, kk) for cc,kk in zip(c, reversed(range(1, len(c)+1)))])
  return int(comb(n, k) - index - 1)

def comb_rank_colex(c: Iterable) -> int:
  c = tuple(sorted(c))
  k = len(c)
  #return sum([comb(ci, i+1) for i,ci in zip(reversed(range(len(c))), reversed(c))])
  return sum([comb(ci,k-i) for i,ci in enumerate(reversed(c))])

def comb_unrank_colex(r: int, k: int) -> tuple:
  """
  Unranks a k-combinations rank 'r' back into the original combination in colex order
  
  From: Unranking Small Combinations of a Large Set in Co-Lexicographic Order
  """
  c = [0]*k
  for i in reversed(range(1, k+1)):
    m = i
    while r >= comb(m,i):
      m += 1
    c[i-1] = m-1
    r -= comb(m-1,i)
  return tuple(c)


def comb_to_rank(C: Iterable[tuple], n: int = None, order: str = ["colex", "lex"]) -> int:
  """
  Ranks k-combinations to integer ranks in either lexicographic or colexicographical order
  
  Parameters:
    C : combination, or Iterable of combinations.
    n : cardinality of the set (lex order only).
    order : the bijection to use.
  
  Returns:
    list : unsigned integers ranks in the chosen order.
  """
  (el,), C = spy(C) 
  if order == ["colex", "lex"] or order == "colex":
    return comb_rank_colex(C) if isinstance(el, Integral) else [comb_rank_colex(c) for c in C]
  else:
    assert n is not None, "Cardinality of set must be supplied for lexicographical ranking"
    return comb_rank_lex(C, n) if isinstance(el, Integral) else [comb_rank_lex(c, n) for c in C]

def rank_to_comb(R: Iterable[int], k: Union[int, Iterable], n: int = None, order: str = ["colex", "lex"]):
  """
  Unranks integer ranks to  k-combinations in either lexicographic or colexicographical order.
  
  Parameters:
    R : Iterable of integer ranks 
    n : cardinality of the set (only required for lex order)
    order : the bijection to use
  
  Returns:
    list : k-combinations derived from R
  """
  # R = [R] if isinstance(R, Integral) else R ## convert single rank into 1-element list  
  if order == ["colex", "lex"] or order == "colex":
    if isinstance(R, Integral):
      return comb_unrank_colex(R, k=k)
    if isinstance(k, Integral):
      return SimplexWrapper((comb_unrank_colex(r, k) for r in R), d=k-1)
    else: 
      assert len(k) == len(R), "If 'k' is an iterable it must match the size of 'R'"
      return [comb_unrank_colex(r, l) for l, r in zip(k,R)]
  else: 
    assert n is not None, "Cardinality of set must be supplied for lexicographical ranking"
    if isinstance(R, Integral):
      return comb_unrank_colex(R, k=k, n=n)
    if isinstance(k, Integral):
      assert k > 0, f"Invalid cardinality {k}"
      if k == 1:
        return SimplexWrapper(((r,) for r in R), d=0)
      if k == 2:
        return SimplexWrapper((c2_lex_unrank(r, n) for r in R), d=1)
        # return [unrank_C2(r, n) for r in R]
      else: 
        return SimplexWrapper((comb_unrank_lex(r, k, n) for r in R), d=k-1)
        # return [unrank_lex(r, k, n) for r in R]
    else:
      assert len(k) == len(R), "If 'k' is an iterable it must match the size of 'R'"
      return SimplexWrapper((comb_unrank_lex(r, k_) for k_, r in zip(k,R)))

def inverse_choose(x: int, k: int):
  """Inverse binomial coefficient (approximately). 

  This function *attempts* to find the integer _n_ such that binom(n,k) = x, where _binom_ is the binomial coefficient: 

  binom(n,k) := n!/(k! * (n-k)!)

  For k <= 2, an efficient iterative approach is used and the result is exact. For k > 2, the same approach is 
  used if x > 10e7; otherwise, an approximation is used based on the formula from this stack exchange post: 

  https://math.stackexchange.com/questions/103377/how-to-reverse-the-n-choose-k-formula
  """
  assert k >= 1 and x >= 0, "k must be >= 1" 
  if k == 1: return(x)
  if k == 2:
    rng = np.arange(np.floor(np.sqrt(2*x)), np.ceil(np.sqrt(2*x)+2) + 1, dtype=np.uint64)
    final_n = rng[np.searchsorted((rng * (rng - 1) / 2), x)]
    if comb(final_n, 2) == x:
      return final_n
    raise ValueError(f"Failed to invert C(n,{k}) = {x}")
    # return int(rng[x == (rng * (rng - 1) / 2)])
  else:
    # From: https://math.stackexchange.com/questions/103377/how-to-reverse-the-n-choose-k-formula
    if x < 10**7:
      lb = (factorial(k)*x)**(1/k)
      potential_n = range(floor(lb), ceil(lb+k)+1)
      comb_cand = [comb(n, k) for n in potential_n]
      if x in comb_cand:
        return potential_n[comb_cand.index(x)]
    else:
      lb = np.floor((4**k)/(2*k + 1))
      C, n = factorial(k)*x, 1
      while n**k < C: n = n*2
      m = first_true((c**k for c in range(1, n+1)), pred=lambda c: c**k >= C)
      potential_n = range(min([m, 2*k]), m+k+1)
      if len(potential_n) == 0: 
        raise ValueError(f"Failed to invert C(n,{k}) = {x}")
      final_n = first_true(potential_n, default = -1, pred = lambda n: comb(n,k) == x)
      if final_n != -1:
        return final_n
      else: 
        from scipy.optimize import minimize_scalar
        binom_loss = lambda n: np.abs(comb(int(n), k) - x)
        res = minimize_scalar(binom_loss, bounds=(comb(2*k, k), x))
        n1, n2 = int(np.floor(res.x)), int(np.ceil(res.x))
        if comb(n1,k) == x: return n1 
        if comb(n2,k) == x: return n2 
        raise ValueError(f"Failed to invert C(n,{k}) = {x}")