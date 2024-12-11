import numpy as np
from math import comb
from simplextree.combinatorial import *
from itertools import combinations
from functools import partial



def test_colex():
  n, k = 10, 3
  ranks = np.array([comb_rank_colex(c) for c in combinations(range(n), k)])
  assert all(np.sort(ranks) == np.arange(comb(n,k))), "Colex ranking is not unique / does not form bijection"
  ranks2 = np.array([comb_rank_colex(reversed(c)) for c in combinations(range(n), k)])
  assert all(ranks == ranks2), "Ranking is not order-invariant"
  combs_test = np.array([comb_unrank_colex(r, k) for r in ranks])
  combs_truth = np.array(list(combinations(range(n),k)))
  assert all((combs_test == combs_truth).flatten()), "Colex unranking invalid"

def test_array_conversion():
  x = np.array(rank_to_comb([0,1,2], k=2))
  assert np.all(x == np.array([[0,1], [0,2], [1,2]], dtype=np.uint16))

def test_lex():
  n, k = 10, 3
  ranks = np.array([comb_rank_lex(c, n) for c in combinations(range(n), k)])
  assert all(ranks == np.arange(comb(n,k))), "Lex ranking is not unique / does not form bijection"
  ranks2 = np.array([comb_rank_lex(reversed(c), n) for c in combinations(range(n), k)])
  assert all(ranks == ranks2), "Ranking is not order-invariant"
  combs_test = np.array([comb_unrank_lex(r, k, n) for r in ranks])
  combs_truth = np.array(list(combinations(range(n),k)))
  assert all((combs_test == combs_truth).flatten()), "Lex unranking invalid"

def test_api():
  n = 20
  for d in range(1, 5):
    combs = list(combinations(range(n), d))
    ranks = comb_to_rank(combs, n=n, order="lex")
    combs_test = rank_to_comb(ranks, k=d, n=n, order="lex")
    assert all([c1 == c2 for c1, c2 in zip(combs, combs_test)])

def test_inverse():
  from math import comb
  assert inverse_choose(10, 2) == 5
  assert inverse_choose(45, 2) == 10
  comb2 = partial(lambda x: comb(x, 2))
  comb3 = partial(lambda x: comb(x, 3))
  N = [10, 12, 16, 35, 48, 78, 101, 240, 125070]
  for n, x in zip(N, map(comb2, N)):
    assert inverse_choose(x, 2) == n
  for n, x in zip(N, map(comb3, N)):
    assert inverse_choose(x, 3) == n
  assert inverse_choose(comb(12501, 15), 15) == 12501


def test_rank2():
  N = 25
  combo_gen = lambda: combinations(range(N), 2)
  ranks = np.array([c2_lex_rank(i,j, n=N) for i,j in combo_gen()])
  assert np.allclose(ranks, np.arange(comb(N,2)))
  assert all([c2_lex_unrank(r,n=N) == c for r, c in zip(ranks, combo_gen())])
