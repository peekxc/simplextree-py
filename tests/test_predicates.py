import numpy as np
from math import comb
from simplextree.predicates import *
from itertools import combinations

def test_predicates():
  x, y = range(10), np.arange(10)
  assert not is_repeatable((i for i in range(10))) and is_repeatable(y), "Repeatable predicate failed"
  assert not is_array_convertible(x) and is_array_convertible(y), "Array predicate failed"
  dist_x = np.array([np.linalg.norm(p - q) for p,q in combinations(range(10),2)])
  assert not is_distance_matrix(dist_x), "distance matrix failed"
  assert is_pairwise_distances(dist_x), "pairwsie distances failed"
  assert is_point_cloud(np.random.uniform(size=(10,2))) and not is_point_cloud(dist_x), "Point cloud check failed"
  assert is_dist_like(dist_x), "dist-like check failed"