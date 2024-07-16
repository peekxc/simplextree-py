import numpy as np
from itertools import combinations
from simplextree import SimplexTree

# def benchmark_expand():
st = SimplexTree()
simplex = np.arange(25)
st.insert(combinations(simplex, 2))
st.expand(k=5) ##  about 1 second