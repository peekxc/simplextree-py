## python -m pytest tests/
import numpy as np 
from simplextree import SimplexTree
from more_itertools import unique_everseen

def test_construct():
  s = SimplexTree()
  assert str(type(s)) == "<class 'simplextree.SimplexTree.SimplexTree'>"
  st = SimplexTree([[0,1,2,3,4]])
  assert all(st.n_simplices == np.array([5,10,10,5,1]))

## Minimal API tests
def test_SimplexTree():
  st = SimplexTree()
  assert repr(st) == '< Empty simplex tree >'
  st.insert([[0,1,2], [0,1], [4,5], [1,4], [1,5]])
  assert all(st.n_simplices == np.array([5,6,1]))
  assert st.simplices() == [(0,),(1,),(2,),(4,),(5,),(0,1),(0,2),(1,2),(1,4),(1,5),(4,5),(0,1,2)]
  assert sorted(st.skeleton(1)) == sorted([(0,), (0, 1), (0, 2), (1,), (1, 2), (1, 4), (1, 5), (2,), (4,), (4, 5), (5,)])
  assert st.simplices(p=1) == [(0,1),(0,2),(1,2),(1,4),(1,5),(4,5)]
  assert [0,1,2] in st
  assert [1,2,3] not in st
  assert repr(st) == 'Simplex Tree with (5, 6, 1) (0, 1, 2)-simplices'
  assert st.expand(2) is None
  assert st.simplices(2) == [(0,1,2), (1,4,5)]
  assert st.cofaces([1]) == [(1,), (1, 2), (1, 4), (1, 4, 5), (1, 5), (0, 1), (0, 1, 2)]
  assert st.maximal() == [(0, 1, 2), (1, 4, 5)]
  assert st.connected_components == [1,1,1,1,1]
  assert st.vertices == [0,1,2,4,5]
  assert st.dimension == 2
  assert all(np.all(st.edges == np.array(st.simplices(p=1)), axis=0))
  assert all(np.all(st.triangles == np.array(st.simplices(p=2)), axis=0))
  assert all(st.degree() == np.array([2,4,2,2,2]))
  assert all(st.degree() == st.degree(st.vertices))
  assert st.print_tree() is None
  assert st.print_cousins() is None
  assert all(st.find([[0],[1],[3],[1,2]]) == np.array([True, True, False, True]))
  assert st.coface_roots([1,2]) == [(1,2),(0,1,2)]
  assert st.simplices() == [f for f in st]
  assert st.card() == (5,6,2)
  assert tuple(st.card(d) for d in range(st.dimension+1)) == st.card()
  assert st.reindex() is None
  assert st.vertices == [0,1,2,3,4] 

def test_insert():
  st = SimplexTree()
  simplex = np.array([[0,1,2,3,4]], dtype=np.int8)
  assert st.insert(simplex) is None 
  assert st.dimension == 4
  assert all(st.n_simplices == np.array([5,10,10,5,1]))

def test_remove():
  st = SimplexTree([[0,1,2,3,4]])
  assert st.dimension == 4
  assert st.remove([[0,1]]) is None 
  assert st.dimension == 3
  assert tuple(st.n_simplices) == (5, 9, 7, 2)
  st = SimplexTree([[0,1,2,3,4]])
  st.remove([(0,1,2)])
  assert all(st.n_simplices == np.array([5,10,9,3]))
  assert all(st.find([[0,1], [1,2], [0,1,2]]) == np.array([True, True, False]))

# def test_print_tree():
#   from io import StringIO
#   sio = StringIO()
#   st = SimplexTree([[0,1,2,3,4]])
#   st.print_tree(sio)


def test_traversals():
  st = SimplexTree([[0,1,2]])
  from io import StringIO
  sio = StringIO()
  st.traverse("dfs", lambda s: print(s, file=sio))    
  assert sio.getvalue().replace("\n", " ").strip() == '(0,) (0, 1) (0, 1, 2) (0, 2) (1,) (1, 2) (2,)'
  sio.close()
  sio = StringIO()
  st.traverse("bfs", lambda s: print(s, file=sio))    
  assert sio.getvalue().replace("\n", " ").strip() == '(0,) (1,) (2,) (0, 1) (0, 2) (1, 2) (0, 1, 2)'
  sio.close()
  sio = StringIO()
  st.traverse("p-skeleton", lambda s: print(s, file=sio), p = 1)    
  assert sio.getvalue().replace("\n", " ").strip() == '(0,) (0, 1) (0, 2) (1,) (1, 2) (2,)'
  sio.close()
  sio = StringIO()
  st.traverse("p-simplices", lambda s: print(s, file=sio), p = 1)   
  assert sio.getvalue().replace("\n", " ").strip() == '(0, 1) (0, 2) (1, 2)'
  sio.close()
  sio = StringIO()
  st.traverse("cofaces", lambda s: print(s, file=sio), sigma=[0])
  assert sio.getvalue().replace("\n", " ").strip() == '(0,) (0, 1) (0, 1, 2) (0, 2)'
  sio.close()
  st = SimplexTree([[0,1,2], [3,4,5], [5,6], [6,7], [5,7], [5,6,7],[8]])
  sio = StringIO()
  st.traverse("maximal", lambda s: print(s, file=sio))
  assert sio.getvalue().replace("\n", " ").strip() == '(0, 1, 2) (3, 4, 5) (5, 6, 7) (8,)'
  sio.close()
  st = SimplexTree([[0,1,2], [3,4,5], [5,6], [6,7], [5,7], [5,6,7],[8]])
  sio = StringIO()
  st.traverse("coface_roots", lambda s: print(s, file=sio), sigma=[1,2])
  assert sio.getvalue().replace("\n", " ").strip() == '(1, 2) (0, 1, 2)'
  sio.close()

def test_link():
  from itertools import chain
  st = SimplexTree([[0,1,2,3,4]])
  closure_star = [tuple(set(s) - set([0])) for s in st.cofaces([0]) if set(s) != set([0])]
  closure_star_test = [tuple(s) for s in st.link([0])]
  assert set(closure_star_test) == set(closure_star)

def test_properties():
  st = SimplexTree([[0,1,2]])
  assert np.allclose(st.vertices, np.array([0,1,2]))
  assert np.allclose(st.edges, np.array([[0,1],[0,2],[1,2]]))
  assert np.allclose(st.triangles, np.array([[0,1,2]]))
  assert np.allclose(st.quads, np.empty(shape=(0,4)))
  assert np.allclose(st.connected_components, [1,1,1])

def test_degenerate():
  st = SimplexTree([[0,1,2], [0,1], [4,5], [1,4], [1,5]])
  assert len(st.simplices(0)) == 5
  assert len(st.simplices(-1)) == 0

def test_expand():
  from itertools import combinations
  st = SimplexTree()
  simplex = np.array([0,1,2,3,4], dtype=np.int8)
  st.insert([(i,j) for i,j in combinations(simplex, 2)])
  st.expand(k=1)
  assert all(st.n_simplices == np.array([5,10]))
  st.expand(k=2)
  assert all(st.n_simplices == np.array([5,10,10]))
  st.expand(k=5)
  assert all(st.n_simplices == np.array([5,10,10,5,1]))
  assert st.insert([simplex]) is None 
  assert all(st.n_simplices == np.array([5,10,10,5,1]))
  assert st.dimension == 4
  st = SimplexTree()
  st.insert([[0,1], [0,2], [1,2], [4,5], [1,4], [1,5]])
  to_expand = []
  st.expand(2, lambda s: to_expand.append(s) is not None)
  assert [0,1,2] in to_expand and [1,4,5] in to_expand
  assert list(st.n_simplices) == [5,6]
  st.expand(3, lambda s: True)
  assert list(st.n_simplices) == [5,6,2]

def test_free_pair_collapse():
  from simplextree import SimplexTree
  PrincipalFaces = [[0,1],[0,3],[1,2],[1,4],[2,5],[4,7],[5,8],[7,8]]
  st = SimplexTree(PrincipalFaces)
  assert st.collapse(sigma=[0,1], tau=[0]) == False, "Collapse should fail"
  assert st.card() == (8,8), "Not supposed to collapse maximal simplices if tau has more than one coface"
  st.insert([[8,9]])
  assert len(st.cofaces([9])) == 2, "This should be a free pair"
  assert st.collapse([9], [8,9]) == False
  assert st.collapse([8,9], [9]) == True
  assert st.collapse([8,9], [9]) == False
  assert st.card() == (8,8)


  # t order, py::function f, simplex_t init = simplex_t(), const size_t k = 0
  # stree._traverse(0, lambda s: print(Simplex(s)), [], 0)
  # stree._traverse(1, lambda s: print(s), [], 0)
  # stree._traverse(2, lambda s: print(s), [1,2,3], 0)
  
  # stree._traverse(7, lambda s: yield s, [], 0) ## maximal
  ## Everything works! Now to wrap up with straverse, ltraverse, generators from orders, wrappers, ....
