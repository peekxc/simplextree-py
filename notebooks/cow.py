# %% Imports 
import numpy as np
import pywavefront
from scipy.spatial.distance import pdist, cdist, squareform
from combin import comb_to_rank, rank_to_comb
from simplextree import SimplexTree

# %% 
cow = pywavefront.Wavefront("/Users/mpiekenbrock/Downloads/cow.obj", parse=True, collect_faces=True)
T = np.array(cow.mesh_list[0].faces, dtype=np.int32)
X = np.array(cow.vertices)

def face_normals(F: np.ndarray, X: np.ndarray) -> np.ndarray:
  """Computes unit normal vectors for each face (i,j,k) in `F`"""
  assert F.ndim == 2 and F.shape[1] == 3, "Face array must have 3 indices"
  N = np.cross(X[F[:,0]] - X[F[:,1]], X[F[:,1]] - X[F[:,2]])
  N = N / np.linalg.norm(N, axis=1, keepdims=True)
  return N

def face_planes(F: np.ndarray, X: np.ndarray, N: np.ndarray, validate: bool = False):
  """Computes the plane(s) p = [a,b,c,d] defined by the equation ax + by + cz + d = 0 for supplied faces."""
  f_ind = F[:,0]
  ds = np.sum(-N[f_ind] * X[f_ind], axis=1)[:, None]
  P = np.hstack((N[f_ind], ds))
  if validate: 
    ## Plane checking 
    norma_ = np.linalg.norm(P[:,:3], axis=1)
    assert np.allclose(norma_, 1.0, 1e-6), "Planes should be unit-norm"
    assert np.allclose(np.sum(P * np.c_[X[F[:,1]], np.ones(len(F))], axis=1), 0.0), "all points don't lie on plane"
  return P

def face_areas(F: np.ndarray, X: np.ndarray):
  """Compute the area of each face"""
  side_diff = np.diff(X[F], axis=1)
  dap = np.cross(side_diff[:,0], side_diff[:,1]) # "directed area product"
  return (np.sum(dap ** 2, axis=1) ** .5) / 2 

# %% Mesh Decimator 
class MeshDecimator():
  """
  Fields: 
    X = (n x 3) matrix of vertex coordinates 
    F = (T x 3) matrix of face (triangle) indices 
    S = mesh, represented as a simplex tree 
    VF = vertex-face adjacency matrix 
    N = (T x 4) matrix of face unit normals 
    P = (T x 4) matrix of face plane equations 
  """
  def __init__(self, X: np.ndarray, F: np.ndarray):
    self.X = X
    self.F = F
    self.S = SimplexTree(F)
    self.VF = np.zeros((len(X), len(F)), dtype=bool) # vertex-triangle adjacencies
    for ii, t in enumerate(F):
      self.VF[t, ii] = True
    self.N = face_normals(self.F, self.X) # normals 
    self.P = face_planes(self.F, self.X, self.N)
    self.F_ind = { tuple(sorted(f)) : i for i,f in enumerate(self.F) }

  def quadric(self, v: int):
    """Computes the sum of fundamental error quadrics of a vertex across a set of its adjacent planes"""
    assert v >= 0 and v < len(self.X), "Invalid vertex id"
    f_idx = np.flatnonzero(self.VF[v])
    return np.einsum('ij,ik->jk', self.P[f_idx], self.P[f_idx])

  def contraction_target(self, valid_pair: tuple) -> tuple:
    """Computes the contraction target vertex + its minimal error for a given valid pair"""
    ## Based on section 4 of "Surface Simplification Using Quadric Error Metrics"
    assert len(valid_pair) == 2, "Must be a index pair of vertex ids"
    i, j = valid_pair
    vi, vj = self.X[i], self.X[j]
    Qh = self.quadric(i) + self.quadric(j)
    A = Qh.copy()
    A[3] = [0,0,0,1]
    if np.linalg.det(A) == 0.0:
      from scipy.optimize import brent
      # vi, vj = np.insert(X[i],3,1)[:,np.newaxis], np.insert(X[j],3,1)[:,np.newaxis]
      q_err = lambda a: ((1.0 - a) * vi + a * vj).T @ Qh @ ((1.0 - a) * vi + a * vj)
      a_opt = brent(func=q_err, brack=(0.0, 1.0), full_output=False).item()
      v_bar = np.ravel((1.0 - a_opt) * vi + a_opt * vj)
    else:
      v_bar = np.linalg.solve(A, [0,0,0,1])
    error = (v_bar[:,np.newaxis].T @ Qh @ v_bar[:,np.newaxis]).item()
    return v_bar, error

  def contract(self, i: int, j: int, p: np.ndarray):
    """Contracts the vertex pair 'i' to 'j', removing 'j' from the mesh and replacing vi with 'p'."""
    # assert len(p) == 3, "point must be 3 coordinates"

    ## Save the info related to the contraction 
    incident_faces = np.array([self.F_ind[c] for c in self.S.cofaces([i,j]) if len(c) == 3])
    ij_contracted = self.S.contract((i,j))
    if not ij_contracted: 
      return 
    
    ## The contraction should remove these faces 
    # assert np.all([self.F[c] not in self.S for c in incident_faces]), "Incident faces still there"

    ## Update F_ind map
    # self.remove(incident_faces) # remove incident edges 

    ## i contracted to j <=> remove vertex j from point cloud 
    self.X[i] = p[:3]
    self.X[j] = [0,0,0]

    ## Remove incident triangles from i, all triangles from j
    self.VF[j,:] = False
    self.VF[i,incident_faces] = False

    ## Update adjacency to reflect i's adjacency 
    f_ind = np.array([self.F_ind[c] for c in self.S.cofaces([i]) if len(c) == 3 and c in self.F_ind])
    self.VF[i,f_ind] = True

    ## Update plane information for all incident faces 
    self.P[f_ind] = face_planes(self.F[f_ind], self.X, self.N)

  def valid_pairs(self, radius: float = 0.50) -> np.ndarray:
    """Generates a set of valid pairs"""
    dX = pdist(self.X)
    valid_ind = np.zeros(len(dX), dtype=bool)
    E = np.array(self.S.simplices(1))
    valid_ind[comb_to_rank(E, k=2, n=len(X), order='lex')] = True # in the mesh 
    valid_ind[dX <= 2*radius] = True 
    valid_pairs = rank_to_comb(np.flatnonzero(valid_ind), order='lex', n=len(self.X), k=2)
    return valid_pairs
  
  def assemble_contractions(self, radius: float = 0.50):
    import heapq
    valid_pairs = self.valid_pairs(radius)
    self.c_pairs = [(self.contraction_target(pair)[1], tuple(pair)) for pair in valid_pairs]
    heapq.heapify(self.c_pairs)

# %% 
import heapq
M = MeshDecimator(X, T)
# M.assemble_contractions(radius=0.10)
# heapq.nsmallest(10, M.c_pairs)

## Initialize set of valid pairs 
valid_pairs = M.valid_pairs(0.10)

## Computes the contraction pairs + their error costs
contract_pairs = [(M.contraction_target(pair)[1], tuple(pair)) for pair in valid_pairs]
heapq.heapify(contract_pairs) 

p, p_err = M.contraction_target(contract_pairs[0][1])



v_bar, err = M.contraction_target([1936, 1937])
M.contract(1936, 1937, v_bar)
i,j = 1936, 1937


# ## Calculate K & Q according to eq. (2)
# ## Einsum equivalent to sum of outer products above
# ## https://stackoverflow.com/questions/17437523/python-fast-way-to-sum-outer-products
# def quadric(v: int, VF: np.ndarray):
#   # f_idx = [T_ind[c] for c in S.cofaces([0]) if len(c) == 3]
#   f_idx = np.flatnonzero(VF[v])
#   Q = np.einsum('ij,ik->jk', PLANES[f_idx], PLANES[f_idx])
#   return Q
# QV = [quadric(v, VF) for v in range(len(X))]


def on_same_plane(v_ids, PLANES) -> bool:
  return np.isclose(np.sum(np.diff(PLANES[v_ids], axis=0)), 0.0)

# %% Obtain the set of contractible valid pairs 
import heapq
cand_pairs = contraction(valid_pairs, X)
heapq.heapify(cand_pairs)

## Contract the minimum cost pair 
c_err, (ci,cj), v_bar = heapq.heappop(cand_pairs)
X[ci,:] = v_bar[:3],
S.contract([ci, cj])

## Remove vertex j from adjacencies
X[cj,:] = [0,0,0]
VF[cj,:] = False 
VF[:,cj] = False 

## Update the set of candidate pairs post - contraction
pairs_to_update = set([c for c in S.cofaces([ci]) if len(c) == 2])
cand_pairs[:] = [c_pair for c_pair in cand_pairs if c_pair[1] not in pairs_to_update]
updated_pairs = contraction(pairs_to_update, X)
heapq.heapify(updated_pairs)
cand_pairs = list(heapq.merge(cand_pairs, updated_pairs))


np.sum(X.sum(axis=1) == 0)


# %% Visualize 
import open3d as o3
import open3d.core as o3c
cow_mesh = o3.io.read_triangle_mesh("/Users/mpiekenbrock/Downloads/cow.obj")
cow_mesh.compute_vertex_normals()
o3.visualization.draw(cow_mesh, raw_mode=False)

from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
T_reduced = S.triangles
cow_mesh = o3.geometry.TriangleMesh(Vector3dVector(X), Vector3iVector(T_reduced))

o3.visualization.draw(cow_mesh, raw_mode=False)

cow_mesh.vertices

# for contractible_pair in cand_pairs:
#   err, v_pair, v_bar = contractible_pair
#   if v_pair in pairs_to_update:
#     cand_pairs
#     heapq.heapreplace()


heapq.merge()

import open3d


X[1936,:] = 0


heapq.nsmallest(10, cand_pairs)


# Kp = np.zeros((4,4))
# for p in PLANES[f_idx]:
#   Kp += p[:,np.newaxis] @ p[:,np.newaxis].T

# [p[:,np.newaxis] @ p[:,np.newaxis].T for p in PLANES[f_idx]]
