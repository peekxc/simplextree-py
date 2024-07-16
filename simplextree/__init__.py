# coding: utf-8
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import importlib.metadata
__version__ = importlib.metadata.version("simplextree")

from .SimplexTree import SimplexTree
from .UnionFind import UnionFind
# from combinatorial import *
# from predicates import *