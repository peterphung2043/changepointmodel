""" Dynamically calculates trf bounds for changepoint models where we approximately expect them to fit energy data.

Tuple return types correspond directly to coeffs in the method signature of `_lib.models`. 
"""

from typing import Tuple
import numpy as np 
from ..base import OneDimNDArray

BoundTuple = Tuple[Tuple[float], Tuple[float]]


def twop() -> BoundTuple: 
    return ((0, -np.inf),(np.inf, np.inf))


def threepc(X: OneDimNDArray) -> BoundTuple:
    min_cp = X[int(len(X)/4)]
    max_cp = X[int(3 * len(X)/4)]
    return ((0,0,min_cp), (np.inf,np.inf, max_cp))


def threeph(X: OneDimNDArray) -> BoundTuple:
    min_cp = X[int(len(X)/4)]
    max_cp = X[int(3 * len(X)/4)]
    return ((0,-np.inf, min_cp), (np.inf,0, max_cp))


def fourp(X: OneDimNDArray) -> BoundTuple:
    min_cp = X[int(len(X)/4)]
    max_cp = X[int(3 * len(X)/4)]
    return ((0,-np.inf, -np.inf, min_cp), (np.inf,np.inf, np.inf, max_cp))


def fivep(X: OneDimNDArray) -> BoundTuple:
    min_cp1 = X[int((2/8) * len(X))]
    max_cp1 = X[int((3/8) * len(X))]
    min_cp2 = X[int((5/8) * len(X))]
    max_cp2 = X[int((6/8) * len(X))]
    return ((0,-np.inf, 0, min_cp1, min_cp2), (np.inf,0, np.inf, max_cp1, max_cp2))