""" Dynamically calculates trf bounds for changepoint models where we approximately expect them to fit energy data based on 
the values of the X array.

Tuple return types correspond directly to coeffs in the method signature.Details on the structure of the bounds 
tuples can be read in the docs for `scipy.optimize.curve_fit`.
"""

from typing import Tuple, Union, Callable
import numpy as np
from ..nptypes import OneDimNDArray, NByOneNDArray

BoundTuple = Tuple[Tuple[float, ...], Tuple[float, ...]]
TwoParameterBoundary = Tuple[float, float]
ThreeParameterBoundary = Tuple[float, float, float]
FourParameterBoundary = Tuple[float, float, float, float]
FiveParameterBoundary = Tuple[float, float, float, float, float]

OpenBoundCallable = Callable[
    [Union[OneDimNDArray[np.float64], NByOneNDArray[np.float64]]], BoundTuple
]

def twop(*args, **kwargs) -> Tuple[TwoParameterBoundary, TwoParameterBoundary]:  # type: ignore
    """Energy bound for a twop (linear) model. Essentially returns a constant but we need this to
    conform to the Bounds interface.

    Returns:
        Tuple[TwoParameterBoundary, TwoParameterBoundary]: The returned bounds for scipy.optimize.curve_fit
    """
    return ((-np.inf, -np.inf), (np.inf, np.inf))


def threepc(
    X: Union[OneDimNDArray[np.float64], NByOneNDArray[np.float64]]
) -> Tuple[ThreeParameterBoundary, ThreeParameterBoundary]:
    """A threepc boundary for energy data.

    Args:
        X (Union[OneDimNDArray,NByOneNDArray]): A numpy X array. NByOneNDArray's will be squeezed internally.

    Returns:
        Tuple[ThreeParameterBoundary, ThreeParameterBoundary]: Resulting bounds tuples.
    """
    X = X.squeeze()
    return ((0, 0, X[2]), (np.inf, np.inf, max(X[-3], X[2] + 0.1)))


def threeph(
    X: Union[OneDimNDArray[np.float64], NByOneNDArray[np.float64]]
) -> Tuple[ThreeParameterBoundary, ThreeParameterBoundary]:
    """A threeph boundary for energy data.

    Args:
        X (Union[OneDimNDArray,NByOneNDArray]): A numpy X array. NByOneNDArray's will be squeezed internally.

    Returns:
        Tuple[ThreeParameterBoundary, ThreeParameterBoundary]: Resulting bounds tuples.
    """
    X = X.squeeze()
    return ((0, -np.inf, X[2]), (np.inf, 0, max(X[-3], X[2] + 0.1)))


def fourp(
    X: Union[OneDimNDArray[np.float64], NByOneNDArray[np.float64]]
) -> Tuple[FourParameterBoundary, FourParameterBoundary]:
    """A fourp boundary for energy data.

    Args:
        X (Union[OneDimNDArray,NByOneNDArray]): A numpy X array. NByOneNDArray's will be squeezed internally.

    Returns:
        Tuple[FourParameterBoundary, FourParameterBoundary]: Resulting bounds tuples.
    """
    X = X.squeeze()
    return ((0, -np.inf, 0, X[2]), (np.inf, 0, np.inf, max(X[-3], X[2] + 0.1)))


def fivep(
    X: Union[OneDimNDArray[np.float64], NByOneNDArray[np.float64]]
) -> Tuple[FiveParameterBoundary, FiveParameterBoundary]:
    """A fivep boundary for energy data.

    Args:
        X (Union[OneDimNDArray,NByOneNDArray]): A numpy X array. NByOneNDArray's will be squeezed internally.

    Returns:
        Tuple[FiveParameterBoundary, FiveParameterBoundary]: Resulting bounds tuples.
    """
    X = X.squeeze()
    return (
        (0, -np.inf, 0, X[2], X[5]),
        (np.inf, 0, np.inf, max(X[-6], X[2] + 0.1), max(X[-3], X[5] + 0.1)),
    )

