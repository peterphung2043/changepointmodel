"""Dynamically calculates trf initial guesses for changepoint models where we approximately expect them to fit energy data based on 
the values of the X array.

Array return types correspond directly to coeffs in the method signature. Details on the structure of the bounds 
array can be read in the docs for `scipy.optimize.curve_fit`.
"""

from typing import Tuple, Union
from collections.abc import Callable
import numpy as np
from changepointmodel.core.nptypes import OneDimNDArray, NByOneNDArray


InitialGuessTuple = Tuple[float, ...]

TwoParameterInitialGuess = Tuple[float, float]
ThreeParameterInitialGuess = Tuple[float, float, float]
FourParameterInitialGuess = Tuple[float, float, float, float]
FiveParameterInitialGuess = Tuple[float, float, float, float, float]

OpenInitialGuessCallable = Callable[
    [
        Union[OneDimNDArray[np.float64], NByOneNDArray[np.float64]],
        OneDimNDArray[np.float64],
    ],
    InitialGuessTuple,
]


def twop(
    X: Union[OneDimNDArray[np.float64], NByOneNDArray[np.float64]],
    y: OneDimNDArray[np.float64],
) -> TwoParameterInitialGuess:
    """Energy initial guess for a twop (linear) model. Essentially returns a list of floats
    corresponding to the initial guesses

    Args:
      X (Union[OneDimNDArray,NByOneNDArray]): A numpy X array. NByOneNDArray's will be squeezed internally.
      y OneDimNDArray[np.float64]: A numpy y array.

    Returns:
        List[float]: The returned initial guesses for scipy.optimize.curve_fit
    """
    X = X.squeeze()
    return (
        y[0],
        (y[-1] - y[0]) / (X[-1] - X[0]),
    )  # `.item` enforces that the value returned is a scalar. This is because `X`` is a multidimensional array. See: https://stackoverflow.com/questions/30311172/convert-list-or-numpy-array-of-single-element-to-float-in-python


def threepc(
    X: Union[OneDimNDArray[np.float64], NByOneNDArray[np.float64]],
    y: OneDimNDArray[np.float64],
) -> ThreeParameterInitialGuess:
    """A threepc initial guess for energy data.

    Args:
        X (Union[OneDimNDArray,NByOneNDArray]): A numpy X array. NByOneNDArray's will be squeezed internally.
        y OneDimNDArray[np.float64]: A numpy y array.

    Returns:
        List[float]: The returned initial guesses for scipy.optimize.curve_fit
    """
    X = X.squeeze()
    return (y[0], ((max(y) - min(y)) / (X[-1] - np.median(X))), np.median(X))


def threeph(
    X: Union[OneDimNDArray[np.float64], NByOneNDArray[np.float64]],
    y: OneDimNDArray[np.float64],
) -> ThreeParameterInitialGuess:
    """A threeph initial guess for energy data.

    Args:
        X (Union[OneDimNDArray,NByOneNDArray]): A numpy X array. NByOneNDArray's will be squeezed internally.
        y OneDimNDArray[np.float64]: A numpy y array.

    Returns:
        List[float]: The returned initial guesses for scipy.optimize.curve_fit
    """
    X = X.squeeze()
    return (y[-1], ((min(y) - max(y)) / (np.median(X) - X[0])), np.median(X))


def fourp(
    X: Union[OneDimNDArray[np.float64], NByOneNDArray[np.float64]],
    y: OneDimNDArray[np.float64],
) -> FourParameterInitialGuess:
    """A fourp initial guess for energy data.

    Args:
        X (Union[OneDimNDArray,NByOneNDArray]): A numpy X array. NByOneNDArray's will be squeezed internally.
        y OneDimNDArray[np.float64]: A numpy y array.

    Returns:
        List[float]: The returned initial guesses for scipy.optimize.curve_fit
    """
    X = X.squeeze()
    return (
        min(y),
        ((y[0] - min(y)) / (X[0] - X[np.where(y == min(y))[0][0]])),
        ((y[-1] - min(y)) / (X[-1] - X[np.where(y == min(y))[0][0]])),
        X[np.where(y == min(y))[0][0]],
    )


def fivep(
    X: Union[OneDimNDArray[np.float64], NByOneNDArray[np.float64]],
    y: OneDimNDArray[np.float64],
) -> FiveParameterInitialGuess:
    """A fivep initial guess for energy data.

    Args:
        X (Union[OneDimNDArray,NByOneNDArray]): A numpy X array. NByOneNDArray's will be squeezed internally.
        y OneDimNDArray[np.float64]: A numpy y array.

    Returns:
        List[float]: The returned initial guesses for scipy.optimize.curve_fit
    """
    X = X.squeeze()
    return (
        y[0],
        ((min(y) - y[0]) / (np.median(X) - X[0])),
        ((max(y) - min(y)) / (X[-1] - np.median(X))),
        np.median(X) - ((X[-1] - X[0]) * 0.25),
        np.median(X) + ((X[-1] - X[0]) * 0.25),
    )
