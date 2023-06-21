import numpy as np
import numpy.typing as npt
from typing import Tuple, List
from changepointmodel.core.nptypes import OneDimNDArrayField

# Return types

SingleSlopeTStat = float
DoubleSlopeTStat = Tuple[float, float]


# Helpers
def _std_error(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
) -> float:
    sse = np.sum((y - y_pred) ** 2)
    n = np.sqrt(sse / (len(y) - 2))
    d = np.sqrt(np.sum((x - np.mean(x)) ** 2))
    return n / d  # type: ignore


def _get_array_right(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    cp: float,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    position = np.where(x >= cp)
    y_out = y[position]
    y_pred_out = y_pred[position]
    x_out = x[position]
    return x_out, y_out, y_pred_out


def _get_array_left(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    cp: float,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    position = np.where(x <= cp)
    y_out = y[position]
    y_pred_out = y_pred[position]
    x_out = x[position]
    return x_out, y_out, y_pred_out


# Main functions
def twop(
    X: OneDimNDArrayField,
    Y: OneDimNDArrayField,
    pred_y: OneDimNDArrayField,
    slope: float,
) -> SingleSlopeTStat:
    return slope / _std_error(X, Y, pred_y)


def threepc(
    X: OneDimNDArrayField,
    Y: OneDimNDArrayField,
    pred_y: OneDimNDArrayField,
    slope: float,
    changepoint: float,
) -> SingleSlopeTStat:
    _x, _y, _pred_y = _get_array_right(
        np.array(X),
        np.array(Y),
        np.array(pred_y),
        changepoint,
    )
    return slope / _std_error(_x, _y, _pred_y)


def threeph(
    X: OneDimNDArrayField,
    Y: OneDimNDArrayField,
    pred_y: OneDimNDArrayField,
    slope: float,
    changepoint: float,
) -> SingleSlopeTStat:
    _x, _y, _pred_y = _get_array_left(
        np.array(X),
        np.array(Y),
        np.array(pred_y),
        changepoint,
    )
    return slope / _std_error(_x, _y, _pred_y)


def fourp(
    X: OneDimNDArrayField,
    Y: OneDimNDArrayField,
    pred_y: OneDimNDArrayField,
    slopes: List[float],
    changepoint: float,
) -> DoubleSlopeTStat:
    xl, yl, pred_yl = _get_array_left(
        np.array(X),
        np.array(Y),
        np.array(pred_y),
        changepoint,
    )

    xr, yr, pred_yr = _get_array_right(
        np.array(X), np.array(Y), np.array(pred_y), changepoint
    )

    tl = slopes[0] / _std_error(xl, yl, pred_yl)
    tr = slopes[1] / _std_error(xr, yr, pred_yr)
    return tl, tr


def fivep(
    X: OneDimNDArrayField,
    Y: OneDimNDArrayField,
    pred_y: OneDimNDArrayField,
    slopes: List[float],
    changepoints: List[float],
) -> DoubleSlopeTStat:
    xl, yl, pred_yl = _get_array_left(
        np.array(X),
        np.array(Y),
        np.array(pred_y),
        changepoints[0],
    )

    xr, yr, pred_yr = _get_array_right(
        np.array(X),
        np.array(Y),
        np.array(pred_y),
        changepoints[1],
    )

    tl = slopes[0] / _std_error(xl, yl, pred_yl)
    tr = slopes[1] / _std_error(xr, yr, pred_yr)
    return (tl, tr)
