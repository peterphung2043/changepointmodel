import numpy as np
from .utils import _get_array_left, _get_array_right, _std_error
from typing import Tuple, List
from changepointmodel.core.nptypes import OneDimNDArrayField

# Return types

SingleSlopeTStat = float
DoubleSlopeTStat = Tuple[float, float]


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
    assert changepoint is not None
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
    assert changepoint is not None
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
    assert changepoint is not None
    assert len(slopes) >= 2
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
    assert len(changepoints) >= 2
    assert len(slopes) >= 2
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
