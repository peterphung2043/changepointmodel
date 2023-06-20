import numpy as np
from utils import _get_array_left, _get_array_right, _std_error
from typing import Tuple
from app.base import ChangepointResultContainer


def twop(result: ChangepointResultContainer) -> float:
    # What gets returned here?
    return 0


def threepc(result: ChangepointResultContainer) -> float:
    assert result.coeffs.changepoints is not None
    x, y, y_pred = _get_array_right(
        np.array(result.input_data.X),
        np.array(result.input_data.y),
        np.array(result.pred_y),
        result.coeffs.changepoints[0],
    )
    return result.coeffs.slopes[0] / _std_error(x, y, y_pred)


def threeph(result: ChangepointResultContainer) -> float:
    assert result.coeffs.changepoints is not None
    x, y, y_pred = _get_array_left(
        np.array(result.input_data.X),
        np.array(result.input_data.y),
        np.array(result.pred_y),
        result.coeffs.changepoints[0],
    )
    return result.coeffs.slopes[0] / _std_error(x, y, y_pred)


def fourp(result: ChangepointResultContainer) -> Tuple[float, float]:
    assert result.coeffs.changepoints is not None
    xl, yl, y_predl = _get_array_left(
        np.array(result.input_data.X),
        np.array(result.input_data.y),
        np.array(result.pred_y),
        result.coeffs.changepoints[0],
    )

    xr, yr, y_predr = _get_array_right(
        np.array(result.input_data.X),
        np.array(result.input_data.y),
        np.array(result.pred_y),
        result.coeffs.changepoints[0],
    )

    tl = result.coeffs.slopes[0] / _std_error(xl, yl, y_predl)
    tr = result.coeffs.slopes[1] / _std_error(xr, yr, y_predr)
    return tl, tr


def fivep(result: ChangepointResultContainer) -> Tuple[float, float]:
    assert result.coeffs.changepoints is not None
    xl, yl, y_predl = _get_array_left(
        np.array(result.input_data.X),
        np.array(result.input_data.y),
        np.array(result.pred_y),
        result.coeffs.changepoints[0],
    )

    xr, yr, y_predr = _get_array_right(
        np.array(result.input_data.X),
        np.array(result.input_data.y),
        np.array(result.pred_y),
        result.coeffs.changepoints[1],
    )

    tl = result.coeffs.slopes[0] / _std_error(xl, yl, y_predl)
    tr = result.coeffs.slopes[1] / _std_error(xr, yr, y_predr)
    return (tl, tr)
