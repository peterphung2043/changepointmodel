import numpy as np
import numpy.typing as npt
from typing import Tuple
from app.base import ChangepointResultContainer
from changepointmodel.core.pmodels import EnergyParameterModelT, ParamaterModelCallableT


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


def twop(result: ChangepointResultContainer) -> float:
    # What gets returned here?
    ...


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
