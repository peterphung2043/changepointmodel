import numpy as np
import numpy.typing as npt


# Used in tstat
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
