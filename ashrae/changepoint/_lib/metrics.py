""" Scoring metrics. These are implemented as wrappers around sklearn.metrics.
The scorer module forwards calls to these implementations.
"""
from typing import Any, Union
from nptyping import NDArray
import numpy as np
from sklearn import metrics as sklmetrics


def r2_score(y: np.array, y_pred: np.array, **kwargs) -> Union[float, NDArray[float]]: 
    return sklmetrics.r2_score(y, y_pred, **kwargs)


def rmse(y: np.array, y_pred: np.array, **kwargs) -> Union[float, NDArray[float]]: 
    return sklmetrics.mean_squared_error(y, y_pred, squared=False, **kwargs)


def cvrmse(y: np.array, y_pred: np.array, **kwargs) -> Union[float, NDArray[float]]:
    rmse = sklmetrics.mean_squared_error(y, y_pred, squared=False, **kwargs)
    return rmse / np.mean(y)

