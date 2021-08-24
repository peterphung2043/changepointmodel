""" Scoring metrics. These are implemented as wrappers around sklearn.metrics.
The scorer module forwards calls to these implementations.
"""
from typing import Any, Union
import numpy as np
from sklearn import metrics as sklmetrics

from ..base import OneDimNDArray

#XXX use nptyping here.

def r2_score(y: OneDimNDArray, y_pred: OneDimNDArray, **kwargs) -> Union[float, OneDimNDArray[float]]: 
    return sklmetrics.r2_score(y, y_pred, **kwargs)


def rmse(y: OneDimNDArray, y_pred: OneDimNDArray, **kwargs) -> Union[float, OneDimNDArray[float]]: 
    return sklmetrics.mean_squared_error(y, y_pred, squared=False, **kwargs)


def cvrmse(y: OneDimNDArray, y_pred: OneDimNDArray, **kwargs) -> Union[float, OneDimNDArray[float]]:
    rmse = sklmetrics.mean_squared_error(y, y_pred, squared=False, **kwargs)
    return rmse / np.mean(y)

