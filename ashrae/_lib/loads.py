""" Loads calculated from changepoint models
"""

from typing import Optional, TypeVar, Any
import numpy as np
from ..base import OneDimNDArray


def _postive_sum(arr: OneDimNDArray) -> float:
    return np.sum(arr * (arr > 0))



def heatload(X: OneDimNDArray, pred_y: OneDimNDArray, slope: float, yint: float, changepoint: float=np.inf) -> Optional[float]:
    if slope is None or slope > 0:
        return None

    predicted_loads = (X < changepoint) * (pred_y - yint)
    return _postive_sum(predicted_loads)


def coolingload(X: OneDimNDArray, pred_y: OneDimNDArray, slope: float, yint: float, changepoint: float=-np.inf) -> Optional[float]:
    if  slope is None or slope < 0:
        return

    predicted_loads = (X > changepoint) * (pred_y - yint)
    return _postive_sum(predicted_loads)


def baseload(total_consumption: float, heatload: Optional[float], coolingload: Optional[float]) -> float: 
    if heatload is None and coolingload is None: 
        raise ValueError('both heatload and coolingload cannot be None')
    
    if heatload is None: 
        return total_consumption - coolingload 
    elif coolingload is None: 
        return total_consumption - heatload 

    return total_consumption - heatload - coolingload   # if this is negative should we zero it out?  

