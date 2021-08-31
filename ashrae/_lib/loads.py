""" Loads calculated from changepoint models
"""

from typing import Optional, TypeVar, Any
import numpy as np
from ..nptypes import OneDimNDArray

# NOTE moved logic handling null up one level but might get rid of that completely

# XXX possibly remove this... possibly use absolute value here .. tynabot do research...
def _positive_sum(arr: OneDimNDArray) -> float:
    return np.sum(arr * (arr > 0))


def _heating_predicted_load(X: OneDimNDArray, pred_y: OneDimNDArray, yint: float, changepoint: float=-np.inf): 
    return (X < changepoint) * (pred_y - yint)


def _cooling_predicted_load(X: OneDimNDArray, pred_y: OneDimNDArray, yint: float, changepoint: float=-np.inf): 
    return (X > changepoint) * (pred_y - yint)


def heatload(X: OneDimNDArray, pred_y: OneDimNDArray, slope: float, yint: float, changepoint: Optional[float]=np.inf) -> float:
    
    if slope > 0: 
        return 0  
    
    predicted_loads = _heating_predicted_load(X, pred_y, yint, changepoint)
    return _positive_sum(predicted_loads)
    

def coolingload(X: OneDimNDArray, pred_y: OneDimNDArray, slope: float, yint: float, changepoint: float=-np.inf) -> float:

    if slope < 0:
        return 0 

    predicted_loads = _cooling_predicted_load(X, pred_y, yint, changepoint)
    return _positive_sum(predicted_loads)


def baseload(total_consumption: float, heatload: float, coolingload: float) -> float: 
    return total_consumption - heatload - coolingload   # if this is negative should we zero it out?  

