import numpy as np

### NOTE to johnny
# ( x < break_point) creates an array of 1 and 0, which basically serve as if else statement

def _calc_true_load(total_predicted_load: float, total_predicted_consumption: float, total_consumption: float) -> float:
    """ As true as it gets... the deepest loads there is. Used by cooling and heating
    load calcs to reconcile predicted load and consumption values with total_consumption
    """
    return (total_predicted_load / total_predicted_consumption) * total_consumption

def calc_heat_load(x: np.array, y_predicted: np.array, slope: float, y_intercept: float, break_point = np.inf) -> float:
    """calculate heating load"""
    if slope is None or slope > 0:
        return None

    predicted_loads = (x < break_point) * (y_predicted - y_intercept)
    return postive_sum(predicted_loads)

def postive_sum(arr: np.array) -> float:
    """calcuate total of array for positive values only"""
    return np.sum(arr * (arr > 0))

def calc_cool_load(x: np.array, y_predicted: np.array, slope: float, y_intercept: float, break_point =  -np.inf) -> float:
    """calculate cooling load"""
    if  slope is None or slope < 0:
        return

    predicted_loads = (x > break_point) * (y_predicted - y_intercept)
    return postive_sum(predicted_loads)