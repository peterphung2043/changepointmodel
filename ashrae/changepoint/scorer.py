"""
"""

import abc
import numpy as np
from nptyping import NDArray
from typing import Dict, List, NamedTuple, Union
from sklearn import metrics
from ._lib import metrics

Score = NamedTuple('Score', [('name', str), ('score', Union[float, NDArray[float]])])


class ScoringFunction(abc.ABC):
    """ Interface for a scoring function. User should override calc with a numpy/Cython based 
    function that takes (y, pred_y) and returns a scalar. A name should be provided so the object 
    deserializations. 
    """ 

    @abc.abstractmethod 
    def calc(self, y: np.array, y_pred: np.array, **kwargs) -> float: 
        pass

    def __call__(self, *args, **kwargs): 
        return (self.__class__.__name__.lower(), self.calc(*args, **kwargs), )



class Scorer(object): 
    """ This should be passed to a fitted model's score function
    """

    def __init__(self, score_functions: List[ScoringFunction]):
        self._score_functions = score_functions
        

    def score(self, y: np.array, y_pred: np.array, **kwargs) -> List[Score]: 
        return [s(y, y_pred, **kwargs) for s in self._score_function]




## TODO model r2, mse, rmse, cvrmse from the ashrae/bema calcs


class Rsquared(ScoringFunction): 

    def calc(self, y: np.array, y_pred: np.array, **kwargs) -> float: 
        return metrics.r2_score(y, y_pred, **kwargs)


class Rmse(ScoringFunction): 

    def calc(self, y: np.array, y_pred: np.array, **kwargs) -> float: 
        return metrics.rmse(y, y_pred, **kwargs)


class Cvrmse(ScoringFunction): 

    def calc(self, y: NDArray[float], y_pred: NDArray[float], **kwargs) -> float: 
        return metrics.cvrmse(y, y_pred, **kwargs)

