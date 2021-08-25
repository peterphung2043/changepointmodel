""" These are wrappers around sklearn.metrics interface. They can interop with sklearn scorers or 
can be used on their own.
"""

import abc

from curvefit_estimator import estimator
from ashrae.base import IComparable, OneDimNDArray
import numpy as np
from nptyping import NDArray
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, TypeVar, Union
from sklearn import metrics
from ._lib import metrics
from .estimator import EnergyChangepointEstimator


Score = NamedTuple('Score', [('name', str), ('score', Union[float, NDArray[float]])])
SklScoreReturnType =  Union[float, NDArray[Any, ...], Any]


class ScoringFunction(abc.ABC):
    """ Interface for a scoring function. User should override calc with a numpy/Cython based 
    function that takes (y, pred_y) and returns a scalar. A name should be provided so the object 
    deserializations. 
    """ 

    @property 
    def name(self) -> str: 
        """Expose the name of the class conveniently for creating collections of scores.

        Returns:
            str: [description]
        """
        return self.__class__.__name__.lower()


    @abc.abstractmethod 
    def calc(self, y: OneDimNDArray, y_pred: OneDimNDArray, **kwargs) -> SklScoreReturnType: ...
        

    def __call__(self, *args, **kwargs): 
        return self.calc(*args, **kwargs)


## TODO model r2, mse, rmse, cvrmse from the ashrae/bema calcs


class R2(ScoringFunction): 

    def calc(self, y: OneDimNDArray, y_pred: OneDimNDArray, **kwargs) -> SklScoreReturnType:
        return metrics.r2_score(y, y_pred, **kwargs)


class Rmse(ScoringFunction): 

    def calc(self, y: OneDimNDArray, y_pred: OneDimNDArray, **kwargs) -> SklScoreReturnType: 
        return metrics.rmse(y, y_pred, **kwargs)


class Cvrmse(ScoringFunction): 

    def calc(self, y: OneDimNDArray, y_pred: OneDimNDArray, **kwargs) -> SklScoreReturnType: 
        return metrics.cvrmse(y, y_pred, **kwargs)



Comparator = TypeVar('Comparator', Callable[[IComparable, IComparable], bool])


class AbstractScoreFilter(abc.ABC): 

    def __init__(self, scorer: ScoringFunction, threshold: float, method: Comparator): 
        self._scorer = scorer
        self._threshold = threshold
        self._method = method 
    
    @abc.abstractmethod
    def ok(self, estimator: EnergyChangepointEstimator) -> bool: ...



class ScoreFilter(AbstractScoreFilter): 

    def ok(self, estimator: EnergyChangepointEstimator) -> Tuple[str, bool]:
        est_val = self._scorer(estimator.y, estimator.pred_y)
        return (self._scorer.name, self._method(est_val, self._threshold), )



class ScoreFilters(object): 

    def check(self, estimator: EnergyChangepointEstimator, filters: List[ScoreFilter]) -> List[Tuple[str, bool]]: 
        return [f.ok(estimator) for f in filters]