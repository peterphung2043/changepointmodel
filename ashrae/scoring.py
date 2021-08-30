""" These are wrappers around sklearn.metrics interface. They can interop with sklearn scorers or 
can be used on their own.
"""

import abc
from dataclasses import dataclass
from ashrae.nptypes import AnyByAnyNDArray, AnyByAnyNDArrayField, OneDimNDArray
from nptyping import NDArray
from typing import Any, Callable, List, NamedTuple, TypeVar, Union
from ._lib import metrics as ashraemetrics
from .estimator import EnergyChangepointEstimator

class IComparable(abc.ABC):  # trick to declare a Comparable type... py3 all comparability is implemented in terms of < so this is a safe descriptor

    @abc.abstractmethod
    def __lt__(self, other: Any) -> bool: ...

ComparableType = TypeVar('ComparableType', bound=IComparable)
SklScoreReturnType =  Union[float, AnyByAnyNDArrayField, Any]
Comparator = Callable[[IComparable, IComparable], bool]

@dataclass 
class Score(object): 
    name: str 
    value: SklScoreReturnType 
    threshold: float 
    ok: bool 


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
        return ashraemetrics.r2_score(y, y_pred, **kwargs)


class Rmse(ScoringFunction): 

    def calc(self, y: OneDimNDArray, y_pred: OneDimNDArray, **kwargs) -> SklScoreReturnType: 
        return ashraemetrics.rmse(y, y_pred, **kwargs)


class Cvrmse(ScoringFunction): 

    def calc(self, y: OneDimNDArray, y_pred: OneDimNDArray, **kwargs) -> SklScoreReturnType: 
        return ashraemetrics.cvrmse(y, y_pred, **kwargs)



class AbstractScoreEval(abc.ABC): 

    def __init__(self, scorer: ScoringFunction, threshold: float, method: Comparator): 
        self._scorer = scorer
        self._threshold = threshold
        self._method = method 
    
    @abc.abstractmethod
    def ok(self, estimator: EnergyChangepointEstimator) -> bool: ...



class ScoreEval(AbstractScoreEval): 

    def ok(self, estimator: EnergyChangepointEstimator) -> Score:
        est_val = self._scorer(estimator.y, estimator.pred_y)
        return Score(self._scorer.name, est_val, self._threshold, self._method(est_val, self._threshold))



class Scorer(object): 

    def check(self, estimator: EnergyChangepointEstimator, evals: List[ScoreEval]) -> List[Score]: 
        return [e.ok(estimator) for e in evals]