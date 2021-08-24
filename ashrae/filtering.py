""" Custom filtering logic 

XXX Too tightly coupled to EnergyParameterModel class? 
If not that what else? 
"""

import abc
from typing import Any, Callable, List, NamedTuple, TypeVar
from .base import AbstractEnergyParameterModel, IComparable


Comparator = TypeVar('Comparator', Callable[[IComparable, IComparable], bool])

# the idea here is that we can potentially filter on any of the model's attributes if we want not just scores 
class ComparableFilter(abc.ABC): 

    def __init__(self, key: str, value: IComparable, method: Comparator): 
        self._key = key
        self._value = value 
        self._method = method 
    
    @abc.abstractmethod
    def ok(self, model: AbstractEnergyParameterModel) -> bool: ...


class ComparableFilters(object): 
    """ container class for a bunch of comparable filters. Allows to check 
    which filters pass in one call.
    """
    def __init__(self, filters: List[ComparableFilter]): 
        self._filters = filters  

    def check(self, model: AbstractEnergyParameterModel) -> List[bool]: 
        return [f.ok(model) for f in self._filters]



class EnergyParameterModelScoreFilter(ComparableFilter): 
    """ Looks up a score 
    """

    def ok(self, model: AbstractEnergyParameterModel) -> bool:
        if model.score is None: 
            raise ValueError(f'scores are not available on this model')
        obj = model.score.get(self._key)  # assumes attr is dictionary... 
        if obj is None: 
            raise ValueError(f'score object does not contain named score: {self._key}')
        return self._method(self._value, obj)





