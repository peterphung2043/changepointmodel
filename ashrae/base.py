from typing import NamedTuple, Tuple, Optional, TypeVar
import numpy as np 
import abc

from typing import Any, TypeVar
from nptyping import NDArray 

from sklearn.base import BaseEstimator

AnyByAnyNDArray = TypeVar('AnyByAnyNDArray', NDArray[(Any, ...), float])
NByOneNDArray = TypeVar('NByOneNDArray', NDArray[(Any, 1,), float])  # [[1.], [2.], [3.], ...]
OneDimNDArray = TypeVar('OneDimNDArray', NDArray[(Any,), float])  # [ 1., 2., 3., ...]




class IComparable(abc.ABC):  # trick to declare a Comparable type... py3 all comparability is implemented in terms of < so this is a safe descriptor

    @abc.abstractmethod
    def __lt__(self, other: Any) -> bool: ...

ComparableType = TypeVar('ComparableType', bound=IComparable)


