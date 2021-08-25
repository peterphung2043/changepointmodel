from typing import NamedTuple, Tuple, Optional, TypeVar
import numpy as np 
import abc

from typing import Any, TypeVar
from nptyping import NDArray 

from sklearn.base import BaseEstimator

AnyByAnyNDArray = TypeVar('AnyByAnyNDArray', NDArray[(Any, ...), float])
NByOneNDArray = TypeVar('NByOneNDArray', NDArray[(Any, 1,), float])  # [[1.], [2.], [3.], ...]
OneDimNDArray = TypeVar('OneDimNDArray', NDArray[(Any,), float])  # [ 1., 2., 3., ...]


# Return types for abstract classes
Load = NamedTuple('Load', [    # holds calculated loads from methodogy and places slopes into energy related context
    ('baseload', float), 
    ('heating', float), 
    ('cooling', float), 
    ('heating_sensitivity', float), 
    ('cooling_sensitivity', float)])
Bound = NamedTuple('Bound', [('lower', Tuple[float]), ('upper', Tuple[float])])  # tuple size changes based on n params
Tstat = NamedTuple('Tstat', [('slopes', Tuple[float])])  # There is a tuple of floats for each coefficient 


class ILoad(abc.ABC):

    @abc.abstractmethod
    def load(self, 
        X: NByOneNDArray, 
        y: NByOneNDArray, 
        pred_y: NByOneNDArray, 
        total_y: float, 
        *coeffs) -> Optional[Load]: ... # see above 


class IComparable(abc.ABC):  # trick to declare a Comparable type... py3 all comparability is implemented in terms of < so this is a safe descriptor

    @abc.abstractmethod
    def __lt__(self, other: Any) -> bool: ...

ComparableType = TypeVar('ComparableType', bound=IComparable)



class IModelFunction(abc.ABC): 

    _name : str = ""

    def name(self):
        assert self._name != "", 'Must provide a model name' 
        return self._name


    @abc.abstractstaticmethod
    def f(X: NByOneNDArray, *coeffs) -> OneDimNDArray: ... 
        # the function we wish to model. Must return  y array for curve_fit
        # NOTE that data possibly needs to be reshaped here into long(array) form... this is because sklearn interface only accepts [[],...] for X


    @abc.abstractstaticmethod
    def bounds(X: np.array) -> Optional[Bound]: ...
        # we model dependent bounds calculation for curve_fit



# an energy parameter 
class AbstractEnergyParameterModel(IModelFunction, ILoad): 
    pass