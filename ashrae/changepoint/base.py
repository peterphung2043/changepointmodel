from typing import NamedTuple, Tuple, Optional
import numpy as np 
import abc

# Return types for abstract classes
Load = NamedTuple('Load', [('baseload', float), ('heating', float), ('cooling', float)])
Bound = NamedTuple('Bound', [('lower', Tuple[float]), ('upper', Tuple[float])])  # tuple size changes based on n params
Tstat = NamedTuple('Tstat', [('slopes', Tuple[float])])  # There is a tuple of floats for each coefficient 


class ILoad(abc.ABC):

    @abc.abstractmethod
    def load(self, X: np.array, y: np.array, y_pred: np.array, total_y: float, total_y_pred: float, *coeffs) -> Optional[Load]: # see above 
        return


class ITStat(abc.ABC): 


    @abc.abstractmethod
    def tstat(self, X: np.array, y: np.array, y_pred: np.array, *coeffs) -> Optional[Tstat]: 
        # standard error needed for this one
        return


class ArgSortable(object): 

    def sort(self, X: np.array, y: np.array) -> Tuple[np.array]: 
        if X.ndim != 1:
            order = np.argsort(X.squeeze())
        else:
            order = np.argsort(X)  # this should work for 1d arrays
        return X[order], y[order]



class IModelFunction(abc.ABC): 


    @abc.abstractstaticmethod
    def f(X: np.array, *coeffs) -> np.array: 
        # the function we wish to model. Must return  y array for curve_fit
        # NOTE that data possibly needs to be reshaped here into long(array) form... this is because sklearn interface only accepts [[],...] for X
        pass


    @abc.abstractstaticmethod
    def bounds(X: np.array) -> Optional[Bound]: 
        # we model dependent bounds calculation for curve_fit
        return


# an energy parameter 
class AbstractEnergyParameterModel(IModelFunction, ILoad, ITStat, ArgSortable): 
    pass