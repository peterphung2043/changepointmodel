""" Derive the loads from the estimator
"""
import abc
import numpy as np
from ashrae.nptypes import NByOneNDArray, OneDimNDArray, OneDimNDArrayField
from . import utils as ashraeutils
from typing import List, NamedTuple, Optional, Tuple
from .estimator import EnergyChangepointEstimator
from .scoring import ScoringFunction

from .lib import loads as _loads 

from .parameter_models import EnergyParameterModel, EnergyParameterModelCoefficients, ParameterModelFunction
from dataclasses import dataclass 

@dataclass 
class Load(object): 
    base: float 
    heating: float 
    cooling: float 
    
# handles linear and changepoint based loads


class ISensitivityLoad(abc.ABC): 

    @abc.abstractmethod 
    def calc(self, X: OneDimNDArray, pred_y: OneDimNDArray, slope: float, yint: float, changepoint: Optional[float]=None) -> float: ...

    def __call__(self, *args, **kwargs) -> float: 
        return self.calc(*args, **kwargs)

class IBaseload(abc.ABC): 

    @abc.abstractmethod 
    def calc(self, total_consumption: float, *loads: float) -> float: ...

    def __call__(self, *args, **kwargs) -> float: 
        return self.calc(*args, **kwargs)


class HeatingLoad(ISensitivityLoad): 

    def calc(self, X: OneDimNDArray, pred_y: OneDimNDArray, slope: float, yint: float, changepoint: Optional[float]=None) -> float: 

        if slope > 0: # pos slope no heat
            return 0

        if changepoint is None: # no changepoint then set to inf  (handles linear model loads)
            changepoint = np.inf 

        return _loads.heatload(X, pred_y, yint, changepoint)



class CoolingLoad(ISensitivityLoad): 

    def calc(self, X: OneDimNDArray, pred_y: OneDimNDArray, slope: float, yint: float, changepoint: Optional[float]=None) -> float: 
        if slope < 0: # neg slope no cool
            return 0
        
        if changepoint is None:  # no cp then set to -inf (handles linear model loads)
            changepoint = -np.inf
        
        return _loads.coolingload(X, pred_y, yint, changepoint)


class Baseload(IBaseload): 

    def calc(self, total_consumption: float, *loads: float) -> float:
        return _loads.baseload(total_consumption, *loads)  # This just subtracts all loads from this total y_pred ... subject to change @tynabot



class AbstractLoadHandler(abc.ABC): 

    def __init__(self, 
        model: EnergyParameterModel, 
        cooling: ISensitivityLoad, 
        heating: ISensitivityLoad, 
        base: Optional[IBaseload]=None):

        self._model = model 
        self._cooling = cooling 
        self._heating = heating 
        self._base = base if base is not None else Baseload()

    @property 
    def model(self): 
        return self._model  # for introspection later.

    def _initial(self, 
        pred_y: OneDimNDArray, 
        coeffs: EnergyParameterModelCoefficients) -> Tuple[float, float]: 

        yint = self._model.yint(coeffs)  # all models need this so placing here...
        total_pred_y = np.sum(pred_y)
        return yint, total_pred_y 

    @abc.abstractmethod
    def run(self, 
        X: OneDimNDArray, 
        pred_y: OneDimNDArray, 
        coeffs: EnergyParameterModelCoefficients) -> Load: ...



class TwoParameterLoadHandler(AbstractLoadHandler): 

    def run(self, 
        X: NByOneNDArray, 
        pred_y: OneDimNDArray, 
        coeffs: EnergyParameterModelCoefficients) -> Load: 

        yint, total_pred_y = self._initial(pred_y, coeffs)

        slope = self._model.slope(coeffs)
        heating = self._heating(X, pred_y, slope, yint)
        cooling = self._cooling(X, pred_y, slope, yint)
        base = self._base(total_pred_y, cooling, heating)
        
        return Load(base=base, heating=heating, cooling=cooling)

    

class ThreeParameterLoadHandler(AbstractLoadHandler): 
    
    def run(self, 
        X: NByOneNDArray, 
        pred_y: OneDimNDArray, 
        coeffs: EnergyParameterModelCoefficients) -> Load: 

        yint, total_pred_y = self._initial(pred_y, coeffs)

        slope = self._model.slope(coeffs)
        cp = self._model.changepoint(coeffs)

        heating = self._heating(X, pred_y, slope, yint, cp)
        cooling = self._cooling(X, pred_y, slope, yint, cp)
        base = self._base(total_pred_y, cooling, heating)

        return Load(base, heating, cooling)



class FourParameterLoadHandler(AbstractLoadHandler): 


    def run(self, 
        X: NByOneNDArray, 
        pred_y: OneDimNDArray, 
        coeffs: EnergyParameterModelCoefficients) -> Load: 

        yint, total_pred_y = self._initial(pred_y, coeffs)

        ls = self._model.left_slope(coeffs)
        rs = self._model.right_slope(coeffs)
        cp = self._model.changepoint(coeffs)

        heating = self._heating(X, pred_y, ls, yint, cp)
        cooling = self._cooling(X, pred_y, rs, yint, cp)
        base = self._base(total_pred_y, cooling, heating)

        return Load(base, heating, cooling)


class FiveParameterLoadHandler(AbstractLoadHandler): 


    def run(self, 
        X: NByOneNDArray, 
        pred_y: OneDimNDArray, 
        coeffs: EnergyParameterModelCoefficients) -> Load: 

        yint, total_pred_y = self._initial(pred_y, coeffs)

        yint = self._model.yint(coeffs)
        ls = self._model.left_slope(coeffs)
        rs = self._model.right_slope(coeffs)
        lcp = self._model.left_changepoint(coeffs)
        rcp = self._model.right_changepoint(coeffs)

        cooling = self._cooling(X, pred_y, rs, yint, rcp)
        heating = self._heating(X, pred_y, ls, yint, lcp)
        base = self._base(total_pred_y, cooling, heating)

        return Load(base, heating, cooling)



class EnergyChangepointLoadsAggregator(object): 

    def __init__(self, handler: AbstractLoadHandler): 
        self._handler = handler 


    def aggregate(self, estimator: EnergyChangepointEstimator) -> Load: 

        # check that estimator model type matches the handler's model interface or else we'll have an issue 
        # coeff parsing needs to match...
        if not isinstance(estimator.model, ParameterModelFunction): # XXX is there a way around this introspection?
            raise TypeError(f'estimator.model must be of type ParameterModelFunction')

        if not isinstance(estimator.model.parameter_model, self._handler.model.__class__): 
            raise TypeError(f'estimator parameter_model must be of type {self._handler.model}')
        
        coeffs = ashraeutils.parse_coeffs(estimator.model, estimator.coeffs)
        X = estimator.X.squeeze()  # have to make this 1d
        pred_y = estimator.pred_y 
        return self._handler.run(X, pred_y, coeffs)

