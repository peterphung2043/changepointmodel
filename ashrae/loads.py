""" Derive the loads from the estimator
"""
import abc
from ashrae.energy_parameter_models import EnergyParameterModelCoefficients, IEnergyParameterModel
import numpy as np
from ashrae.nptypes import NByOneNDArray, OneDimNDArray
from typing import List, NamedTuple, Optional
from .estimator import EnergyChangepointEstimator
from .scoring import ScoringFunction

from ._lib import loads as _loads 

from .energy_parameter_models import AbstractEnergyParameterModel
from dataclasses import dataclass 

@dataclass 
class EnergyChangepointLoad(object): 
    base: float 
    heating: float 
    cooling: float 
    

# Some of the logic in here is bound to how curvefit estimator and by proxy we handle coeffs for model types.
# These are interfaces also specifcally for the current family of changepoint models. 

class IEnergyChangepointLoad(ScoringFunction): 

    @abc.abstractmethod
    def get_slope(self, model: IEnergyParameterModel) -> float: ...
    """ Given a list of slopes for a model should contain the correct logic for returning the slope"""


    @abc.abstractmethod 
    def get_changepoint(self, model: IEnergyParameterModel) -> float: ...
    """ Given a list of changepoints for a model should contain the correct logic for returning the slope """

    @abc.abstractmethod 
    def calc(self, X: OneDimNDArray, pred_y: OneDimNDArray, slope: float, yint: float, changepoint: Optional[float]) -> float: ...



class IBaseload(ScoringFunction): 

    @abc.abstractmethod 
    def calc(self, total_consumption: float,  heating: float, cooling: float) -> float: ...



class HeatingEnergyChangpointModelLoad(IEnergyChangepointLoad): 


    def get_slope(self, model: IEnergyParameterModel, coeffs: EnergyParameterModelCoefficients) -> Optional[float]: 
        return model.heating_slope(coeffs)


    def get_changepoint(self, model: IEnergyParameterModel, coeffs: EnergyParameterModelCoefficients) -> float:
        return model.heating_changepoint(coeffs)


    def calc(self, X: OneDimNDArray, pred_y: OneDimNDArray, slope: float, yint: float, changepoint: Optional[float]) -> float: 

        if slope > 0: # pos slope no heat
            return 0

        if changepoint is None: # no changepoint then set to inf  (handles linear model loads)
            changepoint = np.inf 

        return _loads.heatload(X, pred_y, yint, changepoint)



class CoolingEnergyChangepointModelLoad(IEnergyChangepointLoad): 


    def get_slope(self, model: IEnergyParameterModel, coeffs: EnergyParameterModelCoefficients) -> Optional[float]: 
        return model.cooling_slope(coeffs)


    def get_changepoint(self, model: IEnergyParameterModel, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        return model.cooling_changepoint(coeffs)


    def calc(self, X: OneDimNDArray, pred_y: OneDimNDArray, slope: float, yint: float, changepoint: Optional[float]) -> float: 
        if slope < 0: # neg slope no cool
            return 0
        
        if changepoint is None:  # no cp then set to -inf (handles linear model loads)
            changepoint = -np.inf
        
        return _loads.coolingload(X, pred_y, yint, changepoint)



class Baseload(IBaseload): 

    def calc(self, total_consumption: float, heating: float, cooling: float) -> float: 
        return _loads.baseload(total_consumption, heating, cooling)



class EnergyChangepointLoadHandler(object): 

    def __init__(self, load: EnergyChangepointLoad): 
        self._load = load


    def calc(self, 
        X: NByOneNDArray, 
        pred_y: OneDimNDArray, 
        model: AbstractEnergyParameterModel, 
        coeffs: EnergyParameterModelCoefficients) -> float: 
        
        slope = self._load.get_slope(model, coeffs)        
        cp = self._load.get_changepoint(model, coeffs)
        yint = model.yint(coeffs)

        return self._load(X, pred_y, slope, yint, cp)



class EnergyChangepointLoadsAggregator(object): 

    def __init__(self, 
        heating: EnergyChangepointLoadHandler, 
        cooling: EnergyChangepointLoadHandler, 
        base: IBaseload): 

        self._heating = heating 
        self._cooling = cooling 
        self._base = base

    def aggregate(self, estimator: EnergyChangepointEstimator) -> EnergyChangepointLoad: 

        X = estimator.X.squeeze()  # have to make this 1d
        pred_y = estimator.pred_y 
        
        model = estimator.model 
        coeffs = model.parse_coeffs(estimator.coeffs)

        hl = self._heating.calc(X, pred_y, model, coeffs)
        cl = self._cooling.calc(X, pred_y, model, coeffs)
        bl = self._base.calc(X, pred_y, model, coeffs)

        return EnergyChangepointLoad(bl, hl, cl)