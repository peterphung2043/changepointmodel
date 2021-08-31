import abc 
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple
from .nptypes import NByOneNDArray, OneDimNDArray

from ._lib import models, bounds


Bound = Tuple[Tuple[float, ...], Tuple[float, ...]]

@dataclass 
class EnergyParameterModelCoefficients(object): 
    yint: float 
    slopes: List[float]
    changepoints: Optional[List[float]]


class IChangepointModelFunction(abc.ABC): 

    _name : str = ""

    @property
    def name(self):
        assert self._name != "", 'Must provide a model name' 
        return self._name

    @abc.abstractstaticmethod
    def f(X: NByOneNDArray, *coeffs) -> OneDimNDArray: ... 

    @abc.abstractstaticmethod
    def bounds(X: OneDimNDArray) -> Bound: ...


# an EnergyParameterModel must provide methods that find the energy(domain) specific components from the model coefficients
class IEnergyParameterModel(abc.ABC): 

        
    @abc.abstractmethod
    def parse_coeffs(self, *coeffs) -> EnergyParameterModelCoefficients: ...

    @abc.abstractmethod 
    def yint(self, coeffs: EnergyParameterModelCoefficients) -> float: ...

    @abc.abstractmethod  
    def cooling_slope(self, coeffs: EnergyParameterModelCoefficients) -> float: ... 

    @abc.abstractmethod 
    def heating_slope(self, coeffs: EnergyParameterModelCoefficients) -> float: ... 

    @abc.abstractmethod
    def cooling_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]: ...

    @abc.abstractmethod 
    def heating_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]: ...
 
    
# an energy parameter 
class AbstractEnergyParameterModel(IEnergyParameterModel): 
    
    def yint(self, coeffs: EnergyParameterModelCoefficients) -> float: # yintercept is always
        return coeffs.yint


class TwoParameterEnergyChangepointModel(AbstractEnergyParameterModel, IChangepointModelFunction):

    _name = "2P"

    @staticmethod
    def f(X: NByOneNDArray, *coeffs) -> OneDimNDArray: 
        return models.twop(X, *coeffs)

    @staticmethod
    def bounds(X: OneDimNDArray) -> Bound: 
        return bounds.twop()
    

    def parse_coeffs(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients: 
        yint, slope = coeffs 
        return EnergyParameterModelCoefficients(yint, [slope], None)


    def cooling_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        if coeffs.slopes[0] > 0:
            return coeffs.slopes[0]
        else: 
            return 0


    def heating_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        if coeffs.slopes[0] < 0:
            return coeffs.slopes[0]
        else:
            return 0


    def cooling_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:  # changepoint is a noop in 2P (linear)
        return


    def heating_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        return


class ThreeParameterCoolingEnergyChangepointModel(AbstractEnergyParameterModel, IChangepointModelFunction): 

    _name = "3PC"

    @staticmethod
    def f(X: NByOneNDArray, *coeffs) -> OneDimNDArray: 
        return models.threepc(X, *coeffs)

    @staticmethod
    def bounds(X: OneDimNDArray) -> Bound: 
        return bounds.threepc(X)
    
    def parse_coeffs(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients: 
        yint, slope, changepoint = coeffs 
        return EnergyParameterModelCoefficients(yint, [slope], [changepoint])


    def cooling_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[0]


    def heating_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return 0


    def cooling_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        return coeffs.changepoints[0]


    def heating_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        return coeffs.changepoints[0] 


class ThreeParameterHeatingEnergyChangepointModel(AbstractEnergyParameterModel, IChangepointModelFunction): 

    _name = "3PH"

    @staticmethod
    def f(X: NByOneNDArray, *coeffs) -> OneDimNDArray: 
        return models.threeph(X, *coeffs)

    @staticmethod
    def bounds(X: OneDimNDArray) -> Bound: 
        return bounds.threeph(X)

    def parse_coeffs(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients: 
        yint, slope, changepoint = coeffs 
        return EnergyParameterModelCoefficients(yint, [slope], [changepoint])


    def cooling_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return 0


    def heating_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[0]


    def cooling_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        return coeffs.changepoints[0]


    def heating_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        return coeffs.changepoints[0]
    

class FourParameterEnergyChangepointModel(AbstractEnergyParameterModel, IChangepointModelFunction): 

    _name = "4P"

    @staticmethod
    def f(X: NByOneNDArray, *coeffs) -> OneDimNDArray: 
        return models.fourp(X, *coeffs)

    @staticmethod
    def bounds(X: OneDimNDArray) -> Bound: 
        return bounds.fourp(X)
    
    def parse_coeffs(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients: 
        yint, ls, rs, changepoint = coeffs 
        return EnergyParameterModelCoefficients(yint, [ls, rs], [changepoint])

    
    def cooling_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[1]


    def heating_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[0]


    def cooling_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:   # cooling and heating changepoint are equal in 4P
        return coeffs.changepoints[0]


    def heating_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        return coeffs.changepoints[0]


class FiveParameterEnergyChangepointModel(AbstractEnergyParameterModel, IChangepointModelFunction): 

    _name = "5P"

    @staticmethod
    def f(X: NByOneNDArray, *coeffs) -> OneDimNDArray: 
        return models.fivep(X, *coeffs)

    @staticmethod
    def bounds(X: OneDimNDArray) -> Bound: 
        return bounds.fivep(X)
    
    
    def parse_coeffs(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients: 
        yint, ls, rs, lcp, rcp = coeffs 
        return EnergyParameterModelCoefficients(yint, [ls, rs], [lcp, rcp])


    def cooling_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[1]


    def heating_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[0]


    def cooling_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:   # cooling and heating changepoint are equal in 4P
        return coeffs.changepoints[1]


    def heating_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        return coeffs.changepoints[0]