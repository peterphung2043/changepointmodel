import abc 

from typing import List, NamedTuple, Optional, Tuple
from .base import AbstractEnergyParameterModel, Bound, NByOneNDArray, OneDimNDArray

from ._lib import models, bounds, loads 


Bound = NamedTuple('Bound', [('lower', Tuple[float]), ('upper', Tuple[float])])  # tuple size changes based on n params
EnergyParameterModelCoefficients = NamedTuple('ChangepointChangepointCoefficients', [ ('yint', float), ('slopes', List[float]), ('changepoints', Optional[List[float]]) ])
# XXX coeffs replaces sensitivities which are basically just alias

class IChangepointModelFunction(abc.ABC): 

    _name : str = ""

    def name(self):
        assert self._name != "", 'Must provide a model name' 
        return self._name

    @abc.abstractstaticmethod
    def f(X: NByOneNDArray, *coeffs) -> OneDimNDArray: ... 

    @abc.abstractstaticmethod
    def bounds(X: OneDimNDArray) -> Optional[Bound]: ...


# an EnergyParameterModel must provide methods that find the energy(domain) specific components from the model coefficients
class IEnergyParameterModel(abc.ABC): 

        
    @abc.abstractmethod
    def parse_coeffs(self, *coeffs) -> EnergyParameterModelCoefficients: ...

    @abc.abstracmethod 
    def yint(self, coeffs: EnergyParameterModelCoefficients) -> float: ...

    @abc.abstractmethod  
    def cooling_slope(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]: ... 

    @abc.abstractmethod 
    def heating_slope(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]: ... 

    @abc.abstractmethod
    def cooling_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]: ...

    @abc.abstractmethod 
    def heating_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]: ...
 
    
# an energy parameter 
class AbstractEnergyParameterModel(IEnergyParameterModel, IEnergyParameterModel): 
    
    def yint(self, coeffs: EnergyParameterModelCoefficients) -> float: # yintercept is always
        return coeffs[0]


class TwoParameterEnergyChangepointModel(AbstractEnergyParameterModel):

    _name = "twop"

    @staticmethod
    def f(X: NByOneNDArray, *coeffs) -> OneDimNDArray: 
        return models.twop(X, *coeffs)

    @staticmethod
    def bounds(X: OneDimNDArray) -> Optional[Bound]: 
        return bounds.twop()
    

    def parse_coeffs(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients: 
        yint, slope = coeffs 
        return yint, [slope], None


    def cooling_slope(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        if coeffs.slopes[0] > 0:
            return coeffs.slopes[0]


    def heating_slope(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        if coeffs.slopes[0] < 0:
            return coeffs.slopes[0]


    def cooling_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:  # changepoint is a noop in 2P (linear)
        return


    def heating_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        return


class ThreeParameterCoolingEnergyChangepointModel(AbstractEnergyParameterModel): 

    _name = "threepc"

    @staticmethod
    def f(X: NByOneNDArray, *coeffs) -> OneDimNDArray: 
        return models.threepc(X, *coeffs)

    @staticmethod
    def bounds(X: OneDimNDArray) -> Optional[Bound]: 
        return bounds.threepc(X)
    
    def parse_coeffs(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients: 
        yint, slope, changepoint = coeffs 
        return yint, [slope], [changepoint]


    def cooling_slope(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        return coeffs.slopes[0]


    def heating_slope(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        return


    def cooling_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        return coeffs.changepoints[0]


    def heating_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        return coeffs.changepoints[0] 


class ThreeParameterHeatingEnergyChangepointModel(AbstractEnergyParameterModel): 

    _name = "threeph"

    @staticmethod
    def f(X: NByOneNDArray, *coeffs) -> OneDimNDArray: 
        return models.threepc(X, *coeffs)

    @staticmethod
    def bounds(X: OneDimNDArray) -> Optional[Bound]: 
        return bounds.threepc(X)

    def parse_coeffs(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients: 
        yint, slope, changepoint = coeffs 
        return yint, [slope], [changepoint]


    def cooling_slope(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        return 


    def heating_slope(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        return coeffs.slopes[0]


    def cooling_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        return coeffs.changepoints[0]


    def heating_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        return coeffs.changepoints[0]
    
    

class FourParameterEnergyChangepointModel(AbstractEnergyParameterModel): 

    _name = "fourp"

    @staticmethod
    def f(X: NByOneNDArray, *coeffs) -> OneDimNDArray: 
        return models.fourp(X, *coeffs)

    @staticmethod
    def bounds(X: OneDimNDArray) -> Optional[Bound]: 
        return bounds.fourp(X)
    
    def parse_coeffs(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients: 
        yint, ls, rs, changepoint = coeffs 
        return yint, [ls, rs], [changepoint]

    
    def cooling_slope(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        return coeffs.slopes[0]


    def heating_slope(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        return coeffs.slopes[1]


    def cooling_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:   # cooling and heating changepoint are equal in 4P
        return coeffs.changepoints[0]


    def heating_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        return coeffs.changepoints[0]


class FiveParameterEnergyChangepointModel(AbstractEnergyParameterModel): 

    _name = "fivep"

    @staticmethod
    def f(X: NByOneNDArray, *coeffs) -> OneDimNDArray: 
        return models.fivep(X, *coeffs)

    @staticmethod
    def bounds(X: OneDimNDArray) -> Optional[Bound]: 
        return bounds.fivep(X)
    
    
    def parse_coeffs(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients: 
        yint, ls, rs, lcp, rcp = coeffs 
        return yint, [ls, rs], [lcp, rcp]


    def cooling_slope(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        return coeffs.slopes[0]


    def heating_slope(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        return coeffs.slopes[1]


    def cooling_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:   # cooling and heating changepoint are equal in 4P
        return coeffs.changepoints[0]


    def heating_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
        return coeffs.changepoints[0]