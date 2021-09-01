import abc 
from dataclasses import dataclass
from typing import Callable, List, NamedTuple, Optional, Tuple, Union
from .nptypes import NByOneNDArray, OneDimNDArray

from ._lib import models, bounds

from typing_extensions import Protocol

Bound = Tuple[Tuple[float, ...], Tuple[float, ...]]

# defines two callables
class ModelCallable(Protocol): 
    def __call__(self, X: NByOneNDArray, *args: float) -> OneDimNDArray: ...

class BoundCallable(Protocol): 
    def __call__(self, X: NByOneNDArray) -> Bound: ...


@dataclass 
class EnergyParameterModelCoefficients(object): 
    yint: float 
    slopes: List[float]
    changepoints: Optional[List[float]]


class ICoefficientParser(object): 

    @abc.abstractmethod
    def parse(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients: ... 


class AbstractBaseParameterModel(abc.ABC): 
    
    def yint(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.yint


class ISingleSlopeModel(AbstractBaseParameterModel): 

    @abc.abstractmethod
    def slope(self, coeffs: EnergyParameterModelCoefficients) -> float: ... 


class ISingleChangepointModel(AbstractBaseParameterModel): 

    @abc.abstractmethod
    def changepoint(self, coeffs: EnergyParameterModelCoefficients) -> float: ...


class IDualSlopeModel(AbstractBaseParameterModel): 

    @abc.abstractmethod
    def cooling_slope(self, coeffs: EnergyParameterModelCoefficients) -> float: ...


    @abc.abstractmethod
    def heating_slope(self, coeffs: EnergyParameterModelCoefficients) -> float: ...


class IDualChangepointModel(AbstractBaseParameterModel): 

    @abc.abstractmethod
    def cooling_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> float: ...

    @abc.abstractmethod 
    def heating_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> float: ...



class LinearModel(ISingleSlopeModel): 
    
    def slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[0]
        

class SingleSlopeSingleChangepointModel(LinearModel, ISingleChangepointModel): 
    
    def changepoint(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.changepoints[0]


class DualSlopeSingleChangepointModel(IDualSlopeModel, SingleSlopeSingleChangepointModel): 
    
    def heating_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[0]

    def cooling_slope(self, coeffs: EnergyParameterModelCoefficients) -> float: 
        return coeffs.slopes[1]


class DualSlopeDualChangepointModel(DualSlopeSingleChangepointModel, IDualChangepointModel): 
    
        
    def heating_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.changepoints[0]

    def cooling_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.changepoints[1]


class LinearModelCoefficientsParser(ICoefficientParser): 

    def parse(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients: 
        yint, slope = coeffs 
        return EnergyParameterModelCoefficients(yint, [slope], None)


class SingleSlopeSingleChangepointCoefficientsParser(ICoefficientParser): 

    def parse(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients: 
        yint, slope, changepoint = coeffs 
        return EnergyParameterModelCoefficients(yint, [slope], [changepoint])


class DualSlopeSingleChangepointCoefficientsParser(ICoefficientParser): 

    def parse(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients: 
        yint, ls, rs, changepoint = coeffs 
        return EnergyParameterModelCoefficients(yint, [ls, rs], [changepoint])


class DualSlopeDualChangepointCoefficientsParser(ICoefficientParser): 

    def parse(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients: 
        yint, ls, rs, lcp, rcp = coeffs 
        return EnergyParameterModelCoefficients(yint, [ls, rs], [lcp, rcp])


class ModelFunction(object): 
    
    def __init__(self, 
        name: str, 
        f: ModelCallable, 
        bounds: Union[BoundCallable, Bound], 
        parameter_model: AbstractBaseParameterModel, 
        coefficients_parser: ICoefficientParser): 
        
        self._name = name 
        self._f = f 
        self._bounds = bounds 
        self._parameter_model = parameter_model
        self._coefficients_parser = coefficients_parser 

    @property 
    def name(self) -> str:
        return self._name 

    @property 
    def f(self) -> ModelCallable:
        return self._f  

    @property 
    def bounds(self) -> Union[BoundCallable, Bound]:
        return self._bounds 

    @property
    def parameter_model(self) -> AbstractBaseParameterModel: 
        return self._parameter_model 

    def parse_coeffs(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients: 
        return self._coefficients_parser(coeffs)


#an EnergyParameterModel must provide methods that find the energy(domain) specific components from the model coefficients


# class IEnergyParameterModel(abc.ABC): 

        
#     @abc.abstractmethod
#     def parse_coeffs(self, *coeffs) -> EnergyParameterModelCoefficients: ...

#     @abc.abstractmethod 
#     def yint(self, coeffs: EnergyParameterModelCoefficients) -> float: ...

#     @abc.abstractmethod  
#     def cooling_slope(self, coeffs: EnergyParameterModelCoefficients) -> float: ... 

#     @abc.abstractmethod 
#     def heating_slope(self, coeffs: EnergyParameterModelCoefficients) -> float: ... 

#     @abc.abstractmethod
#     def cooling_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]: ...

#     @abc.abstractmethod 
#     def heating_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]: ...
 



# # an energy parameter 
# class AbstractEnergyParameterModel(IEnergyParameterModel): 
    
#     def yint(self, coeffs: EnergyParameterModelCoefficients) -> float: # yintercept is always
#         return coeffs.yint



# class TwoParameterEnergyChangepointModel(AbstractEnergyParameterModel, IChangepointModelFunction):

#     _name = "2P"

#     def f(self) -> ModelCallable:
#         return models.twop


#     def bounds(self) -> Union[BoundCallable, Bound]: 
#         bounds.twop
    

#     def parse_coeffs(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients: 
#         yint, slope = coeffs 
#         return EnergyParameterModelCoefficients(yint, [slope], None)


#     def cooling_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
#         if coeffs.slopes[0] > 0:
#             return coeffs.slopes[0]
#         else: 
#             return 0


#     def heating_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
#         if coeffs.slopes[0] < 0:
#             return coeffs.slopes[0]
#         else:
#             return 0


#     def cooling_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:  # changepoint is a noop in 2P (linear)
#         return


#     def heating_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
#         return


# class ThreeParameterCoolingEnergyChangepointModel(AbstractEnergyParameterModel, IChangepointModelFunction): 

#     _name = "3PC"


#     def f(self) -> ModelCallable:
#         return models.threepc

#     def bounds(self) -> Union[BoundCallable, Bound]: 
#         return bounds.threepc
    
#     def parse_coeffs(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients: 
#         yint, slope, changepoint = coeffs 
#         return EnergyParameterModelCoefficients(yint, [slope], [changepoint])


#     def cooling_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
#         return coeffs.slopes[0]


#     def heating_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
#         return 0


#     def cooling_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
#         return coeffs.changepoints[0]


#     def heating_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
#         return coeffs.changepoints[0] 


# class ThreeParameterHeatingEnergyChangepointModel(AbstractEnergyParameterModel, IChangepointModelFunction): 

#     _name = "3PH"


#     def f(self) -> ModelCallable: 
#         return models.threeph


#     def bounds(self) -> Union[BoundCallable, Bound]:
#         return bounds.threeph


#     def parse_coeffs(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients: 
#         yint, slope, changepoint = coeffs 
#         return EnergyParameterModelCoefficients(yint, [slope], [changepoint])


#     def cooling_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
#         return 0


#     def heating_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
#         return coeffs.slopes[0]


#     def cooling_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
#         return coeffs.changepoints[0]


#     def heating_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
#         return coeffs.changepoints[0]
    

# class FourParameterEnergyChangepointModel(AbstractEnergyParameterModel, IChangepointModelFunction): 

#     _name = "4P"

#     def f(self) -> ModelCallable: 
#         return models.fourp

#     def bounds(self) -> Union[BoundCallable, Bound]:
#         return bounds.fourp
    
#     def parse_coeffs(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients: 
#         yint, ls, rs, changepoint = coeffs 
#         return EnergyParameterModelCoefficients(yint, [ls, rs], [changepoint])

    
#     def cooling_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
#         return coeffs.slopes[1]


#     def heating_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
#         return coeffs.slopes[0]


#     def cooling_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:   # cooling and heating changepoint are equal in 4P
#         return coeffs.changepoints[0]


#     def heating_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
#         return coeffs.changepoints[0]


# class FiveParameterEnergyChangepointModel(AbstractEnergyParameterModel, IChangepointModelFunction): 

#     _name = "5P"


#     def f(self) -> ModelCallable:
#         return models.fivep


#     def bounds(self) -> Union[BoundCallable, Bound]:
#         return bounds.fivep
    
    
#     def parse_coeffs(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients: 
#         yint, ls, rs, lcp, rcp = coeffs 
#         return EnergyParameterModelCoefficients(yint, [ls, rs], [lcp, rcp])


#     def cooling_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
#         return coeffs.slopes[1]


#     def heating_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
#         return coeffs.slopes[0]


#     def cooling_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:   # cooling and heating changepoint are equal in 4P
#         return coeffs.changepoints[1]


#     def heating_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> Optional[float]:
#         return coeffs.changepoints[0]