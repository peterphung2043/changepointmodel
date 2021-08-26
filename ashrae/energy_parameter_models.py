import abc 

from typing import List, NamedTuple, Optional, Tuple
from .base import AbstractEnergyParameterModel, Bound, NByOneNDArray, OneDimNDArray

from ._lib import models, bounds, loads 

Load = NamedTuple('Load', [('base', Optional[float]),('heating', Optional[float]), ('cooling', Optional[float]),])
Bound = NamedTuple('Bound', [('lower', Tuple[float]), ('upper', Tuple[float])])  # tuple size changes based on n params
Coeffs = NamedTuple('Coeffs', [ ('yint', float), ('slopes', List[float]), ('changepoints', List[float]) ])


#XXX the Load API needs to be rewritten to behave more like the scoring functions 
class ILoad(abc.ABC):

    @abc.abstractmethod
    def load(self, 
        X: NByOneNDArray, 
        y: NByOneNDArray, 
        pred_y: NByOneNDArray, 
        total_y: float, 
        *coeffs) -> Optional[Load]: ... # see above 


class IModelFunction(abc.ABC): 

    _name : str = ""

    def name(self):
        assert self._name != "", 'Must provide a model name' 
        return self._name

    
    @abc.abstractmethod
    def parse_coeffs(self, *coeffs) -> Coeffs: ...
        # split yint/ycp

    @abc.abstractstaticmethod
    def f(X: NByOneNDArray, *coeffs) -> OneDimNDArray: ... 
        # the function we wish to model. Must return  y array for curve_fit
        # NOTE that data possibly needs to be reshaped here into long(array) form... this is because sklearn interface only accepts [[],...] for X


    @abc.abstractstaticmethod
    def bounds(X: OneDimNDArray) -> Optional[Bound]: ...
        # we model dependent bounds calculation for curve_fit


# an energy parameter 
class AbstractEnergyParameterModel(IModelFunction, ILoad): 
    pass


class TwoParameterEnergyChangepointModel(AbstractEnergyParameterModel):

    _name = "twop"

    @staticmethod
    def f(X: NByOneNDArray, *coeffs) -> OneDimNDArray: 
        return models.twop(X, *coeffs)

    @staticmethod
    def bounds(X: OneDimNDArray) -> Optional[Bound]: 
        return bounds.twop()
    

    def parse_coeffs(self, coeffs: Tuple[float, ...]) -> Coeffs: 
        yint, slope = coeffs 
        return yint, [slope]

    def load(self, 
        X: OneDimNDArray, 
        y: OneDimNDArray, 
        pred_y: OneDimNDArray, 
        total_y: float, 
        coeffs: Tuple[float, ...]) -> Load:

        ### XXX this interface should be more like 
        yint, slope = self.parse_coeffs(coeffs)
        hl = loads.heatload(X, pred_y, slope, yint)
        cl = loads.coolingload(X, pred_y, slope, yint)
        bl = loads.baseload(total_y, hl, cl)
        
        return Load(bl, hl, cl)


class ThreeParameterCoolingEnergyChangepointModel(AbstractEnergyParameterModel): 

    _name = "threepc"

    @staticmethod
    def f(X: NByOneNDArray, *coeffs) -> OneDimNDArray: 
        return models.threepc(X, *coeffs)

    @staticmethod
    def bounds(X: OneDimNDArray) -> Optional[Bound]: 
        return bounds.threepc(X)
    
    def parse_coeffs(self, coeffs: Tuple[float, ...]) -> Coeffs: 
        yint, slope, changepoint = coeffs 
        return yint, [slope], [changepoint]

    def load(self, 
        X: OneDimNDArray, 
        y: OneDimNDArray, 
        pred_y: OneDimNDArray, 
        total_y: float, 
        coeffs: Tuple[float, ...]) -> Load:

        yint, slope, changepoint = self.parse_coeffs(coeffs)

        hl = None 
        cl = loads.coolingload(X, pred_y, slope, yint, changepoint)
        bl = loads.baseload(total_y, hl, cl)
        return Load(bl, hl, cl)


class ThreeParameterHeatingEnergyChangepointModel(AbstractEnergyParameterModel): 

    _name = "threeph"

    @staticmethod
    def f(X: NByOneNDArray, *coeffs) -> OneDimNDArray: 
        return models.threepc(X, *coeffs)

    @staticmethod
    def bounds(X: OneDimNDArray) -> Optional[Bound]: 
        return bounds.threepc(X)

    def parse_coeffs(self, coeffs: Tuple[float, ...]) -> Coeffs: 
        yint, slope, changepoint = coeffs 
        return yint, [slope], [changepoint]
    
    def load(self, 
        X: OneDimNDArray, 
        y: OneDimNDArray, 
        pred_y: OneDimNDArray, 
        total_y: float, 
        coeffs: Tuple[float, ...]) -> Load:
        
        yint, slope, changepoint = self.parse_coeffs(coeffs) 

        hl = loads.heatload(X, pred_y, slope, yint, changepoint)
        cl = None
        bl = loads.baseload(total_y, hl, cl)
        return Load(bl, hl, cl)


class FourParameterEnergyChangepointModel(AbstractEnergyParameterModel): 

    _name = "fourp"

    @staticmethod
    def f(X: NByOneNDArray, *coeffs) -> OneDimNDArray: 
        return models.fourp(X, *coeffs)

    @staticmethod
    def bounds(X: OneDimNDArray) -> Optional[Bound]: 
        return bounds.fourp(X)
    
    def parse_coeffs(self, coeffs: Tuple[float, ...]) -> Coeffs: 
        yint, ls, rs, changepoint = coeffs 
        return yint, [ls, rs], [changepoint]

    def load(self, 
        X: OneDimNDArray, 
        y: OneDimNDArray, 
        pred_y: OneDimNDArray, 
        total_y: float, 
        coeffs: Tuple[float, ...]) -> Load:
        
        yint, ls, rs, changepoint = self.parse_coeffs(coeffs) 

        hl = loads.heatload(X, pred_y, ls, yint, changepoint)
        cl = loads.coolingload(X, pred_y, rs, yint, changepoint)
        bl = loads.baseload(total_y, hl, cl)
        return Load(bl, hl, cl)



class FiveParameterEnergyChangepointModel(AbstractEnergyParameterModel): 

    _name = "fivep"

    @staticmethod
    def f(X: NByOneNDArray, *coeffs) -> OneDimNDArray: 
        return models.fivep(X, *coeffs)

    @staticmethod
    def bounds(X: OneDimNDArray) -> Optional[Bound]: 
        return bounds.fivep(X)
    
    
    def parse_coeffs(self, coeffs: Tuple[float, ...]) -> Coeffs: 
        yint, ls, rs, lcp, rcp = coeffs 
        return yint, [ls, rs], [lcp, rcp]


    def load(self, 
        X: OneDimNDArray, 
        y: OneDimNDArray, 
        pred_y: OneDimNDArray, 
        total_y: float, 
        coeffs: Tuple[float, ...]) -> Load:   

        yint, ls, rs, lcp, rcp = self.parse_coeffs(coeffs)  

        hl = loads.heatload(X, pred_y, ls, yint, lcp)
        cl = loads.coolingload(X, pred_y, rs, yint, rcp) 
        bl = loads.baseload(total_y, hl, cl)
        return Load(bl, hl, cl)