from typing import Optional
from .base import AbstractEnergyParameterModel, Load, Bound, NByOneNDArray, OneDimNDArray

from ._lib import models, bounds, loads 


class TwoParameterEnergyChangepointModel(AbstractEnergyParameterModel):

    _name = "twop"

    @staticmethod
    def f(X: NByOneNDArray, *coeffs) -> OneDimNDArray: 
        return models.twop(X, *coeffs)

    @staticmethod
    def bounds(X: OneDimNDArray) -> Optional[Bound]: 
        return bounds.twop()
    
    def load(self, 
        X: OneDimNDArray, 
        y: OneDimNDArray, 
        pred_y: OneDimNDArray, 
        total_y: float, 
        *coeffs) -> Load:
        yint, slope = coeffs
        hl = loads.heatload(X, pred_y, slope, yint)
        cl = loads.coolingload(X, pred_y, slope, yint)
        bl = loads.baseload(total_y, hl, cl)

        if slope > 0: # XXX if we define heating and cooling models for anything this logic becomes unnessaray
            hs = None
            cs = slope
        else:
            hs = slope, 
            cs = None
        
        return Load(bl, hl, cl, hs, cs)


class ThreeParameterCoolingEnergyChangepointModel(AbstractEnergyParameterModel): 

    _name = "threepc"

    @staticmethod
    def f(X: NByOneNDArray, *coeffs) -> OneDimNDArray: 
        return models.threepc(X, *coeffs)

    @staticmethod
    def bounds(X: OneDimNDArray) -> Optional[Bound]: 
        return bounds.threepc(X)
    
    def load(self, 
        X: OneDimNDArray, 
        y: OneDimNDArray, 
        pred_y: OneDimNDArray, 
        total_y: float, 
        *coeffs) -> Load:

        yint, slope, changepoint = coeffs 

        hl = None 
        cl = loads.coolingload(X, pred_y, slope, yint, changepoint)
        bl = loads.baseload(total_y, hl, cl)
        hs = None 
        cs = slope  

        return Load(bl, hl, cl, hs, cs)


class ThreeParameterHeatingEnergyChangepointModel(AbstractEnergyParameterModel): 

    _name = "threeph"

    @staticmethod
    def f(X: NByOneNDArray, *coeffs) -> OneDimNDArray: 
        return models.threepc(X, *coeffs)

    @staticmethod
    def bounds(X: OneDimNDArray) -> Optional[Bound]: 
        return bounds.threepc(X)
    
    def load(self, 
        X: OneDimNDArray, 
        y: OneDimNDArray, 
        pred_y: OneDimNDArray, 
        total_y: float, 
        *coeffs) -> Load:
        
        yint, slope, changepoint = coeffs 

        hl = loads.heatload(X, pred_y, slope, yint, changepoint)
        cl = None
        bl = loads.baseload(total_y, hl, cl)
        cs = None 
        hs = slope 

        return Load(bl, hl, cl, hs, cs)


class FourParameterEnergyChangepointModel(AbstractEnergyParameterModel): 

    _name = "fourp"

    @staticmethod
    def f(X: NByOneNDArray, *coeffs) -> OneDimNDArray: 
        return models.threepc(X, *coeffs)

    @staticmethod
    def bounds(X: OneDimNDArray) -> Optional[Bound]: 
        return bounds.threepc(X)
    
    def load(self, 
        X: OneDimNDArray, 
        y: OneDimNDArray, 
        pred_y: OneDimNDArray, 
        total_y: float, 
        *coeffs) -> Load:
        
        yint, ls, rs, changepoint = coeffs 

        hl = loads.heatload(X, pred_y, ls, yint, changepoint)
        cl = loads.coolingload(X, pred_y, rs, yint, changepoint)
        bl = loads.baseload(total_y, hl, cl)
        cs = rs 
        hs = ls 

        return Load(bl, hl, cl, hs, cs)



class FiveParameterEnergyChangepointModel(AbstractEnergyParameterModel): 

    _name = "fivep"

    @staticmethod
    def f(X: NByOneNDArray, *coeffs) -> OneDimNDArray: 
        return models.threepc(X, *coeffs)

    @staticmethod
    def bounds(X: OneDimNDArray) -> Optional[Bound]: 
        return bounds.threepc(X)
    
    def load(self, 
        X: OneDimNDArray, 
        y: OneDimNDArray, 
        pred_y: OneDimNDArray, 
        total_y: float, 
        *coeffs) -> Load:   

        yint, ls, rs, lcp, rcp = coeffs 

        hl = loads.heatload(X, pred_y, ls, yint, lcp)
        cl = loads.coolingload(X, pred_y, rs, yint, rcp) 
        bl = loads.baseload(total_y, hl, cl)
        cs = rs 
        hs = ls

        return Load(bl, hl, cl, hs, cs)