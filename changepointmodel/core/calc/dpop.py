import numpy as np
import numpy.typing as npt
from typing import Tuple, List, Optional
from changepointmodel.core.nptypes import OneDimNDArrayField

HeatingCoolingPoints = Tuple[float, float]


def twop(X: OneDimNDArrayField, slope: float) -> HeatingCoolingPoints:
    return (len(X), 0) if slope <= 0 else (0, len(X))


def threepc(X: OneDimNDArrayField, changepoint: float) -> HeatingCoolingPoints:
    return 0, sum(X >= changepoint)


def threeph(X: OneDimNDArrayField, changepoint: float) -> HeatingCoolingPoints:
    return sum(X <= changepoint), 0


def fourp(X: OneDimNDArrayField, changepoint: float) -> HeatingCoolingPoints:
    heatnum = sum(X <= changepoint)
    coolnum = sum(X > changepoint)
    return coolnum, heatnum


def fivep(X: OneDimNDArrayField, changepoints: List[float]) -> HeatingCoolingPoints:
    heatnum = sum(X <= changepoints[0])
    coolnum = sum(X >= changepoints[1])
    return coolnum, heatnum
