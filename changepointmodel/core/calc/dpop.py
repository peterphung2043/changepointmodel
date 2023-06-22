import numpy as np
import numpy.typing as npt
from typing import Tuple, List, Optional
from changepointmodel.core.nptypes import OneDimNDArrayField


def twop():
    ...


def threepc(X: OneDimNDArrayField, changepoint: float) -> float:
    return sum(X >= changepoint)


def threeph(X: OneDimNDArrayField, changepoint: float) -> float:
    return sum(X <= changepoint)


def fourp(X: OneDimNDArrayField, changepoint: float) -> Tuple[float, float]:
    heatnum = sum(X <= changepoint)
    coolnum = sum(X > changepoint)
    return coolnum, heatnum


def fivep(X: OneDimNDArrayField, changepoints: List[float]) -> Tuple[float, float]:
    heatnum = sum(X <= changepoints[0])
    coolnum = sum(X >= changepoints[1])
    return coolnum, heatnum
