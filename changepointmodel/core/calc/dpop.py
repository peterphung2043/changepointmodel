import numpy as np
import numpy.typing as npt
from typing import Tuple, List, Optional
from changepointmodel.core.nptypes import OneDimNDArrayField


def twop():
    ...


def threepc(
    X: OneDimNDArrayField, changepoint: float, threshold: Optional[float] = None
) -> bool:
    if threshold is None:
        threshold = len(X) / 4

    heatnum = sum(X <= changepoint)
    return heatnum >= threshold and len(X) - heatnum >= threshold


def threeph():
    ...


def fourp():
    ...


def fivep():
    ...
