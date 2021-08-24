# XXX clean this up and integrate
# NOTE that 

from ..base import NByOneFloatNDArray, OneDimNDArray


def twop(X:NByOneFloatNDArray, yint: float, m: float) -> OneDimNDArray: 
    return (m * X + yint).squeeze()



def threepc(X: NByOneFloatNDArray, yint: float, m: float, cp: float) -> OneDimNDArray: 
    return (
        (X < cp) * (yint) +
        (X >= cp) * (m * (X - cp) + yint)).squeeze()


def threeph(X: NByOneFloatNDArray, yint: float, m: float, cp:float) -> OneDimNDArray: 
    return(
        (X < cp) * (m * (X - cp) + yint) +
        (X >= cp) * (yint)).squeeze()


def fourp(X: NByOneFloatNDArray, yint: float, m1: float, m2: float, cp: float) -> OneDimNDArray: 
    return(
        (X < cp) * (m1 * (X - cp) + yint) +
        (X >= cp) * (m2 * (X - cp) + yint)).squeeze()


def fivep(X: NByOneFloatNDArray, yint: float, m1: float, m2: float, cp1: float, cp2: float) -> OneDimNDArray: 
    return(
        (X < cp1) * (m1 * (X - cp1) + yint) +
        ((X < cp2) & (X >= cp1)) * (yint) +
        (X >= cp2) * (m2 * (X - cp2) + yint)).squeeze()

