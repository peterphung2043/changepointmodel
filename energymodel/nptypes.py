import numpy as np 
from typing import Any
from nptyping import NDArray 

AnyByAnyNDArray = NDArray[(Any, ...), float]
NByOneNDArray = NDArray[(Any, 1,), float]  # [[1.], [2.], [3.], ...]
OneDimNDArray = NDArray[(Any,), float]  # [ 1., 2., 3., ...]



class AnyByAnyNDArrayField(np.ndarray):
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        arr = np.array(v, dtype=float)
        if len(arr.shape) != 2:
            raise ValueError('Shape of data should be M x n')
        return arr
    
class NByOneNDArrayField(np.ndarray):
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        arr = np.array(v, dtype=float)
        if len(arr.shape) != 2:
            raise ValueError('Shape of data should be M x n')
        if arr.shape[1] != 1:
            raise ValueError(f'Second dimension must be of size 1, got {arr.shape[1]}')
        return arr 
    
class OneDimNDArrayField(np.ndarray): 
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        arr = np.array(v, dtype=float)
        if len(arr.shape) != 1:
            raise ValueError('Shape of data should be One dimension')
        return arr 

