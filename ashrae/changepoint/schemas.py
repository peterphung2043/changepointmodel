"""Defines types to use as inputs to CurvefitEstimator. This is agnostic of function since 
NumpyArray can be n-dimensional
"""

import abc
from datetime import datetime
import numpy as np
from typing import Any, Dict, Optional

import pydantic
from ..base_types import NumpyArray


class CurvefitEstimatorRegressorTargetData(pydantic.BaseModel): 
    X: NumpyArray

    @pydantic.validator('X')
    def validate_X(cls, v: NumpyArray) -> NumpyArray: # we don't want to overdo this check... but a basic reshape here is extremely helpful for processing
        if v.ndim == 1:  # assure 1d is reshaped according skl spec
            return v.reshape(-1, 1)        
        return np.atleast_2d(v)   # assure that anything else is at least 2d .. NOTE will not check for nested data... just know what your doing...



class CurvefitEstimatorTargetData(CurvefitEstimatorRegressorTargetData): 
    y: NumpyArray
    sigma: Optional[NumpyArray]=None
    absolute_sigma: Optional[bool]=None

    @pydantic.root_validator
    def validate(cls, values: Dict[str, Any]) -> Dict[str, Any]: 
        # check if X and y  
        xlen = len(values['X'])
        ylen = len(values['y'])

        if xlen != ylen: 
            raise ValueError(f'X and y lengths do not match.')
        
        return values 

    
# class ChangepointModelInput(pydantic.BaseModel): 
#     meta: Optional[Metadata] # application specific (anything that is serializable)
#     measures: CurvefitEstimatorInputData

