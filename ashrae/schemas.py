"""Defines types to use as inputs to CurvefitEstimator. This is agnostic of function since 
NumpyArray can be n-dimensional
"""

from datetime import datetime
import numpy as np
from typing import Any, Dict, List, Optional

import pydantic

from .scoring import Score
from .savings import AdjustedSavingsResult, NormalizedSavingsResult
from .energy_parameter_models import EnergyParameterModelCoefficients
from .loads import EnergyChangepointLoad

from .base import AnyByAnyNDArray, NByOneNDArray



class CurvefitEstimatorDataModel(pydantic.BaseModel): 
    X: AnyByAnyNDArray
    y: Optional[NByOneNDArray]   # NOTE this is optional so that different X values may be passed to a fit model
    sigma: Optional[NByOneNDArray]=None
    absolute_sigma: Optional[bool]=None


    @pydantic.validator('X')
    def validate_X(cls, v: AnyByAnyNDArray) -> AnyByAnyNDArray: # we don't want to overdo this check... but a basic reshape here is extremely helpful for processing
        if v.ndim == 1:  # assure 1d is reshaped according skl spec
            return v.reshape(-1, 1)        
        return np.atleast_2d(v)   # assure that anything else is at least 2d .. NOTE will not check for nested data... just know what your doing...


    @pydantic.root_validator
    def validate(cls, values: Dict[str, Any]) -> Dict[str, Any]: 

        if values['y'] is not None:  # if we are only passing X then we can skip validation

            xlen = len(values['X'])
            ylen = len(values['y'])

            if xlen != ylen: 
                raise ValueError(f'X and y lengths do not match.')
            
        return values 

# result types generate by factory methods
# XXX should probably make these dataclasses internally ... might be less confusing then handling from named tuple

EnergyParameterModelCoefficientsModel = pydantic.create_model_from_namedtuple(EnergyParameterModelCoefficients)
EnergyChangepointLoadModel = pydantic.create_model_from_namedtuple(EnergyChangepointLoad)
ScoreModel = pydantic.create_model_from_namedtuple(Score)
AdjustedSavingsResultModel = pydantic.create_model_from_namedtuple(AdjustedSavingsResult)
NormalizedSavingsResultModel = pydantic.create_model_from_namedtuple(NormalizedSavingsResult)


class EnergyChangepointModelResult(pydantic.BaseModel): 
    name: str 
    coeffs: EnergyParameterModelCoefficientsModel
    pred_y: NByOneNDArray
    load: EnergyChangepointLoadModel
    score: List[ScoreModel]

class AdjustedEnergyChangepointModelSavingsResult(pydantic.BaseModel): 

    pre: EnergyChangepointModelResult 
    post: EnergyChangepointModelResult
    adjusted_savings: AdjustedSavingsResultModel
    normalized_savings = Optional[NormalizedSavingsResult]


