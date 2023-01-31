from changepointmodel.nptypes import OneDimNDArrayField, NByOneNDArrayField, AnyByAnyNDArrayField
import pydantic 
from typing import List, Optional, Union, Any
import numpy as np
from changepointmodel.schemas import NpConfig, CurvefitEstimatorDataModel
from pydantic import Field
from .enums import FilterHowEnum, FilterWhichEnum


SklScoreReturnType =  Union[float, AnyByAnyNDArrayField, Any]


class UsageTemperatureData(pydantic.BaseModel): 
    oat: List[float]=Field(..., description='Outside air temperature for changepoint regression. Should be same len as usage.')
    usage: List[float]=Field(..., description='Usage points for changepoint regression. Should be same len as oat.')
    
    @pydantic.validator('oat')
    def validate_oat(cls, v):
        if len(v) == 0:
            raise ValueError("oat cannot be empty")
        return v

    @pydantic.validator('usage')
    def validate_usage(cls, v, values):
        if len(v) == 0:
            raise ValueError("usage cannot be empty")
        if 'oat' in values and len(v) != len(values['oat']):
            raise ValueError('sizes of oat and usage arrays are mismatched.')
        return v




class FilterConfig(pydantic.BaseModel): 
    which: FilterWhichEnum=Field(default=FilterWhichEnum.r2, description='Filter on r2 or cvrmse value.')
    how: FilterHowEnum=Field(default=FilterHowEnum.threshold_ok_first_is_best, description='How to filter the changepointmodels. Default is threshold_ok_first_is_best.')
    extras: bool=Field(default=False, description='Flag to filter models through data population test, shape test and t-test.')    

#filtered to calcuale savings, but return all the models
class ChangepointModelConfig(pydantic.BaseModel):
    models: List[str]=Field(..., description="Model types to attempt. Options are 5P, 4P, 3PC, 3PH, 2P.")
    r2_threshold: pydantic.confloat(ge=0, le=1)=Field(0.75, description="The r2 threshold to report. After modeling we will analyze the r2 and see if it is gte this number.")
    cvrmse_threshold: pydantic.confloat(ge=0, le=1)=Field(0.25, description="The cvrmse threshold. After modeling we will analyze the cvrmse value and see if it lte this number.")
    model_filter: Optional[FilterConfig]

    @pydantic.validator('models')
    def validate_models_name(cls, v):
        for i in v:
            if i not in ['2P', '3PC', '3PH', '4P', '5P']:
                raise ValueError(f"{i} not a valid model")
        return v


class EnergyChangepointModelRequest(pydantic.BaseModel):
    nonzero_threshold: pydantic.confloat(le=1, gt=0)=Field(0.8, description='Threshold for percent of number of non-zero input data points.')
    model_config: ChangepointModelConfig=Field(..., description='A configuration for this modeling request.')
    usage: UsageTemperatureData=Field(..., description='Usage and temperature data for this modeling request.')
    
    @pydantic.validator('usage')
    def validate_usage_threshold(cls, v, values): #usage UsageTemperatureData will get validated first
        def _validate_with_threshold(value_ls):
            zeros = [i for i in value_ls if i == 0.0]
            percent_non_zeros = 1 - (len(zeros)/len(value_ls))
            if percent_non_zeros < values['nonzero_threshold']:
                raise ValueError("number of non-zeros data point exceeds allowed threshold.")
        #_validate_with_threshold(v.oat) -- see #211
        _validate_with_threshold(v.usage)
        return v

class BaselineChangepointModelRequest(EnergyChangepointModelRequest):

    norms: Optional[List[float]]=Field(None, description='Optional normalized outside air temperature data. Used to calculate NAC. Length can be 12 (monthly), 365 (daily) or 24 (hourly).')
    
    @pydantic.validator('norms')
    def validate_norms(cls, v): 
        if v and len(v) not in [12, 24, 365]:
            raise ValueError('Invalid length of normalized temperature data. Should be 12 (monthly), 24 (hourly) or 365 (daily).')
        return v


class SavingsRequest(pydantic.BaseModel):
    pre: EnergyChangepointModelRequest=Field(..., description='The pre-retrofit data for this option-c request.')
    post: EnergyChangepointModelRequest=Field(..., description='The post-retrofit data for this option-c request.')
    confidence_interval: Optional[pydantic.confloat(le=1, ge=0.5)]=Field(0.8, description='Optional confidence interval for savings calculations using monthly data. The library default value is 80%')
    scalar: Optional[pydantic.confloat(ge=1.0)]=Field(30.473, description='Value to scale savings data. If giving data by per/day values, the default of 30.473 will scale these values to a correct monthly value.')
    norms: Optional[List[float]]=Field(None, description='Optional normalized outside air temperature data. Used to calculate NAC. Length must be 12 (monthly)')
    
    @pydantic.validator('pre', 'post')
    def validate_usage(cls, v):
        if len(v.usage.usage) != 12 or len(v.usage.oat) != 12: 
            raise ValueError('Pre and post number of points must be 12 to perform monthly savings calculations.')
        return v

    @pydantic.validator('norms')
    def validate_norms(cls, v): 
        if v and len(v) != 12:
            raise ValueError('Invalid length of normalized temperature data. Length must be 12 to perform monthly savings calculations.')
        return v


# below models are ported from energymodel to solve data class pickling issues
class PredictedSum(pydantic.BaseModel): 
    value: float 

class Load(pydantic.BaseModel): 
    base: float 
    heating: float 
    cooling: float 

class EnergyParameterModelCoefficients(pydantic.BaseModel): 
    yint: float 
    slopes: List[float]
    changepoints: Optional[List[float]]

class Score(pydantic.BaseModel): 
    name: str 
    value: SklScoreReturnType 
    threshold: float 
    ok: bool 


class EnergyChangepointModelInputData(pydantic.BaseModel): 
    X: OneDimNDArrayField
    y: OneDimNDArrayField


class EnergyChangepointModelResult(pydantic.BaseModel): 
    name: str 
    coeffs: EnergyParameterModelCoefficients
    pred_y: OneDimNDArrayField
    load: Load
    scores: List[Score]
    input_data: EnergyChangepointModelInputData # XXX <--- not a data class...it is returning a dictionary s
    nac: Optional[PredictedSum]=None

    class Config(NpConfig): ... 
    

class AdjustedSavingsResult(pydantic.BaseModel): 
    adjusted_y: OneDimNDArrayField
    total_savings: float 
    average_savings: float
    percent_savings: float 
    percent_savings_uncertainty: float 

class NormalizedSavingsResult(pydantic.BaseModel): 
    normalized_y_pre: OneDimNDArrayField
    normalized_y_post: OneDimNDArrayField
    total_savings: float
    average_savings : float 
    percent_savings : float 
    percent_savings_uncertainty : float 


class AdjustedSavingsResultData(pydantic.BaseModel): 
    confidence_interval: float
    result: AdjustedSavingsResult

    class Config(NpConfig): ... 

class NormalizedSavingsResultData(pydantic.BaseModel): 

    confidence_interval: float
    result: NormalizedSavingsResult

    class Config(NpConfig): ... 

class SavingsResult(pydantic.BaseModel): 

    pre: EnergyChangepointModelResult 
    post: EnergyChangepointModelResult
    adjusted_savings: AdjustedSavingsResultData
    normalized_savings: Optional[NormalizedSavingsResultData]=None

    class Config(NpConfig): ... 


# api response
class EnergyChangepointModelResponse(pydantic.BaseModel):
    results: List[EnergyChangepointModelResult]

    class Config(NpConfig): ...


class SavingsResponse(pydantic.BaseModel):
    results: List[SavingsResult]

    class Config(NpConfig): ...
