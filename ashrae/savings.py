""" APIs for handling adjusted and normalized savings as per ashrae.
"""

import numpy as np
from ashrae.base import NByOneNDArray, OneDimNDArray
from typing import NamedTuple, Optional
from ashrae.estimator import EnergyChangepointEstimator
from ._lib import savings as ashraesavings

from .scoring import Cvrmse
cvrmse_score = Cvrmse()

import abc 


AdjustedSavingsResult = NamedTuple('SavingsResult', [
    ('adjusted_y', OneDimNDArray),
    ('total_savings', float), 
    ('average_monthly_savings', float), 
    ('percent_savings', float), 
    ('percent_savings_uncertainty', float)])

NormalizedSavingsResult = NamedTuple('NormalizedSavingsResult', [
    ('normalized_X', NByOneNDArray),
    ('normalized_y_pre', OneDimNDArray),
    ('normalized_y_post', OneDimNDArray), 
    ('total_savings', float), 
    ('average_monthly_savings', float), 
    ('percent_savings', float), 
    ('percent_savings_uncertainty', float)])


class ISavingsCalculator(abc.ABC): 
    
    def save(self, 
        pre: EnergyChangepointEstimator, 
        post: EnergyChangepointEstimator) -> AdjustedSavingsResult: ...
    

class AbstractSavings(abc.ABC):

    def __init__(self, confidence_interval: float=0.80): 
        self._confidence_interval = confidence_interval



class AshraeAdjustedSavingsCalculator(AbstractSavings, ISavingsCalculator): 
    
    
    def save(self, 
        pre: EnergyChangepointEstimator, 
        post: EnergyChangepointEstimator) -> AdjustedSavingsResult:

        adjusted_y = pre.adjust(post)# XXX this is expensive... :/ 
        pre_cvrmse = cvrmse_score(pre.y, pre.pred_y) # XXX this is expensive... :/ 
        pre_p = len(pre.coeffs)
        
        pre_n = pre.len_y 
        post_n = post.len_y 
                 
        gross_adjusted_y = np.sum(adjusted_y)  # annual_adjusted_baseline in old code (?)
        gross_post_y = post.total_y      # annual_measured_reporting in old code(?)

        savings = ashraesavings.adjusted(
            gross_adjusted_y, 
            gross_post_y, 
            pre_cvrmse, 
            pre_p, 
            pre_n, 
            post_n, 
            self._confidence_interval)
        return adjusted_y, *savings




class AshraeNormalizedSavingsCalculator(AbstractSavings, ISavingsCalculator): 
    
    def __init__(self, X_pre: NByOneNDArray, X_post: NByOneNDArray, **kwargs): 
        super().__init__(**kwargs)
        self._X_pre = X_pre 
        self._X_post = X_post


    def save(self, 
        pre: EnergyChangepointEstimator, 
        post: EnergyChangepointEstimator,  
        ) -> NormalizedSavingsResult:  # X is an array of normalized temperature data

        # setup
        normalized_pred_y_pre = pre.predict(self._X_pre)   # XXX expensive :(
        normalized_pred_y_post = post.predict(self._X_post)
        
        gross_normalized_y_pre = np.sum(normalized_pred_y_pre)
        gross_normalized_y_post = np.sum(normalized_pred_y_post)

        pre_cvrmse = cvrmse_score(pre.y, pre.pred_y)  # XXX expensive :( -- NOTE also this is from the original model on the actual X
        post_cvrmse = cvrmse_score(post.y, post.pred_y)

        pre_n = pre.len_y 
        post_n = post.len_y 

        pre_p = len(pre.coeffs)
        post_p = len(post.coeffs)

        n_norm = len(normalized_X)

        savings = ashraesavings.weather_normalized(
            gross_normalized_y_pre, 
            gross_normalized_y_post, 
            pre_cvrmse, 
            post_cvrmse, 
            pre_n, 
            post_n, 
            pre_p, 
            post_p, 
            n_norm,
            self._confidence_interval)
        
        return normalized_X, normalized_pred_y_pre, normalized_pred_y_post, *savings 