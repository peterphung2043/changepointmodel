""" APIs for handling adjusted and normalized savings as per ashrae.
"""

import numpy as np
from ashrae.base import NByOneNDArray
from typing import NamedTuple, Optional
from ashrae.estimator import EnergyChangepointEstimator
from ._lib import savings as ashraesavings

from .scoring import Cvrmse
cvrmse_score = Cvrmse()

import abc 

# XXX after building this out I think it might need to be decoupled... 

# The calling scripts below should take numerical data. 
# We can create a factory method for `normalized` and `adjusted` that can act as a wrapper to clean up the public facing API 
# This would allow to return an object that has both adjusted y data and x data. 



SavingsResult = NamedTuple('SavingsResult', [
    ('total_savings', float), 
    ('average_monthly_savings', float), 
    ('percent_savings', float), 
    ('percent_savings_uncertainty', float)])


class IAdjusted(abc.ABC): 
    
    def save(self, 
        pre: EnergyChangepointEstimator, 
        post: EnergyChangepointEstimator) -> SavingsResult: ...
    

class INormalized(abc.ABC): 

    def save(self, 
        pre: EnergyChangepointEstimator, 
        post: EnergyChangepointEstimator,  
        X: NByOneNDArray) -> SavingsResult: ...


class AbstractSavings(abc.ABC):

    def __init__(self, confidence_interval: float=0.80): 
        self._confidence_interval = confidence_interval



class AshraeAdjustedSavings(AbstractSavings, IAdjusted): 
    
    
    def save(self, 
        pre: EnergyChangepointEstimator, 
        post: EnergyChangepointEstimator) -> SavingsResult:

        adjusted_y = pre.adjust(post)# XXX this is expensive... :/ 
        pre_cvrmse = cvrmse_score(pre.y, pre.pred_y) # XXX this is expensive... :/ 
        pre_p = len(pre.coeffs)
        
        pre_n = pre.len_y 
        post_n = post.len_y 
                 
        gross_adjusted_y = np.sum(adjusted_y)  # annual_adjusted_baseline in old code (?)
        gross_post_y = post.total_y      # annual_measured_reporting in old code(?)

        return ashraesavings.adjusted(
            gross_adjusted_y, 
            gross_post_y, 
            pre_cvrmse, 
            pre_p, 
            pre_n, 
            post_n, 
            self._confidence_interval)

        

# XXX for this to work the data has to be perfect... Xpre and Xpost must be somehow joined to ydata... how to do this without adding datetime objs?
class AshraeNormalizedSavings(AbstractSavings, INormalized): 
    

    def save(self, 
        pre: EnergyChangepointEstimator, 
        post: EnergyChangepointEstimator,  
        X: NByOneNDArray) -> SavingsResult:  # X is an array of normalized temperature data

        # setup
        normalized_y_pre = pre.predict(X)   # XXX expensive :(
        normalized_y_post = post.predict(X)
        
        gross_normalized_y_pre = np.sum(normalized_y_pre)
        gross_normalized_y_post = np.sum(normalized_y_post)

        pre_cvrmse = cvrmse_score(pre.y, pre.pred_y)  # XXX expensive :(
        post_cvrmse = cvrmse_score(post.y, post.pred_y)

        pre_n = pre.len_y 
        post_n = post.len_y 

        pre_p = len(pre.coeffs)
        post_p = len(post.coeffs)

        n_norm = len(X)

        return ashraesavings.weather_normalized(
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
