import logging
from typing import List, Optional
from changepointmodel.utils import parse_coeffs
from ...models.cpmodel import EnergyChangepointModelResult, AdjustedSavingsResultData,\
    SavingsResult, NormalizedSavingsResultData, AdjustedSavingsResult
from changepointmodel.estimator import EnergyChangepointEstimator
from changepointmodel.loads import EnergyChangepointLoadsAggregator
from changepointmodel.scoring import Scorer
from changepointmodel.schemas import CurvefitEstimatorDataModel

from changepointmodel.predsum import PredictedSumCalculator
from changepointmodel.savings import AshraeAdjustedSavingsCalculator, AshraeNormalizedSavingsCalculator

from .base import BemaChangepointResultContainer
from changepointmodel import nptypes
import numpy as np


def unargsort_1d_idx(arr: nptypes.AnyByAnyNDArrayField, original_order: List[int]) -> nptypes.OneDimNDArrayField:
    """flattens and resorts numpy array back to its original order.

    Args:
        arr (nptypes.AnyByAnyNDArrayField): _description_
        original_order (List[int]): _description_

    Returns:
        nptypes.OneDimNDArrayField: _description_
    """
    out = arr.flatten()  # this would flatten X (oat)
    unsort_index = np.argsort(original_order)
    return out[unsort_index]


class BemaChangepointResult(object): 

    def create(self, 
        estimator: EnergyChangepointEstimator, 
        loads: EnergyChangepointLoadsAggregator, 
        scorer: Scorer, 
        nac: PredictedSumCalculator): 

        original_ordering = estimator._original_ordering # XXX is this access safe?
        data = {
            'name': estimator.name,
            'input_data': {
                'X': unargsort_1d_idx(estimator.X, original_ordering),           # becomes 1 dimensional 
                'y': unargsort_1d_idx(estimator.y, original_ordering), 
                # 'sigma': estimator.sigma, 
                # 'absolute_sigma': estimator.absolute_sigma
            }, 

            # XXX some of these are dataclasses and doesn't integrate well with pydantic, .___dict___ for now
            'coeffs': vars(parse_coeffs(estimator.model, estimator.coeffs)), 
            'pred_y': unargsort_1d_idx(estimator.pred_y, original_ordering), 
            'load': vars(loads.aggregate(estimator)),
            'scores': [vars(i) for i in scorer.check(estimator)],  
            'nac': vars(nac.calculate(estimator)) if nac else None
        }

        return EnergyChangepointModelResult(**data)


class BemaSavingsResult(object): 

    def create(self, 
        pre: BemaChangepointResultContainer, 
        post: BemaChangepointResultContainer, 
        adjcalc: AshraeAdjustedSavingsCalculator, 
        normcalc: AshraeNormalizedSavingsCalculator=None):


        # XXX How else? we have to modify the state of the result after drilling into post.estimator
        result = adjcalc.save(pre.estimator, post.estimator)
        ordering = post.estimator._original_ordering 
        result.adjusted_y = unargsort_1d_idx(result.adjusted_y, ordering)

        adj = AdjustedSavingsResultData(
            result=vars(result), 
            confidence_interval=adjcalc.confidence_interval)
        
        #XXX this is only calculated if norms were provided... otherwise it returns null
        if normcalc: 
            result = normcalc.save(pre.estimator, post.estimator)
            norm = NormalizedSavingsResultData( 
                confidence_interval=normcalc.confidence_interval, 
                result=vars(result))
        else: 
            norm = None 

        data = {
            'pre': pre.result, 
            'post': post.result, 
            'adjusted_savings': adj, 
            'normalized_savings': norm
        }
        return SavingsResult(**data)

