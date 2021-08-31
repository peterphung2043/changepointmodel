"""Some factory methods for creating pydantic models from EnergyChangepointEstimator 
"""

from typing import Optional

from curvefit_estimator import estimator
from ashrae.schemas import AdjustedEnergyChangepointModelSavingsResult, \
    EnergyChangepointModelResult

from ashrae.loads import EnergyChangepointLoadsAggregator
from ashrae.scoring import Scorer
from ashrae.estimator import EnergyChangepointEstimator
from .savings import AshraeNormalizedSavingsCalculator, \
    AshraeAdjustedSavingsCalculator


def create_energychangepointmodelresult(
    estimator: EnergyChangepointEstimator, 
    scorer: Scorer, 
    loadagg: EnergyChangepointLoadsAggregator,
    ) -> EnergyChangepointModelResult:

    score = scorer.check(estimator)
    load = loadagg.run(estimator)

    # Not sure to add input data here (X, y)
    data = {
        'name': estimator.name, 
        'coeffs': estimator.parse_coeffs(),
        'pred_y': estimator.pred_y, 
        'load': load, 
        'score': score
    }

    return EnergyChangepointModelResult(**data)



def create_adjustedenergychangepointmodelsavingsresult(
    pre: EnergyChangepointEstimator, 
    post: EnergyChangepointEstimator, 
    scorer: Scorer,
    loadagg: EnergyChangepointLoadsAggregator, 
    adjcalc: AshraeAdjustedSavingsCalculator,
    normcalc: Optional[AshraeNormalizedSavingsCalculator]=None
) -> AdjustedEnergyChangepointModelSavingsResult: 

    pre_score = scorer.check(pre)
    pre_load = loadagg.run(pre)

    post_score = scorer.check(post)
    post_load = loadagg.run(post)

    pre_result = create_energychangepointmodelresult(pre, pre_score, pre_load)
    post_result = create_energychangepointmodelresult(post, post_score, post_load)

    adj = adjcalc.save(pre, post)

    if normcalc: 
        norm = normcalc.save(pre, post)
    else: 
        norm = None 

    data = {
        'pre': pre_result, 
        'post': post_result, 
        'adjusted_savings': adj, 
        'normalized_savings': norm
    }

    return AdjustedEnergyChangepointModelSavingsResult(**data)