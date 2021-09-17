"""Some factory methods for configuration and generating result pydantic schemas to and from internals.
"""

from energymodel.predsum import PredictedSumCalculator
from .savings import AshraeAdjustedSavingsCalculator, AshraeNormalizedSavingsCalculator
from . import utils

from pydantic.utils import GetterDict
from . import schemas, scoring
from .scoring import Score, Scorer
from .estimator import EnergyChangepointEstimator
from typing import Any, Dict, List, Optional, Union
from . import loads
from dataclasses import dataclass
from . import pmodels as pmodels
from .calc import models

@dataclass
class EnergyModel(object): 
    model: pmodels.ParameterModelFunction
    load: loads.EnergyChangepointLoadsAggregator

    def create_estimator(self) -> EnergyChangepointEstimator:
        """Spawn a new estimator from the model.

        Returns:
            EnergyChangepointEstimator: [description]
        """
        return EnergyChangepointEstimator(model=self.model)
        

class EnergyModelFactory(object): 

    @classmethod 
    def create(cls, 
        name: str, 
        f: pmodels.ModelCallable, 
        b: Union[pmodels.BoundCallable, pmodels.Bound], 
        parser: pmodels.ICoefficientParser, 
        parameter_model: pmodels.EnergyParameterModel, 
        load_handler: loads.AbstractLoadHandler) -> EnergyModel:
        """Construct an model and a loads factory simultaneously. 
        Creates a convenient container object which will help keep model dependent 
        calculations together within more complicated workflows. 

        Args:
            name (str): [description]
            f (pmodels.ModelCallable): [description]
            b (Union[pmodels.BoundCallable, pmodels.Bound]): [description]
            parser (pmodels.ICoefficientParser): [description]
            parameter_model (pmodels.EnergyParameterModel): [description]
            load_handler (loads.AbstractLoadHandler): [description]

        Returns:
            EnergyModel: [description]
        """

        model = pmodels.ParameterModelFunction(name, f, b, parameter_model, parser)
        load = loads.EnergyChangepointLoadsAggregator(load_handler)

        return EnergyModel(model=model, load=load)



class EnergyChangepointModelResultFactory(object): 

    @classmethod
    def create(cls, 
        estimator: EnergyChangepointEstimator, 
        loads: Optional[loads.EnergyChangepointLoadsAggregator]=None, 
        scorer: Optional[scoring.Scorer]=None, 
        nac: Optional[PredictedSumCalculator]=None) -> schemas.EnergyChangepointModelResult: 

        data = {
            'name': estimator.name,
            'input_data': {
                'X': estimator.X, 
                'y': estimator.y, 
                'sigma': estimator.sigma,
                'absolute_sigma': estimator.absolute_sigma 
            }, 
            'coeffs': utils.parse_coeffs(estimator.model, estimator.coeffs), 
            'pred_y': estimator.pred_y, 
            'load': loads.aggregate(estimator) if loads else None, 
            'scores': scorer.check(estimator) if scorer else None, 
            'nac': nac.calculate(estimator) if nac else None
        }

        return schemas.EnergyChangepointModelResult(**data)


class SavingsResultFactory(object): 

    @classmethod 
    def create(cls, 
        pre: EnergyChangepointEstimator, 
        post: EnergyChangepointEstimator, 
        adjcalc: AshraeAdjustedSavingsCalculator, 
        normcalc: Optional[AshraeNormalizedSavingsCalculator]=None, 
        pre_loads: Optional[loads.EnergyChangepointLoadsAggregator]=None, 
        post_loads: Optional[loads.EnergyChangepointLoadsAggregator]=None, 
        scorer: Optional[Scorer]=None) -> schemas.SavingsResult: 

        pre_result = EnergyChangepointModelResultFactory.create(pre, pre_loads, scorer)
        post_result = EnergyChangepointModelResultFactory.create(post, post_loads, scorer)   

        adj = {
            'result': adjcalc.save(pre, post), 
            'confidence_interval': adjcalc.confidence_interval
        }
        if normcalc: 
            result = normcalc.save(pre, post)
            norm = {
                'X_pre': normcalc.X_pre, 
                'X_post': normcalc.X_post,
                'confidence_interval': normcalc.confidence_interval, 
                'result': result
            }
        else: 
            norm = None 

        data = {
            'pre': pre_result, 
            'post': post_result, 
            'adjusted_savings': adj, 
            'normalized_savings': norm
        }
        return schemas.SavingsResult(**data)
