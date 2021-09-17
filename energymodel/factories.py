"""Some factory methods for configuration and generating result pydantic schemas to and from internals.

Some of the configuration is complicated to produce the vast amount of results we require along with a model. (We used to call this the post-modeling calcs at BPL)
These are here as higher level abstractions that help keep all the library components loosely coupled but still able to work together. While we are take some 
steps to make sure the configuration between the various components are valid it is largely at this time still up to the caller to make sure they are using and 
testing a correct configuration.

These Factory objects do not make any assumptions about workflow and should be wrapped 
in even higher order functions or classes on in application (i.e. an RPC service or batch processing script). 
"""

from energymodel.predsum import PredictedSumCalculator
from .savings import AshraeAdjustedSavingsCalculator, AshraeNormalizedSavingsCalculator
from . import utils

from . import schemas, scoring
from .estimator import EnergyChangepointEstimator
from typing import Optional, Union
from . import loads
from dataclasses import dataclass
from . import pmodels as pmodels

class EnergyModelConfigurationError(TypeError): 
    """ raised if an EnergyModel is constructed with the wrong load_handler"""


@dataclass
class EnergyModel(object):
    """ The purpose of this container object is to keep the correct ParameterModelFunction 
    and LoadHandler in the same place.
    """ 

    model: pmodels.ParameterModelFunction
    load_handler: loads.AbstractLoadHandler

    def create_estimator(self) -> EnergyChangepointEstimator:
        """Spawn a new estimator from the model.

        Returns:
            EnergyChangepointEstimator: An instance 
        """
        return EnergyChangepointEstimator(model=self.model)
    
    def create_load_aggregator(self) -> loads.EnergyChangepointLoadsAggregator: 
        """Convenience method to get a reference to this model's load.
        
        XXX I added this to create a public API since this part of the object might change.

        Returns:
            loads.EnergyChangepointLoadsAggregator: An Aggregator that initializes the handler.
        """
        return loads.EnergyChangepointLoadsAggregator(handler=self.load_handler)


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

        Raises: 
            EnergyModelConfigurationError(TypeError): If the parameter model and load_handler values are incorrect.

        Returns:
            EnergyModel: [description]
        """
        # XXX maybe should try to figure out a way to make sure this config is valid.
        # These internals to construct ParameterModelFunction simply are not aware of each other at the moment so it is difficult. 

        model = pmodels.ParameterModelFunction(name, f, b, parameter_model, parser)
        
        if not isinstance(model.parameter_model, load_handler.model.__class__):  # we can at least check this one so we do... this is similar to loads.py
            raise EnergyModelConfigurationError(
                f'parameter_model and load_handler models do not match: {parameter_model.__class__}, {load_handler.model.__class__}') 
        
        return EnergyModel(model=model, load_handler=load_handler)



class EnergyChangepointModelResultFactory(object): 

    @classmethod
    def create(cls, 
        estimator: EnergyChangepointEstimator, 
        loads: Optional[loads.EnergyChangepointLoadsAggregator]=None, 
        scorer: Optional[scoring.Scorer]=None, 
        nac: Optional[PredictedSumCalculator]=None) -> schemas.EnergyChangepointModelResult: 
        """Constructs a EnergyChangepointModelResult given at least a fit estimator. Optionally will 
        provide scores, nac and loads if given configured instances of their handlers.

        Args:
            estimator (EnergyChangepointEstimator): [description]
            loads (Optional[loads.EnergyChangepointLoadsAggregator], optional): [description]. Defaults to None.
            scorer (Optional[scoring.Scorer], optional): [description]. Defaults to None.
            nac (Optional[PredictedSumCalculator], optional): [description]. Defaults to None.

        Returns:
            schemas.EnergyChangepointModelResult: [description]
        """

        input_data = schemas.CurvefitEstimatorDataModel(
            X=estimator.X, 
            y=estimator.y, 
            sigma=estimator.sigma, 
            absolute_sigma=estimator.absolute_sigma)

        data = {
            'name': estimator.name,
            'input_data': input_data, 
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
        scorer: Optional[scoring.Scorer]=None) -> schemas.SavingsResult: 
        """Creates a SavingsResult given pre and post retrofit models. Designed for usage with the 
        option-c methodology.

        Args:
            pre (EnergyChangepointEstimator): [description]
            post (EnergyChangepointEstimator): [description]
            adjcalc (AshraeAdjustedSavingsCalculator): [description]
            normcalc (Optional[AshraeNormalizedSavingsCalculator], optional): [description]. Defaults to None.
            pre_loads (Optional[loads.EnergyChangepointLoadsAggregator], optional): [description]. Defaults to None.
            post_loads (Optional[loads.EnergyChangepointLoadsAggregator], optional): [description]. Defaults to None.
            scorer (Optional[Scorer], optional): [description]. Defaults to None.

        Returns:
            schemas.SavingsResult: The SavingsResult.
        """

        pre_result = EnergyChangepointModelResultFactory.create(pre, pre_loads, scorer)
        post_result = EnergyChangepointModelResultFactory.create(post, post_loads, scorer)   

        adj = schemas.AdjustedSavingsResultData(
            result=adjcalc.save(pre, post), 
            confidence_interval=adjcalc.confidence_interval)
        
        if normcalc: 
            result = normcalc.save(pre, post)
            norm = schemas.NormalizedSavingsResultData(
                X_pre=normcalc.X_pre, 
                X_post=normcalc.X_post, 
                confidence_interval=normcalc.confidence_interval, 
                result=result)
        else: 
            norm = None 

        data = {
            'pre': pre_result, 
            'post': post_result, 
            'adjusted_savings': adj, 
            'normalized_savings': norm
        }
        return schemas.SavingsResult(**data)
