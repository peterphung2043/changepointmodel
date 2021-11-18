from energymodel.pmodels import EnergyParameterModelCoefficients
from energymodel import estimator, scoring
import pytest 
import numpy as np
import json
import random

from . import GENERATED_DATA_ALL_MODELS_FILE, GENERATED_DATA_POST


@pytest.fixture 
def schema_scores(): 
    return [
        {
            'name': 'r2', 
            'value': 0, 
            'threshold': 0, 
            'ok': False
        }, 
        {
            'name': 'cvrmse', 
            'value': 0, 
            'threshold': 0, 
            'ok': False
        },
    ]

@pytest.fixture 
def schema_load(): 
    return {
        'base': 0, 
        'cooling': 0, 
        'heating': 0
    }

@pytest.fixture
def schema_coeffs(): 
    return {
        'yint': 42., 
        'slopes': [1.], 
        'changepoints': None
    }

@pytest.fixture 
def schema_adjustedsavings(): 
    return {
        'confidence_interval': 0.8,
        'result': {
            'adjusted_y': np.array([1., 2., 3.]), 
            'total_savings': 42., 
            'average_savings': 42., 
            'percent_savings': 42.,
            'percent_savings_uncertainty': 42.}
    }

@pytest.fixture 
def schema_normalizedsavings(): 
    return {
        'X_pre': np.array([[1.,]]), 
        'X_post': np.array([[1.,]]),
        'confidence_interval': 0.8,
        'result':{
            'normalized_y_pre': np.array([1., 2., 3.]),
            'normalized_y_post': np.array([1., 2., 3.]),
            'total_savings': 42., 
            'average_savings': 42., 
            'percent_savings': 42., 
            'percent_savings_uncertainty': 42.,
        }
    }


@pytest.fixture 
def score_mock_estimator(): 
    
    class ScoreMockEstimator(estimator.EnergyChangepointEstimator): 
        y_ = np.array([1.,2.,3.])
        pred_y_ = np.array([4.,5.,6.])
    
    return ScoreMockEstimator()
    

@pytest.fixture 
def score_mock_scorefunction(): 
    
    class Dummy(scoring.ScoringFunction):
        def calc(self, y, pred_y, **kwargs): 
            return 42.0 
    
    return Dummy()



@pytest.fixture 
def dummy_twopcoefficients(): 
    return EnergyParameterModelCoefficients(98, [99], None)

@pytest.fixture 
def dummy_threepcoefficients(): 
    return EnergyParameterModelCoefficients(98, [99], 100)

@pytest.fixture 
def dummy_fourpcoefficients(): 
    return EnergyParameterModelCoefficients(98, [99,100], 101)

@pytest.fixture 
def dummy_fivepcoefficients(): 
    return EnergyParameterModelCoefficients(98, [99, 100], [101, 102])


@pytest.fixture 
def loads_dummyestimator(loads_dummyenergyparametermodel): 

    # this is tricky to mock...
    class _estimator(object): 

        popt_ = (99, 99,)
    
    class LoadsDummyEstimator(estimator.EnergyChangepointEstimator):
        estimator_ = _estimator
        X_ = np.array([[1.,],])
        pred_y_ = np.array([1.,])

    return LoadsDummyEstimator(model=loads_dummyenergyparametermodel)


@pytest.fixture
def generated_data_all_models():
    with open(GENERATED_DATA_ALL_MODELS_FILE, 'r') as f:
        return json.load(f)

@pytest.fixture
def generated_data_for_post():
    with open(GENERATED_DATA_POST, 'r') as f:
        return json.load(f)


def _parse_generated_mode_data(data, model_type):
    for i in data:
        if i['model_type'] == model_type:
            return i

@pytest.fixture
def generated_2p_data(generated_data_all_models):
    return _parse_generated_mode_data(generated_data_all_models, '2P')

@pytest.fixture
def generated_3pc_data(generated_data_all_models):
    return _parse_generated_mode_data(generated_data_all_models, '3PC')

@pytest.fixture
def generated_3ph_data(generated_data_all_models):
    return _parse_generated_mode_data(generated_data_all_models, '3PH')

@pytest.fixture
def generated_4p_data(generated_data_all_models):
    return _parse_generated_mode_data(generated_data_all_models, '4P')

@pytest.fixture
def generated_5p_data(generated_data_all_models):
    return _parse_generated_mode_data(generated_data_all_models, '5P')

@pytest.fixture
def generated_2p_post_data(generated_data_for_post):
    return _parse_generated_mode_data(generated_data_for_post, '2P')

@pytest.fixture
def generated_3pc_post_data(generated_data_for_post):
    return _parse_generated_mode_data(generated_data_for_post, '3PC')

@pytest.fixture
def generated_3ph_post_data(generated_data_for_post):
    return _parse_generated_mode_data(generated_data_for_post, '3PH')

@pytest.fixture
def generated_4p_post_data(generated_data_for_post):
    return _parse_generated_mode_data(generated_data_for_post, '4P')

@pytest.fixture
def generated_5p_post_data(generated_data_for_post):
    return _parse_generated_mode_data(generated_data_for_post, '5P')


