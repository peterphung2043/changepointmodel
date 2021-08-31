from ashrae import estimator, scoring
import pytest 
import numpy as np

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
        'adjusted_y': np.array([1., 2., 3.]), 
        'total_savings': 42., 
        'average_monthly_savings': 42., 
        'percent_savings': 42.,
        'percent_savings_uncertainty': 42.
    }

@pytest.fixture 
def schema_normalizedsavings(): 
    return {
        'normalized_y_pre': np.array([1., 2., 3.]),
        'normalized_y_post': np.array([1., 2., 3.]),
        'total_savings': 42., 
        'average_monthly_savings': 42., 
        'percent_savings': 42., 
        'percent_savings_uncertainty': 42.,
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