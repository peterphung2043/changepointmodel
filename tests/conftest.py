from ashrae.energy_parameter_models import AbstractEnergyParameterModel, EnergyParameterModelCoefficients, IChangepointModelFunction
from ashrae import estimator, scoring
import pytest 
import numpy as np

class DummyEnergyParameterModel(AbstractEnergyParameterModel): 

    def parse_coeffs(self, coeffs): 
        return EnergyParameterModelCoefficients(99, [99], None)

    def cooling_slope(self, coeffs): 
        return 42 
    
    def cooling_changepoint(self, coeffs): 
        return 43

    def heating_slope(self, coeffs): 
        return 44 

    def heating_changepoint(self, coeffs): 
        return 45 


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


@pytest.fixture 
def loads_dummyenergyparametermodel(): 
    return DummyEnergyParameterModel()


@pytest.fixture 
def loads_dummyenergycoefficients(): 

    return EnergyParameterModelCoefficients(99, [99], None)


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
def estimator_dummymodel(): 

    class LinearModel(IChangepointModelFunction):
        # pulled this from twop 
 
        def f(self): 
            def line(X, yint, m): 
                return (m * X + yint).squeeze()
            return line
        
        def bounds(self): 
            return ((0, -np.inf),(np.inf, np.inf))
    
    return LinearModel()
