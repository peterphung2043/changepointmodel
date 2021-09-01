from ashrae.parameter_models import ModelFunction
from curvefit_estimator.estimator import CurvefitEstimator
import numpy as np
from numpy.testing import assert_array_almost_equal
from ashrae.estimator import EnergyChangepointEstimator 
import pytest 

from curvefit_estimator import CurvefitEstimator
from numpy.testing import assert_array_almost_equal

def test_energychangepointestimator_fit_calls_curvefitestimator_fit(mocker): 

    def line(X, yint, m): 
        return (m * X + yint).squeeze()

    bounds = ((0, -np.inf),(np.inf, np.inf))

    mymodel = ModelFunction('line', line, bounds, object(), object()) 
    mocker.spy(CurvefitEstimator, 'fit')

    est = EnergyChangepointEstimator(model=mymodel)

    X = np.linspace(1,10,10).reshape(-1, 1)
    y = np.linspace(1,10,10)

    est.fit(X, y)
    assert_array_almost_equal(est.pred_y, y, decimal=1)


