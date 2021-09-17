from energymodel.predsum import PredictedSum, PredictedSumCalculator 
from energymodel.estimator import EnergyChangepointEstimator

import numpy as np


def test_predicted_sum(mocker): 

    est = EnergyChangepointEstimator()
    
    check = np.array([[1.,],[2.,],[3.,],[4.,],])
    mocker.patch.object(est, 'predict', return_value=check) # just return the check from predict and see if its summed
    
    calc = PredictedSumCalculator(check)
    res = calc.calculate(est)
    assert res.value == 10 
