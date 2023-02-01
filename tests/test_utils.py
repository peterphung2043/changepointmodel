
from changepointmodel.core import utils as energymodelutils 
from changepointmodel.core.pmodels import EnergyParameterModelCoefficients, FourParameterModel, ParameterModelFunction, FourParameterCoefficientsParser
import numpy as np

def test_argsort_1d(): 
    x = np.array([[5,],[3,],[2,],[1,],[4,]])  # [3 2 1 4 0 ] is sorting index
    y = np.array([5,3,2,1,4])

    _x, _y = energymodelutils.argsort_1d(x, y)

    assert _y.tolist() == [1,2,3,4,5]


def test_parse_coeffs(): 

    m = ParameterModelFunction('dumb', None, None, FourParameterModel(), FourParameterCoefficientsParser())
    res = energymodelutils.parse_coeffs(m, (1, 2,3, 4))
    assert isinstance(res, EnergyParameterModelCoefficients)
