
from ashrae import utils as ashraeutils 
import numpy as np

def test_argsort_1d(): 
    x = np.array([[5,],[3,],[2,],[1,],[4,]])  # [3 2 1 4 0 ] is sorting index
    y = np.array([5,3,2,1,4])

    _x, _y = ashraeutils.argsort_1d(x, y)

    assert _y.tolist() == [1,2,3,4,5]