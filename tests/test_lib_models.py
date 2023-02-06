
from changepointmodel.core.calc import models 
import numpy as np
from numpy.testing import assert_array_equal

def test_twop_model(): 
    X = np.linspace(1, 10, 10)
    X = X.reshape(-1, 1)
    yint = 0 
    slope = 2 
    res = models.twop(X, yint, slope)
    assert_array_equal(np.array([2.,4.,6.,8.,10.,12.,14.,16.,18.,20.]), res)
 

def test_threepc_model(): 
    X = np.linspace(1, 10, 10)
    X = X.reshape(-1, 1)
    yint = 2 
    slope = 2 
    changepoint = 5
    res = models.threepc(X, yint, slope, changepoint)
    assert_array_equal(np.array([2.,2.,2.,2.,2.,4.,6.,8.,10.,12.,]), res)


def test_threeph_model(): 
    X = np.linspace(1, 10, 10)
    X = X.reshape(-1, 1)
    yint = 2 
    slope = -2
    changepoint = 5 
    res = models.threeph(X, yint, slope, changepoint)
    assert_array_equal(np.array([10.,8.,6.,4.,2.,2.,2.,2.,2.,2.,]), res)


def test_fourp_model(): 
    X = np.linspace(1, 10, 10)
    X = X.reshape(-1, 1)
    yint = 2 
    ls = -2
    rs = 2
    changepoint = 5 

    res = models.fourp(X, yint, ls, rs, changepoint)
    assert_array_equal(np.array([10.,8.,6.,4.,2.,4.,6.,8.,10.,12.]), res)

def test_fivep_model(): 
    
    X = np.linspace(1, 10, 10)
    X = X.reshape(-1, 1)
    yint = 2 
    ls = -2
    rs = 2
    lcp = 4
    rcp = 5

    res = models.fivep(X, yint, ls, rs, lcp, rcp) 
    assert_array_equal(np.array([8.,6.,4.,2.,2.,4.,6.,8.,10.,12.]), res)

