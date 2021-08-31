from ashrae.estimator import EnergyChangepointEstimator
from tests.conftest import loads_dummyenergycoefficients, loads_dummyenergyparametermodel
from ashrae.nptypes import OneDimNDArray
from ashrae import loads 
import numpy as np 


def test_heatingchangepointmodelload_correctly_forwards_call(mocker): 
    mock = mocker.patch('ashrae._lib.loads.heatload')

    X = np.array([1.,])
    pred_y = np.array([1.,])
    slope = -1
    yint = 1 
    changepoint = 1 

    l = loads.HeatingEnergyChangpointModelLoad()
    l(X, pred_y, slope, yint, changepoint)
    mock.assert_called_once_with(X, pred_y, yint, changepoint)

def test_heatingchangepointmodelload_pos_slope_returns_zero(): 

    X = np.array([1.,])
    pred_y = np.array([1.,])
    slope = 1
    yint = 1 
    changepoint = 1 

    l = loads.HeatingEnergyChangpointModelLoad()
    assert l(X, pred_y, slope, yint, changepoint) == 0


def test_heatingchangepointmodelload_none_changepoint_passes_pos_inf(mocker): 
    
    mock = mocker.patch('ashrae._lib.loads.heatload')

    X = np.array([1.,])
    pred_y = np.array([1.,])
    slope = -1
    yint = 1 
    changepoint = None 

    l = loads.HeatingEnergyChangpointModelLoad()
    l(X, pred_y, slope, yint, changepoint)
    mock.assert_called_once_with(X, pred_y, yint, np.inf)



def test_coolingchangepointmodelload_forwards_call(mocker):

    mock = mocker.patch('ashrae._lib.loads.coolingload')

    X = np.array([1.,])
    pred_y = np.array([1.,])
    slope = 1
    yint = 1 
    changepoint = 1 

    l = loads.CoolingEnergyChangepointModelLoad()
    l(X, pred_y, slope, yint, changepoint)
    mock.assert_called_once_with(X, pred_y, yint, changepoint)


def test_coolingchangepointmodelload_neg_slope_returns_zero(): 

    X = np.array([1.,])
    pred_y = np.array([1.,])
    slope = -1
    yint = 1 
    changepoint = 1 

    l = loads.CoolingEnergyChangepointModelLoad()
    assert l(X, pred_y, slope, yint, changepoint) == 0


def test_coolingchangepointmodelload_none_changepoint_passes_neg_inf(mocker): 
    
    mock = mocker.patch('ashrae._lib.loads.coolingload')

    X = np.array([1.,])
    pred_y = np.array([1.,])
    slope = 1
    yint = 1 
    changepoint = None 

    l = loads.CoolingEnergyChangepointModelLoad()
    l(X, pred_y, slope, yint, changepoint)
    mock.assert_called_once_with(X, pred_y, yint, -np.inf)


def test_baseload_forwards_call(mocker): 
    mock = mocker.patch('ashrae._lib.loads.baseload')

    tc = 42  
    hl = 43
    cl = 44

    l = loads.Baseload()
    l(tc, hl, cl)
    mock.assert_called_once_with(tc, hl, cl)


def test_coolingchangepointmodelload_get_slope_calls_correct_model_method(
    loads_dummyenergyparametermodel, 
    loads_dummyenergycoefficients): 
    
    l = loads.CoolingEnergyChangepointModelLoad() 
    slope = l.get_slope(loads_dummyenergyparametermodel, loads_dummyenergycoefficients)
    assert slope == 42 

def test_coolingchangepointmodelload_get_changepoint_calls_correct_model_method(
    loads_dummyenergyparametermodel, 
    loads_dummyenergycoefficients):

    l = loads.CoolingEnergyChangepointModelLoad()
    cp = l.get_changepoint(loads_dummyenergyparametermodel, loads_dummyenergycoefficients)
    assert cp == 43


def test_heatingchangepointmodelload_get_slope_calls_correct_model_method(
    loads_dummyenergyparametermodel, 
    loads_dummyenergycoefficients): 
    
    l = loads.HeatingEnergyChangpointModelLoad()
    slope = l.get_slope(loads_dummyenergyparametermodel, loads_dummyenergycoefficients) 
    assert slope == 44


def test_heatingchangepointmodelload_get_changepoint_calls_correct_model_method(
    loads_dummyenergyparametermodel, 
    loads_dummyenergycoefficients): 

    l = loads.HeatingEnergyChangpointModelLoad()
    cp = l.get_changepoint(loads_dummyenergyparametermodel, loads_dummyenergycoefficients)
    assert cp == 45


def test_energychangepointloadhandler_makes_correct_calls(
    mocker, 
    loads_dummyenergyparametermodel, 
    loads_dummyenergycoefficients):  
    
    load = loads.CoolingEnergyChangepointModelLoad()

    slope_spy = mocker.spy(load, 'get_slope')
    cp_spy = mocker.spy(load, 'get_changepoint')
    calc_spy = mocker.spy(load, 'calc')
    yintspy = mocker.spy(loads_dummyenergyparametermodel, 'yint')

    handler = loads.EnergyChangepointLoadHandler(load)

    X = np.array([1.,])
    pred_y = np.array([1.,])

    res = handler.calc(X, pred_y, loads_dummyenergyparametermodel, loads_dummyenergycoefficients)
    
    assert slope_spy.spy_return == 42
    assert cp_spy.spy_return == 43 
    assert yintspy.spy_return == 99 

    calc_spy.assert_called_with(X, pred_y, 42, 99, 43)

def test_energychangepointloadsaggregator_builds_result(mocker, loads_dummyestimator): 
    
    cload = loads.CoolingEnergyChangepointModelLoad()
    hload = loads.HeatingEnergyChangpointModelLoad()
    bload = loads.Baseload()

    clhandler = loads.EnergyChangepointLoadHandler(cload)
    hlhandler = loads.EnergyChangepointLoadHandler(hload)
    
    agg = loads.EnergyChangepointLoadsAggregator(hlhandler, clhandler, bload)

    mocker.patch.object(hload, 'calc', return_value=42)
    mocker.patch.object(cload, 'calc', return_value=43)
    mocker.patch.object(bload, 'calc', return_value=44) 

    print(loads_dummyestimator.model)

    res = agg.aggregate(loads_dummyestimator)
    
    assert res.base == 44 
    assert res.cooling == 43 
    assert res.heating == 42