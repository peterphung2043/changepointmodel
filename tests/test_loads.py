import pytest
from ashrae.parameter_models import EnergyParameterModelCoefficients, TwoParameterModel
from ashrae.estimator import EnergyChangepointEstimator
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

    l = loads.HeatingLoad()
    l(X, pred_y, slope, yint, changepoint)
    mock.assert_called_once_with(X, pred_y, yint, changepoint)

def test_heatingchangepointmodelload_pos_slope_returns_zero(): 

    X = np.array([1.,])
    pred_y = np.array([1.,])
    slope = 1
    yint = 1 
    changepoint = 1 

    l = loads.HeatingLoad()
    assert l(X, pred_y, slope, yint, changepoint) == 0


def test_heatingchangepointmodelload_none_changepoint_passes_pos_inf(mocker): 
    
    mock = mocker.patch('ashrae._lib.loads.heatload')

    X = np.array([1.,])
    pred_y = np.array([1.,])
    slope = -1
    yint = 1 
    changepoint = None 

    l = loads.HeatingLoad()
    l(X, pred_y, slope, yint, changepoint)
    mock.assert_called_once_with(X, pred_y, yint, np.inf)



def test_coolingchangepointmodelload_forwards_call(mocker):

    mock = mocker.patch('ashrae._lib.loads.coolingload')

    X = np.array([1.,])
    pred_y = np.array([1.,])
    slope = 1
    yint = 1 
    changepoint = 1 

    l = loads.CoolingLoad()
    l(X, pred_y, slope, yint, changepoint)
    mock.assert_called_once_with(X, pred_y, yint, changepoint)


def test_coolingchangepointmodelload_neg_slope_returns_zero(): 

    X = np.array([1.,])
    pred_y = np.array([1.,])
    slope = -1
    yint = 1 
    changepoint = 1 

    l = loads.CoolingLoad()
    assert l(X, pred_y, slope, yint, changepoint) == 0


def test_coolingchangepointmodelload_none_changepoint_passes_neg_inf(mocker): 
    
    mock = mocker.patch('ashrae._lib.loads.coolingload')

    X = np.array([1.,])
    pred_y = np.array([1.,])
    slope = 1
    yint = 1 
    changepoint = None 

    l = loads.CoolingLoad()
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


def test_twoparameter_load_handler_calls_correct_methods(mocker, dummy_twopcoefficients): 
    # spy on changepoint 
    class DummyModel: 
        def slope(self, coeffs): 
            return 42 
        def yint(self, coeffs): 
            return 43 

    cl = loads.CoolingLoad()
    hl = loads.HeatingLoad()
    bl = loads.Baseload()

    mockcl = mocker.patch.object(cl, 'calc', return_value=44)
    mockhl = mocker.patch.object(hl, 'calc', return_value=45)
    mockbl = mocker.patch.object(bl, 'calc', return_value=46) 

    X = np.array([1,])
    pred_y = np.array([100,])
    
    handler = loads.TwoParameterLoadHandler(DummyModel(),cooling=mockcl, heating=mockhl, base=mockbl)
    result = handler.run(X, pred_y, dummy_twopcoefficients)

    assert result == loads.Load(46,45,44)
    mockcl.assert_called_once_with(X, pred_y, 42, 43)
    mockhl.assert_called_once_with(X, pred_y, 42, 43)
    mockbl.assert_called_once_with(100, 44, 45)


def test_threeparameter_load_handler_calls_correct_methods(mocker, dummy_threepcoefficients): 
    class DummyModel: 
        def slope(self, coeffs): 
            return 42 
        def yint(self, coeffs): 
            return 43 
        def changepoint(self, coeffs): 
            return 45

    cl = loads.CoolingLoad()
    hl = loads.HeatingLoad()
    bl = loads.Baseload()

    mockcl = mocker.patch.object(cl, 'calc', return_value=44)
    mockhl = mocker.patch.object(hl, 'calc', return_value=45)
    mockbl = mocker.patch.object(bl, 'calc', return_value=46) 

    X = np.array([1,])
    pred_y = np.array([100,])
    
    handler = loads.ThreeParameterLoadHandler(DummyModel(),cooling=mockcl, heating=mockhl, base=mockbl)
    result = handler.run(X, pred_y, dummy_threepcoefficients)

    assert result == loads.Load(46,45,44)
    mockcl.assert_called_once_with(X, pred_y, 42, 43, 45)
    mockhl.assert_called_once_with(X, pred_y, 42, 43, 45)
    mockbl.assert_called_once_with(100, 44, 45) 


def test_fourparameter_load_handler_calls_correct_methods(mocker, dummy_fourpcoefficients): 
    class DummyModel: 
        def left_slope(self, coeffs): 
            return 42
        def right_slope(self, coeffs): 
            return 43
        def yint(self, coeffs): 
            return 44 
        def changepoint(self, coeffs): 
            return 45


    cl = loads.CoolingLoad()
    hl = loads.HeatingLoad()
    bl = loads.Baseload()

    mockcl = mocker.patch.object(cl, 'calc', return_value=44)
    mockhl = mocker.patch.object(hl, 'calc', return_value=45)
    mockbl = mocker.patch.object(bl, 'calc', return_value=46) 

    X = np.array([1,])
    pred_y = np.array([100,])
    
    handler = loads.FourParameterLoadHandler(DummyModel(),cooling=mockcl, heating=mockhl, base=mockbl)
    result = handler.run(X, pred_y, dummy_fourpcoefficients)

    assert result == loads.Load(46,45,44)
    mockcl.assert_called_once_with(X, pred_y, 43, 44, 45)
    mockhl.assert_called_once_with(X, pred_y, 42, 44, 45)
    mockbl.assert_called_once_with(100, 44, 45)  



def test_fiveparameter_load_handler_calls_correct_methods(mocker, dummy_fivepcoefficients): 
    class DummyModel: 
        def left_slope(self, coeffs): 
            return 42
        def right_slope(self, coeffs): 
            return 43
        def yint(self, coeffs): 
            return 44 
        def left_changepoint(self, coeffs): 
            return 45
        def right_changepoint(self, coeffs): 
            return 46

    cl = loads.CoolingLoad()
    hl = loads.HeatingLoad()
    bl = loads.Baseload()

    mockcl = mocker.patch.object(cl, 'calc', return_value=44)
    mockhl = mocker.patch.object(hl, 'calc', return_value=45)
    mockbl = mocker.patch.object(bl, 'calc', return_value=46) 

    X = np.array([1,])
    pred_y = np.array([100,])
    
    handler = loads.FiveParameterLoadHandler(DummyModel(),cooling=mockcl, heating=mockhl, base=mockbl)
    result = handler.run(X, pred_y, dummy_fivepcoefficients)

    assert result == loads.Load(46,45,44)
    mockcl.assert_called_once_with(X, pred_y, 43, 44, 46)
    mockhl.assert_called_once_with(X, pred_y, 42, 44, 45)
    mockbl.assert_called_once_with(100, 44, 45)   


def test_loads_aggregator_raises_typeerror_if_wrong_model(mocker):
    
    class DummyEstimator(object): 
        model = TwoParameterModel()

    class DummyModel:
        pass 
    
    cl = loads.CoolingLoad()
    hl = loads.HeatingLoad()
    bl = loads.Baseload()


    mockcl = mocker.patch.object(cl, 'calc', return_value=44)
    mockhl = mocker.patch.object(hl, 'calc', return_value=45)
    mockbl = mocker.patch.object(bl, 'calc', return_value=46) 

    handler = loads.FiveParameterLoadHandler(DummyModel(),cooling=mockcl, heating=mockhl, base=mockbl)
    
    agg = loads.EnergyChangepointLoadsAggregator(handler)
    with pytest.raises(TypeError):  
        agg.aggregate(DummyEstimator())


def test_loads_aggregator_calls_handler(mocker): 

    Xtest = np.array([[1.,]])
    ytest = np.array([100,])
    coeffs = EnergyParameterModelCoefficients(42, [99], None)

    class DummyEstimator(object): 
        model = TwoParameterModel()
        X = Xtest
        pred_y = ytest

        def parse_coeffs(self): 
            return coeffs

    pmodel = TwoParameterModel()
    cl = loads.CoolingLoad()
    hl = loads.HeatingLoad()
    bl = loads.Baseload()

    handler = loads.FiveParameterLoadHandler(pmodel,cooling=cl, heating=hl, base=bl)
    
    mockhandler = mocker.patch.object(handler, 'run')
    agg = loads.EnergyChangepointLoadsAggregator(handler)
    agg.aggregate(DummyEstimator())

    mockhandler.assert_called_once_with(Xtest, ytest, coeffs)



