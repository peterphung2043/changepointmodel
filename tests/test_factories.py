
from ashrae.savings import AdjustedSavingsResult, AshraeAdjustedSavingsCalculator, AshraeNormalizedSavingsCalculator, NormalizedSavingsResult
from ashrae.parameter_models import EnergyParameterModelCoefficients
import numpy as np
from ashrae import factories, schemas, scoring, loads


# factories.EnergyChangepointModelResultFactory

def test_energymodelfactory_returns_configured_modelresult(): 

    class DummyLoadHandler: ...
    class DummyParamModel: ... 
    class DummyParser: ...

    def f(): ... 
    def b(): ... 
    parser = DummyParser()
    parameter_model = DummyParamModel()
    load_handler = DummyLoadHandler()


    res = factories.EnergyModelFactory.create(
        'mymodel', f, b, parser, parameter_model, load_handler) 

    assert res.model.name == 'mymodel' 
    assert res.model.f == f 
    assert res.model.bounds == b 
    assert res.model._coefficients_parser == parser 
    assert res.model._parameter_model == parameter_model 
    assert res.load._handler == load_handler 


def test_energychangepointmodelresultfactory_correctly_configures_schema(mocker): 


    class DummyModel: 
        def parse_coeffs(self, coeffs): ...

    class DummyEstimator: 
        name = 'name'
        X = np.array([[1.,],])
        y = np.array([2.,])
        pred_y = np.array([3.,])
        sigma = None 
        absolute_sigma = None 
        model = DummyModel()
        coeffs = (99, 99,)

    d = DummyEstimator()
    m = mocker.patch('ashrae.utils.parse_coeffs', return_value=EnergyParameterModelCoefficients(99, [99,],None))
    l = loads.EnergyChangepointLoadsAggregator(object())
    mocker.patch.object(l, 'aggregate', return_value=loads.Load(42., 43., 44.))

    s = scoring.Scorer([])
    mocker.patch.object(s, 'check', return_value=[
        scoring.Score('s1', 42., 42.,True), 
        scoring.Score('s2', 43., 43., False)])

    # this should not fail to build
    res = factories.EnergyChangepointModelResultFactory.create(d, l, s)
    assert res.scores is not None 
    assert res.load is not None


def test_savingsresultfactory_correctly_configures_schema(mocker): 

    class DummyModel: 
        def parse_coeffs(self, coeffs): ...

    class DummyEstimator: 
        name = 'name'
        X = np.array([[1.,],])
        y = np.array([2.,])
        pred_y = np.array([3.,])
        sigma = None 
        absolute_sigma = None 
        model = DummyModel()
        coeffs = (99, 99,)

    dpre = DummyEstimator()
    dpost = DummyEstimator()
    m = mocker.patch('ashrae.utils.parse_coeffs', return_value=EnergyParameterModelCoefficients(99, [99,],None))
    l = loads.EnergyChangepointLoadsAggregator(object())
    mocker.patch.object(l, 'aggregate', return_value=loads.Load(42., 43., 44.))

    s = scoring.Scorer([])
    mocker.patch.object(s, 'check', return_value=[
        scoring.Score('s1', 42., 42.,True), 
        scoring.Score('s2', 43., 43., False)])

    adjcalc = AshraeAdjustedSavingsCalculator()
    mocker.patch.object(adjcalc, 'save', return_value=AdjustedSavingsResult(np.array([42.,]), 42., 42., 42., 42.))

    normcalc = AshraeNormalizedSavingsCalculator(X_pre=np.array([[100.],]), X_post=np.array([[200.,]]))
    mocker.patch.object(normcalc, 'save', return_value=NormalizedSavingsResult(np.array([42.,]),np.array([42.,]),42., 42., 42., 42.))
    

    res = factories.SavingsResultFactory.create(dpre, dpost, adjcalc, normcalc)
    res = factories.SavingsResultFactory.create(dpre, dpost, adjcalc, normcalc, l, l, s)