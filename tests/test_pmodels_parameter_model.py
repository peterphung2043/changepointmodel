from changepointmodel.core.pmodels import (
    factories as pmodel_factories,
    parameter_model as pm,
    coeffs_parser as cpar,
    base as pmbase,
)
import numpy as np


def test_two_parameter_model_correctly_forwards_calls(mocker):
    tstat_twop = mocker.patch("changepointmodel.core.calc.tstat.twop")
    dpop_twop = mocker.patch("changepointmodel.core.calc.dpop.twop")
    heat = mocker.patch("changepointmodel.core.calc.loads.heatload")
    cool = mocker.patch("changepointmodel.core.calc.loads.coolingload")
    base = mocker.patch("changepointmodel.core.calc.loads.baseload")

    X = np.array(
        [
            1.0,
        ]
    )
    y = np.array(
        [
            2.0,
        ]
    )

    pred_y = np.array(
        [
            3.0,
        ]
    )
    yint = 42.0
    slopes = [0.1]

    coeffs = pmbase.EnergyParameterModelCoefficients(yint, slopes)

    model = pm.TwoParameterModel()

    assert 0.1 == model.slope(coeffs)

    model.tstat(X, y, pred_y, coeffs)
    tstat_twop.assert_called_once_with(X, y, pred_y, 0.1)

    model.dpop(X, coeffs)
    dpop_twop.assert_called_once_with(X, 0.1)

    assert model.shape(coeffs) == True

    model.load(X, pred_y, coeffs)
    assert not heat.called
    cool.assert_called_once_with(X, pred_y, 42.0, -np.inf)

    assert base.call_args[0][0] == 3.0


def test_threepc_parameter_model_correctly_forwards_calls(mocker):
    tstat = mocker.patch("changepointmodel.core.calc.tstat.threepc")
    dpop = mocker.patch("changepointmodel.core.calc.dpop.threepc")
    heat = mocker.patch("changepointmodel.core.calc.loads.heatload")
    cool = mocker.patch("changepointmodel.core.calc.loads.coolingload")
    base = mocker.patch("changepointmodel.core.calc.loads.baseload")

    X = np.array(
        [
            1.0,
        ]
    )
    y = np.array(
        [
            2.0,
        ]
    )

    pred_y = np.array(
        [
            3.0,
        ]
    )
    yint = 42.0
    slopes = [0.1]
    changepoints = [0.2]

    coeffs = pmbase.EnergyParameterModelCoefficients(yint, slopes, changepoints)

    model = pm.ThreeParameterCoolingModel()

    assert 0.1 == model.slope(coeffs)
    assert 0.2 == model.changepoint(coeffs)

    model.tstat(X, y, pred_y, coeffs)
    tstat.assert_called_once_with(X, y, pred_y, 0.1, 0.2)

    model.dpop(X, coeffs)
    dpop.assert_called_once_with(X, 0.2)  # usese the slope

    assert model.shape(coeffs) == True

    model.load(X, pred_y, coeffs)
    assert not heat.called
    cool.assert_called_once_with(X, pred_y, 42.0, 0.2)
    assert base.call_args[0][0] == 3.0


def test_threeph_parameter_model_correctly_forwards_calls(mocker):
    tstat = mocker.patch("changepointmodel.core.calc.tstat.threeph")
    dpop = mocker.patch("changepointmodel.core.calc.dpop.threeph")
    heat = mocker.patch("changepointmodel.core.calc.loads.heatload")
    cool = mocker.patch("changepointmodel.core.calc.loads.coolingload")
    base = mocker.patch("changepointmodel.core.calc.loads.baseload")

    X = np.array(
        [
            1.0,
        ]
    )
    y = np.array(
        [
            2.0,
        ]
    )

    pred_y = np.array(
        [
            3.0,
        ]
    )
    yint = 42.0
    slopes = [-0.1]
    changepoints = [0.2]

    coeffs = pmbase.EnergyParameterModelCoefficients(yint, slopes, changepoints)

    model = pm.ThreeParameterHeatingModel()

    assert -0.1 == model.slope(coeffs)
    assert 0.2 == model.changepoint(coeffs)

    model.tstat(X, y, pred_y, coeffs)
    tstat.assert_called_once_with(X, y, pred_y, -0.1, 0.2)

    model.dpop(X, coeffs)
    dpop.assert_called_once_with(X, 0.2)  # usese the slope

    assert model.shape(coeffs) == True

    model.load(X, pred_y, coeffs)
    assert not cool.called
    assert heat.called
    assert base.called


def test_fourp_parameter_model_correctly_forwards_calls(mocker):
    tstat = mocker.patch("changepointmodel.core.calc.tstat.fourp")
    dpop = mocker.patch("changepointmodel.core.calc.dpop.fourp")
    heat = mocker.patch("changepointmodel.core.calc.loads.heatload")
    cool = mocker.patch("changepointmodel.core.calc.loads.coolingload")
    base = mocker.patch("changepointmodel.core.calc.loads.baseload")

    X = np.array(
        [
            1.0,
        ]
    )
    y = np.array(
        [
            2.0,
        ]
    )

    pred_y = np.array(
        [
            3.0,
        ]
    )
    yint = 42.0
    slopes = [-0.1, 0.2]
    changepoints = [0.3]

    coeffs = pmbase.EnergyParameterModelCoefficients(yint, slopes, changepoints)

    model = pm.FourParameterModel()

    assert -0.1 == model.left_slope(coeffs)
    assert 0.2 == model.right_slope(coeffs)

    model.tstat(X, y, pred_y, coeffs)
    tstat.assert_called_once_with(X, y, pred_y, -0.1, 0.2, 0.3)

    model.dpop(X, coeffs)
    dpop.assert_called_once_with(X, 0.3)

    assert model.shape(coeffs) == False

    model.load(X, pred_y, coeffs)
    assert cool.called
    assert heat.called
    assert base.called


def test_fivep_parameter_model_correctly_forwards_calls(mocker):
    tstat = mocker.patch("changepointmodel.core.calc.tstat.fivep")
    dpop = mocker.patch("changepointmodel.core.calc.dpop.fivep")
    heat = mocker.patch("changepointmodel.core.calc.loads.heatload")
    cool = mocker.patch("changepointmodel.core.calc.loads.coolingload")
    base = mocker.patch("changepointmodel.core.calc.loads.baseload")

    X = np.array(
        [
            1.0,
        ]
    )
    y = np.array(
        [
            2.0,
        ]
    )

    pred_y = np.array(
        [
            3.0,
        ]
    )
    yint = 42.0
    slopes = [-0.1, 0.2]
    changepoints = [0.3, 0.4]

    coeffs = pmbase.EnergyParameterModelCoefficients(yint, slopes, changepoints)

    model = pm.FiveParameterModel()

    assert -0.1 == model.left_slope(coeffs)
    assert 0.2 == model.right_slope(coeffs)
    assert 0.3 == model.left_changepoint(coeffs)
    assert 0.4 == model.right_changepoint(coeffs)

    model.tstat(X, y, pred_y, coeffs)
    tstat.assert_called_once_with(X, y, pred_y, -0.1, 0.2, 0.3, 0.4)

    model.dpop(X, coeffs)
    dpop.assert_called_once_with(X, 0.3, 0.4)

    assert model.shape(coeffs) == False

    model.load(X, pred_y, coeffs)
    assert cool.called
    assert heat.called
    assert base.called
