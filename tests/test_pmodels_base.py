from changepointmodel.core.pmodels import (
    factories as pmodel_factories,
    parameter_model as pm,
    coeffs_parser as cpar,
    base as pmbase,
)

import numpy as np

from changepointmodel.core.nptypes import OneDimNDArray
from changepointmodel.core.pmodels.base import EnergyParameterModelCoefficients, Load


def test_energyparameter_coefficients_n_params():
    e = pmbase.EnergyParameterModelCoefficients(
        yint=42.0, slopes=[42.0], changepoints=[42.0]
    )

    assert e.n_params() == 3

    e = pmbase.EnergyParameterModelCoefficients(
        yint=42.0, slopes=[42.0], changepoints=[]
    )

    assert e.n_params() == 2

    e = pmbase.EnergyParameterModelCoefficients(
        yint=42.0, slopes=[42.0, 43.0], changepoints=[42.0]
    )

    assert e.n_params() == 4

    e = pmbase.EnergyParameterModelCoefficients(
        yint=42.0, slopes=[42.0, 42.0], changepoints=[42.0, 43.0]
    )

    assert e.n_params() == 5


def test_metrics_mixins_proxy_calls_to_correct_calc(mocker):
    r2_score = mocker.patch("changepointmodel.core.calc.metrics.r2_score")
    rmse = mocker.patch("changepointmodel.core.calc.metrics.rmse")
    cvrmse = mocker.patch("changepointmodel.core.calc.metrics.cvrmse")
    adj_r2 = mocker.patch("changepointmodel.core.calc.metrics.adjusted_r2_score")

    y = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    coeffs = pmbase.EnergyParameterModelCoefficients(
        yint=42.0, slopes=[42.0, 43.0], changepoints=[42.0]
    )

    mix = pmbase.R2MetricMixin()
    mix.r2(y, y_pred)
    assert r2_score.called_once_with(y, y_pred)

    mix = pmbase.CvRmseMetricMixin()
    mix.cvrmse(y, y_pred)
    assert cvrmse.called_once_with(y, y_pred)

    mix = pmbase.RmseMetricMixin()
    mix.rmse(y, y_pred)
    assert rmse.called_once_with(y, y_pred)

    mix = pmbase.AdjR2MetricMixin()
    mix.adjusted_r2(y, y_pred, coeffs)
    assert cvrmse.called_once_with(y, y_pred, 4)


def test_load_interface_private_methods(mocker):
    heat = mocker.patch("changepointmodel.core.calc.loads.heatload")
    cool = mocker.patch("changepointmodel.core.calc.loads.coolingload")
    base = mocker.patch("changepointmodel.core.calc.loads.baseload")

    X = np.array(
        [
            1.0,
        ]
    )
    pred_y = np.array(
        [
            1.0,
        ]
    )
    slope = -1
    yint = 1
    changepoint = 1

    class MockLoad(pmbase.ILoad):
        def load(
            self,
            X: OneDimNDArray[np.float64],
            pred_y: OneDimNDArray[np.float64],
            coeffs: EnergyParameterModelCoefficients,
        ) -> Load:
            raise NotImplementedError("We don't care right here...")

    load = MockLoad()

    load._base(42.0, 43.0, 44.0)
    assert base.called_once_with(42.0, 43.0, 44.0)

    load._heating(X, pred_y, slope, yint, changepoint)
    assert heat.called_once_with(X, pred_y, yint, changepoint)

    load._cooling(X, pred_y, slope, yint, changepoint)
    assert cool.called_once_with(X, pred_y, yint, changepoint)


def test_iload_private_method_branch_arms(mocker):
    heat = mocker.patch("changepointmodel.core.calc.loads.heatload")
    cool = mocker.patch("changepointmodel.core.calc.loads.coolingload")

    X = np.array(
        [
            1.0,
        ]
    )
    pred_y = np.array(
        [
            1.0,
        ]
    )
    yint = 1
    changepoint = 1

    class MockLoad(pmbase.ILoad):
        def load(
            self,
            X: OneDimNDArray[np.float64],
            pred_y: OneDimNDArray[np.float64],
            coeffs: EnergyParameterModelCoefficients,
        ) -> Load:
            raise NotImplementedError("We don't care right here...")

    load = MockLoad()

    ans = load._heating(X, pred_y, 42.0, yint, changepoint)
    assert ans == 0

    ans = load._heating(X, pred_y, -42.0, yint, changepoint)
    assert ans != 0

    ans = load._cooling(X, pred_y, 42.0, yint, changepoint)
    assert ans != 0

    ans = load._cooling(X, pred_y, -42.0, yint, changepoint)
    assert ans == 0

    changepoint = None

    load._cooling(X, pred_y, 42.0, yint, changepoint)
    assert cool.called_once_with(X, pred_y, yint, -np.inf)

    load._heating(X, pred_y, -42.0, yint, changepoint)
    assert heat.called_once_with(X, pred_y, yint, np.inf)
