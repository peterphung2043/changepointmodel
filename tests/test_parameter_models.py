from changepointmodel.core.pmodels import (
    FiveParameterModel,
    FourParameterModel,
    ParameterModelFunction,
    ParameterModelFunction,
    ThreeParameterCoolingModel,
    ThreeParameterHeatingModel,
    TwoParameterModel,
    EnergyParameterModelCoefficients,
)
import numpy as np


from changepointmodel.core.pmodels.base import YinterceptMixin
from changepointmodel.core.pmodels.coeffs_parser import (
    TwoParameterCoefficientParser,
    ThreeParameterCoefficientsParser,
    FiveParameterCoefficientsParser,
    FourParameterCoefficientsParser,
)


def test_yinterceptmixin():
    coeffs = EnergyParameterModelCoefficients(42, [99], [])

    model = YinterceptMixin()
    assert model.yint(coeffs) == 42


def test_twoparametermodel():
    coeffs = EnergyParameterModelCoefficients(42, [99], [])

    model = TwoParameterModel()
    assert model.slope(coeffs) == 99


def test_threeparametercoolingmodel():
    coeffs = EnergyParameterModelCoefficients(42, [99], [43])
    model = ThreeParameterCoolingModel()
    assert model.slope(coeffs) == 99
    assert model.changepoint(coeffs) == 43


def test_threeparameterheatingmodel():
    coeffs = EnergyParameterModelCoefficients(42, [99], [43])
    model = ThreeParameterHeatingModel()
    assert model.slope(coeffs) == 99
    assert model.changepoint(coeffs) == 43


def test_fourparametermodel():
    coeffs = EnergyParameterModelCoefficients(42, [98, 99], [43])
    model = FourParameterModel()
    assert model.right_slope(coeffs) == 99
    assert model.left_slope(coeffs) == 98
    assert model.changepoint(coeffs) == 43


def test_fiveparametermodel():
    coeffs = EnergyParameterModelCoefficients(42, [98, 99], [43, 44])
    model = FiveParameterModel()
    assert model.right_slope(coeffs) == 99
    assert model.left_slope(coeffs) == 98
    assert model.right_changepoint(coeffs) == 44
    assert model.left_changepoint(coeffs) == 43


def test_linear_coefficient_parser():
    test = EnergyParameterModelCoefficients(42, [99], [])
    parser = TwoParameterCoefficientParser()
    assert parser.parse((42, 99)) == test


def test_singleslope_slinglechangepoint_coefficient_parser():
    test = EnergyParameterModelCoefficients(42, [99], [43])
    parser = ThreeParameterCoefficientsParser()
    assert parser.parse((42, 99, 43)) == test


def test_dualslope_singlechangepoint_coefficient_parser():
    test = EnergyParameterModelCoefficients(42, [98, 99], [43])
    parser = FourParameterCoefficientsParser()
    assert parser.parse((42, 98, 99, 43)) == test


def test_dualslope_dualchangepoint_coefficient_parser():
    test = EnergyParameterModelCoefficients(42, [98, 99], [43, 44])
    parser = FiveParameterCoefficientsParser()
    assert parser.parse((42, 98, 99, 43, 44)) == test


def test_modelfunction():
    def f(X, y):
        return (X + y).squeeze()

    bound = (42,), (43,)
    parmeter_model = TwoParameterModel()
    parser = TwoParameterCoefficientParser()

    model = ParameterModelFunction(
        "mymodel",
        f=f,
        bounds=bound,
        parameter_model=parmeter_model,
        coefficients_parser=parser,
    )

    assert model.name == "mymodel"
    assert model.f == f
    assert model.bounds == bound
    assert model.parameter_model == parmeter_model

    assert model.parse_coeffs((42, 99)) == EnergyParameterModelCoefficients(
        42, [99], []
    )


from changepointmodel.core.pmodels import (
    factories as pmodel_factories,
    parameter_model as pm,
    coeffs_parser as cpar,
)
from changepointmodel.core.calc import models as cpmodels, bounds as cpbounds


def test_parameter_model_factories():
    # new for 3.1
    # check these were properly configured

    m = pmodel_factories.twop()

    assert m.name == "2P"
    assert m.f is cpmodels.twop
    assert m.bounds is cpbounds.twop
    assert isinstance(m._parameter_model, pm.TwoParameterModel)
    assert isinstance(m._coefficients_parser, cpar.TwoParameterCoefficientParser)

    m = pmodel_factories.threepc()

    assert m.name == "3PC"
    assert m.f is cpmodels.threepc
    assert m.bounds is cpbounds.threepc
    assert isinstance(m._parameter_model, pm.ThreeParameterCoolingModel)
    assert isinstance(m._coefficients_parser, cpar.ThreeParameterCoefficientsParser)

    m = pmodel_factories.threeph()

    assert m.name == "3PH"
    assert m.f is cpmodels.threeph
    assert m.bounds is cpbounds.threeph
    assert isinstance(m._parameter_model, pm.ThreeParameterHeatingModel)
    assert isinstance(m._coefficients_parser, cpar.ThreeParameterCoefficientsParser)

    m = pmodel_factories.fourp()

    assert m.name == "4P"
    assert m.f is cpmodels.fourp
    assert m.bounds is cpbounds.fourp
    assert isinstance(m._parameter_model, pm.FourParameterModel)
    assert isinstance(m._coefficients_parser, cpar.FourParameterCoefficientsParser)

    m = pmodel_factories.fivep()

    assert m.name == "5P"
    assert m.f is cpmodels.fivep
    assert m.bounds is cpbounds.fivep
    assert isinstance(m._parameter_model, pm.FiveParameterModel)
    assert isinstance(m._coefficients_parser, cpar.FiveParameterCoefficientsParser)
