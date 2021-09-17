from energymodel import schemas 

import numpy as np
import pytest
import pydantic

def test_curvefitestimatordatamodel_handles_1d_xdata(): 

    xdata = np.array([1., 2., 3., 4., 5.])
    schemas.CurvefitEstimatorDataModel(X=xdata)

    
def test_curvefittestimatordatamodel_handles_2d_xdata(): 

    xdata = np.array([[1.], [2.], [3.], [4.], [5.]])
    schemas.CurvefitEstimatorDataModel(X=xdata)


def test_curvefittestimatordatamodel_transforms_1d_xdata_to_2d_xdata(): 

    xdata = np.array([1., 2., 3., 4., 5.])
    d = schemas.CurvefitEstimatorDataModel(X=xdata) 
    
    test = np.array([[1.], [2.], [3.], [4.], [5.]])
    assert test.shape == d.X.shape


def test_curvefitestimatordatamodel_handles_optional_data(): 

    xdata = np.array([1., 2., 3., 4., 5.])
    ydata = np.array([1., 2., 3., 4., 5.])
    sigma = np.array([1., 2., 3., 4., 5.])

    schemas.CurvefitEstimatorDataModel(X=xdata, y=ydata, sigma=sigma, absolute_sigma=False)


def test_curvefitestimatordatamodel_raises_validationerror_on_len_mismatch(): 

    xdata = np.array([1., 2., 3., 4., 5.])
    ydata = np.array([1., 2., 3., 4.])
    sigma = np.array([1., 2., 3., 4., 5.])

    # check various combos of len mismatch
    with pytest.raises(pydantic.ValidationError): 
        schemas.CurvefitEstimatorDataModel(X=xdata, y=ydata)
    
    with pytest.raises(pydantic.ValidationError): 
        schemas.CurvefitEstimatorDataModel(X=xdata, y=ydata)
    
    with pytest.raises(pydantic.ValidationError): 
        schemas.CurvefitEstimatorDataModel(X=xdata, y=ydata, sigma=sigma)

    with pytest.raises(pydantic.ValidationError): 
        ydata = np.array([1., 2., 3., 4., 5.])
        sigma = np.array([1., 2., 3., 4.])
        schemas.CurvefitEstimatorDataModel(X=xdata, y=ydata, sigma=sigma)
    
    with pytest.raises(pydantic.ValidationError): 
        ydata = np.array([1., 2., 3., 4.])
        sigma = np.array([1., 2., 3.])
        schemas.CurvefitEstimatorDataModel(X=xdata, y=ydata, sigma=sigma)


def test_curvefitestimatordatamodel_raises_validationerror_if_sigma_not_y(): 

    xdata = np.array([1., 2., 3., 4., 5.])
    sigma = np.array([1., 2., 3., 4., 5.])

    with pytest.raises(pydantic.ValidationError): 
        schemas.CurvefitEstimatorDataModel(X=xdata, sigma=sigma)


def test_curvefitestimatordatamodel_returns_valid_json(): 

    xdata = np.array([1., 2., 3., 4., 5.])
    ydata = np.array([1., 2., 3., 4., 5.])
    
    d = schemas.CurvefitEstimatorDataModel(X=xdata, y=ydata)
    d.json()



def test_energychangepointmodelresult_with_required_data(schema_coeffs): 

    data = {
        'name': 'model', 
        'coeffs': schema_coeffs, 
        'pred_y': np.array([1., 2., 3.])
    }

    schemas.EnergyChangepointModelResult(**data)


def test_energychangepointmodelresult_with_optional_data(
    schema_coeffs, 
    schema_load, 
    schema_scores): 

    data = {
        'name': 'model', 
        'coeffs': schema_coeffs, 
        'pred_y': np.array([1., 2., 3.]), 
        'load': schema_load, 
        'scores': schema_scores
    }

    schemas.EnergyChangepointModelResult(**data)



def test_adjustedenergychangepointmodelsavingsresult_with_required_data(
    schema_coeffs, 
    schema_load, 
    schema_scores, 
    schema_adjustedsavings, 
    schema_normalizedsavings): 

    pre = {
        'name': 'model', 
        'coeffs': schema_coeffs, 
        'pred_y': np.array([1., 2., 3.]), 
        'load': schema_load, 
        'scores': schema_scores
    }

    post = {
        'name': 'model', 
        'coeffs': schema_coeffs, 
        'pred_y': np.array([1., 2., 3.]), 
        'load': schema_load, 
        'scores': schema_scores
    }

    data = {
        'pre': pre, 
        'post': post, 
        'adjusted_savings': schema_adjustedsavings
    }

    schemas.SavingsResult(**data)


def test_adjustedenergychangepointmodelsavingsresult_with_non_required_data(
    schema_coeffs, 
    schema_load, 
    schema_scores, 
    schema_adjustedsavings, 
    schema_normalizedsavings): 

    pre = {
        'name': 'model', 
        'coeffs': schema_coeffs, 
        'pred_y': np.array([1., 2., 3.]), 
        'load': schema_load, 
        'scores': schema_scores
    }

    post = {
        'name': 'model', 
        'coeffs': schema_coeffs, 
        'pred_y': np.array([1., 2., 3.]), 
        'load': schema_load, 
        'scores': schema_scores
    }

    data = {
        'pre': pre, 
        'post': post, 
        'adjusted_savings': schema_adjustedsavings, 
        'normalized_savings': schema_normalizedsavings
    }

    schemas.SavingsResult(**data)
