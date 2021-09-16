from .pmodels import EnergyParameterModelCoefficients, ParameterModelFunction
from typing import Tuple, Union
import numpy as np
from .nptypes import NByOneNDArray, OneDimNDArray


def argsort_1d(X: Union[NByOneNDArray, OneDimNDArray], y: OneDimNDArray) -> Tuple[NByOneNDArray, OneDimNDArray]:
    """ Helper to argsort X and y for 1d changepoint modeling. The output of this data should 
    be used before data is passed into curvefit estimator.

    Args:
        X (NByOneNDArray): [description]
        y (OneDimNDArray): [description]

    Returns:
        Tuple[NByOneNDArray, OneDimNDArray]: [description]
    """
    order = np.argsort(X.squeeze())
    return X[order], y[order]



def parse_coeffs(model: ParameterModelFunction, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients: 
    """Given an ParameterModelFunction and raw coefficients tuple from CurvefitEstimator.fit
    will return an EnerguParameterModelCoefficients accessor object.

    Args:
        model (EnergyParameterModel): [description]
        coeffs (Tuple[float, ...]): [description]

    Returns:
        EnergyParameterModelCoefficients: [description]
    """
    return model.parse_coeffs(coeffs)