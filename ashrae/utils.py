from typing import Tuple, Union
import numpy as np
from .nptypes import AnyByAnyNDArray, NByOneNDArray, OneDimNDArray


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
