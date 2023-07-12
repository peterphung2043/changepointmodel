"""Defines types to use as inputs to CurvefitEstimator. This is agnostic of function since 
NumpyArray can be n-dimensional
"""
import numpy as np
from typing import Any, Dict, Optional, Union

import pydantic
from .nptypes import (
    AnyByAnyNDArrayField,
    OneDimNDArrayField,
    AnyByAnyNDArray,
    ArgSortRetType,
)

from .utils import argsort_1d_idx


class NpConfig:
    json_encoders = {np.ndarray: lambda v: v.tolist()}


class CurvefitEstimatorDataModel(pydantic.BaseModel):
    X: Union[OneDimNDArrayField, AnyByAnyNDArrayField]
    y: OneDimNDArrayField
    sigma: Optional[OneDimNDArrayField] = None
    absolute_sigma: Optional[bool] = None

    @pydantic.validator("X")
    def validate_X(cls, v: AnyByAnyNDArray[np.float64]) -> AnyByAnyNDArray[np.float64]:
        if v.ndim == 1:  # assure 1d is reshaped according skl spec
            return v.reshape(-1, 1)
        # assure that anything else is at least 2d .. NOTE will not check for nested data... just know what your doing...
        return np.atleast_2d(v)

    @pydantic.root_validator
    def validate_all(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values["y"] is None and values["sigma"] is not None:
            raise ValueError("Cannot pass `sigma` without `y`")

        if (
            values["y"] is not None
        ):  # if we are only passing X then we can skip validation
            xlen = len(values["X"])
            ylen = len(values["y"])

            if values["sigma"] is None:
                if xlen != ylen:
                    raise ValueError("X and y lengths do not match.")
            else:
                siglen = len(values["sigma"])
                if not xlen == ylen == siglen:
                    raise ValueError("X, y and sigma lengths to not match.")

        return values

    def sorted_X_y(self) -> ArgSortRetType:
        """Helper to sort y in terms of X together for changepoint modeling.

        Returns:
            ArgSortRetType: The reordered X and y plus the original index needed to reverse the sort if needed
        """
        return argsort_1d_idx(self.X, self.y)

    class Config(NpConfig):
        ...
