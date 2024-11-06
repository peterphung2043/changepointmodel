# changepointmodel 
--- 
This is a fork of `cunybpl/changepointmodel`. Please see documentation from the source repo for more information and general usage patterns
[![pytest](https://github.com/cunybpl/changepointmodel/actions/workflows/pytest.yaml/badge.svg)](https://github.com/cunybpl/changepointmodel/actions/workflows/pytest.yaml) ![pybadge](./pybadge.svg)


Changepoint modeling, load disaggregation and savings methodologies consistent with ashrae guidelines. 

**Note: None of the changes shown in ["Differences From `cunybpl/changepointmodel`"](#differences-from-cunybplchangepointmodel) have been tested.**

## About 
---

This is a small toolkit for processing building energy data centered around changepoint modeling. Much of our work is based on industry standard methodologies taken from Ashrae. 

We have found over time that while doing these individual calculations is not difficult, doing them at scale in a well defined way can be. Therefore we tried to design a flexible library that can be used in a variety of contexts such as data science, machine learning and batch processing.


## Features 
----

* Loosely coupled, interface oriented and extensible (`changepoint.core`)
* Wrapper classes for `scipy.optimize.curve_fit` that conform to the `scikit-learn` interface. 
* PEP484 complient (from 3.0.0). Overall strong typing approach. 
* Heavily tested and production ready 


The `core` package consists of the lower level machinery and calculation APIs. The `app` package contains higher level components organized into a ready made application for common building energy modeling tasks and batch processing. 

We have heavily documented each module's public interface. Below is a brief outline of what is inside 

## Differences From `cunybpl\changepointmodel`

* `init_guesses.py` derives initial guesses from `X` (outside air temperature data)
* The `AdjustedSavingsResult` object in `changepointmodel.core.savings` from the source repo was

```
@dataclass
class AdjustedSavingsResult(object):
    adjusted_y: npt.NDArray[np.float64]
    total_savings: float
    average_savings: float
    percent_savings: float
    percent_savings_uncertainty: float
```

Now it is

```
@dataclass
class AdjustedSavingsResult(object):
    adjusted_y: npt.NDArray[np.float64]
    total_savings: float
    average_savings: float
    percent_savings: float
    percent_savings_uncertainty: float
    gross_adjusted_pred_y: float
    gross_post_y: float
    fractional_savings: float
    rel_unc: float
    absolute_uncertainty_of_total_savings: float
```
* `_savings` from `AbstractAdjustedSavingsCalculator` now returns a `Tuple` with 9 `float` values instead of the original 4.
* `save` from `AbstractAdjustedSavingsCalculator` now returns an AdjustedSavingsResult with 5 additional values:
    - `gross_adjusted_pred_y`, 
    - `gross_post_y`, 
    - `fractional_savings`, 
    - `rel_unc`, 
    - `absolute_uncertainty_of_total_savings`
- The `ParameterModelFunction` class was rewritten to now require a `p0` (from the `init_guesses.py` module) instance.
- `EnergyChangepointEstimator.fit` was rewritten so the `CurvefitEstimator` instance now uses `p0` in addition to `model_func` and `bounds` from the `ParameterModelFunction` instance. Before, the `CurvefitEstimator` instance embedded in the `EnergyChangepointEstimator.fit` method only used `model_func` and `bounds` which are both embedded in the `ParameterModelFunction` instance.