# changepointmodel 
--- 
[![pytest](https://github.com/cunybpl/changepointmodel/actions/workflows/pytest.yaml/badge.svg)](https://github.com/cunybpl/changepointmodel/actions/workflows/pytest.yaml) ![pybadge](./pybadge.svg)


Changepoint modeling, load disaggregation and savings methodologies consistent with ashrae guidelines. 


## About 
---

This is a small toolkit for processing building energy data centered around changepoint modeling. Much of our work is based on industry standard methodologies taken from Ashrae. 

We have found over time that while doing these individual calculations is not difficult, doing them at scale in a well defined way can be. Therefore we tried to design a flexible library that can be used in a variety of contexts such as data science, machine learning and batch processing.


## Features 
----

* Loosely coupled, interface oriented and extensible (`changepoint.core`)
* Ready built high level application tooling (`changepoint.app`)
* Wrapper classes for `scipy.optimize.curve_fit` that conform to the `scikit-learn` interface. 
* PEP484 complient (from 3.0.0). Overall strong typing approach. 
* Heavily tested and production ready 


The `core` package consists of the lower level machinery and calculation APIs. The `app` package contains higher level components organized into a ready made application for common building energy modeling tasks and batch processing. 

We have heavily documented each module's public interface. Below is a brief outline of what is inside 

__core__
----
The core APIs are a loosely coupled set of classes that work together to build changepoint models, load aggregations and savings calculations.


* `estimator` - Wrapper classes around `scipy.optimize.curve_fit` with some bells and whistles. 
    * These classes interop with scikit learn and can be incorporated into the sklearn API (`cross_val_score`, `GridSearchCV` etc.) for ML projects.

* `predsum` - Predicted Sum (used for nac calculations)
* `schemas` - Input validation 

_Deprecated from 3.1_

* `loads` - Load calculations for various ParameterModels
* `savings` - High level interfaces for savings calculations.
* `scoring` - Standard statistical reporting for model validation.
* `factories` - Helper classes for combining Estimators and LoadAggregators

__core.calc__ 
----
Model dependent calculations and ashrae savings formulas. 

* `models` - the standard parameter modeling functions. We use these with `scipy.optimize.curve_fit`'s "trf" algorithm that allows for approximate derivitives and bounded search. 
* `bounds` - a standard set of bounds functions to use. We determine these by analyzing the X value input. 
* `metrics` - scoring metrics borrowed from `sklearn` 
* `loads` - loads calculations for base, heating and cooling 
* `savings` - ashrae savings formulas for monthly data 
* `uncertainties` - ashrae uncertainty calculations for savings 

_Added in 3.1_ 
* tstat - perform a tstat on the slopes of your model for statistical significance 
* dpop - return the heating and cooling points from your model based on slope/changepoint 


__core.pmodels__
_Since 3.1_ 

This was moved into its own package. We consider these private but accessible for anyone wishing to extend the library APIs. 

For most cases you will want to simply use the `factories` provided to create an appropriate changepoint model that can be 
used with our EnergyChangepointEstimator class. These and other useful types are exposed in the top level package.


## Example

From 3.1, we went to great lengths to simplfy the use of the core library. 

I will walk through a simple example to show how to estimate a single model and get its load and some statistical scores.

Here is the data we want to model...
```python 

oat = [66.0, 92.0, 98.0, 17.0, 83.0, 57.0, 86.0, 97.0, 96.0, 47.0, 73.0, 32.0]
usage = [834.311, 1992.275, 2304.786, 193.692, 1699.326, 257.449, 1720.430, 2271.0722, 2396.914, 345.639, 1166.869, 225.720]
```







tpc = pmodels.threepc_factory(name='3PC')

estimator = EnergyChangepointmodelEstimator(model=tpc)

sX, sy, _ = argsort_1d_idx(X, y)

est.fit(X, y)   


```





