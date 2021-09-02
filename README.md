# ashrae 

Changepoint modeling, load disaggregation and savings methodologies consistent with ashrae guidelines. 

Meant to replace `bema` python library package in future production environments from fy22.


## features 

- timescale agnostic changepoint modeling and load disaggregation 
- ashrae savings methodologies and formulas similar to `bema` package
- an `Estimator` interface this is interoperability with sklearn APIs including `cross_val_score` and `GridSearchCV` 
- `pydantic` models and factory methods for high level schema validation 
- a loosely coupled and extensible API to allow for new non-linear function fitting with `CurvefitEstimator`

### todo 

- [ ] CI github
- [ ] Integration tests for all modules 
- [ ] docstrings + documentation 
- [ ] live testing QA with real data.
- [ ] publishing 