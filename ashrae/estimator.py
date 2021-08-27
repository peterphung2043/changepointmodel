from typing import List, NamedTuple, Optional

from ashrae.base import AbstractEnergyParameterModel, Load, NByOneNDArray, OneDimNDArray
from curvefit_estimator import CurvefitEstimator

from .schemas import CurvefitEstimatorData
from .utils import argsort_1d

import functools

from sklearn.base import BaseEstimator, RegressorMixin

# support skl regressor interface -> need to test interop with GridsearchCV and cross val score. 
# class methods support working without cross validation 


class EnergyChangepointEstimator(BaseEstimator, RegressorMixin): 
    """A container object for a changepoint model. After a model is fit you can access scores and 
    load calculations via propeties. 
    """

    def __init__(self, model: Optional[AbstractEnergyParameterModel]=None): 
        self.model = model  

    @classmethod
    def _sort_input_data(cls, data: CurvefitEstimatorData, sort: bool=True): 
        if sort: 
            return argsort_1d(data.X, data.y)
        else:
            return data.X, data.y 


    @classmethod 
    def fit_many(cls, models: List[AbstractEnergyParameterModel], 
        data: CurvefitEstimatorData,
        sort: bool=True, 
        **estimator_kwargs) -> List[Optional['EnergyChangepointEstimator']]:
        """ Fits a list of 
        """ 
        
        X, y = cls._sort_input_data(data, sort)

        fitted = []
        for m in models: 
            est = cls(m)
            try:
                fitted.append( (est.name, est.fit(X, y, **estimator_kwargs),) ) 
            except Exception: # XXX this is bad... need to figure out exactly what to catch here .. prob LinAlgError? or something from skl...
                fitted.append( (m.name, None))
        
        return fitted 


    @classmethod 
    def fit_one(cls, model: AbstractEnergyParameterModel, 
        data: CurvefitEstimatorData,
        sort: bool=True, 
        **estimator_kwargs) -> Optional['EnergyChangepointEstimator']: 
        """Fits a single model using CurvefitEstimatorData as an entry point. Will sort data if needed.

        Args:
            model (AbstractEnergyParameterModel): [description]
            data (CurvefitEstimatorData): [description]
            sort (bool, optional): [description]. Defaults to True.

        Returns:
            Optional[AbstractEnergyParameterModel]: [description]
        """

        X, y = cls._sort_input_data(data, sort)
        est = cls(model)
        try: 
            return est.name, est.fit(X, y, **estimator_kwargs)
        except Exception: # XXX  same as above... Note I can't really combine these methods or we'd be sorting the multiple times for no reason
            return model.name, None 


    def fit(self, 
        X: NByOneNDArray, 
        y: OneDimNDArray, 
        sigma: Optional[OneDimNDArray]=None, 
        absolute_sigma: Optional[bool]=None, 
        **estimator_kwargs) -> 'EnergyChangepointEstimator':
        """This is wrapper around CurvefitEstimator.fit and allows interoperability with sklearn

        Args:
            data (CurvefitEstimatorInputData): [description]
            reshape (Optional[Tuple[int]], optional): [description]. Defaults to None.
        """
        self.estimator_ = CurvefitEstimator(model_func=self._model.f, bounds=self._model.bounds, **estimator_kwargs)
        self.pred_y_ = self.estimator_.fit(X, y, sigma, absolute_sigma).predict(X)  # call estimator fit  # XXX scipy/skl error in our own here.
        self.X_, self.y_ = self.estimator_.X_, self.estimator_.y_  # XXX I think these need to be here for interop with skl cv
        self.sigma_ = sigma 
        self.absolute_sigma_ = absolute_sigma
        
        return self


    def predict(self, X: NByOneNDArray) -> OneDimNDArray:  
        """Proxy a call to estimator.predict in order to use the model to generate a changepoint model with 
        different X vals on the fit estimator. This also allows interoperability with sklearn.

        Args:
            data (CurvefitEstimatorRegressorData): [description]

        Returns:
            np.array : predicted y values
        """  
        return self._estimator.predict(X) 


    def adjust(self, other: 'EnergyChangepointEstimator') -> OneDimNDArray:
        """ A convenience method that predicts using the X values of another EnergyChangepointEstimator. 
        In option-c methodology this other would be the post retrofit model, making this calling instance the pre model.

        Args:
            other (EnergyChangepointEstimator): [description]

        Returns:
            OneDimNDArray: [description]
        """
        return self.predict(other.X)


    @property 
    def name(self): 
        return self.model.name


    @property 
    def X(self):
        return self.X_


    @property 
    def y(self):
        return self.y_

    @property 
    def coeffs(self): # tuple
        return self.estimator_.popt_


    @property 
    def cov(self): 
        return self.estimator_.pcov_


    @property 
    def pred_y(self): 
        return self.pred_y_


    @property 
    def n_params(self): 
        return len(self.coeffs)


    @property 
    def total_pred_y(self): 
        return sum(self.pred_y_)


    @property 
    def total_y(self): 
        return sum(self.estimator_.y_)


    @property 
    def len_y(self): 
        return len(self.estimator_.y_)

    @property
    def sigma(self): 
        return self.sigma_ 
    

    @property 
    def absolute_sigma(self): 
        return self.absolute_sigma_