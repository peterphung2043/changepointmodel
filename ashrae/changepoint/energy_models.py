from typing_extensions import TypeAlias
import numpy as np
from typing import Optional, Tuple
from curvefit_estimator import CurvefitEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from ..frozen import FreezableObject

# subclass this for each model type... this is abstract
# this is lowlevel stuff... should take numpy arrays/primitives and call into a static function libary developed in a different module
# this seperates the implementation from the interface nicely... it gives the option of using Cython/C++ at this level if we want

# XXX should check if all of these methods are able to return a named tuple type... prob it will work with scipy since it will handle a tuple

from typing import NamedTuple 

import functools

#ModelFunctionResponse = NamedTuple('_ModelFunctionResponse', [('X', np.array), ('y',np.array)])
Load = NamedTuple('Load', [('baseload', float), ('heating', float), ('cooling', float)])
Bound = NamedTuple('Bound', [('lower', Tuple[float]), ('upper', Tuple[float])])  # tuple size changes based on n params
Tstat = NamedTuple('Tstat', [('slopes', Tuple[float])])  # There is a tuple of floats for each coefficient 
Score = NamedTuple('Score', [('score', float)])


# XXX more tuples here.
# XXX possible we define each of these as seperate interfaces and then use interface mixin to create types?  

# model agnostic score calculations. These methods aren't static since they might call other functions
# NOTE I'm pretty sure that since these are all interdependent and share the same interface we can streamline this into a Score class and subclass 
# based

# call into numpy
class ScoreCalc(object): 


    @functools.cache
    def r2(self, X: np.array, y: np.array, pred_y: np.array, p: int) -> Score: 
        """r2 score"""
        pass 

    
    @functools.cache
    def se(self, X: np.array, y: np.array, pred_y: np.array, p: int) -> Score:
        """ standard error """
        pass

    
    @functools.cache
    def mse(self, X: np.array, y: np.array, pred_y: np.array, p: int) -> Score:   # p is n params 
        """ mean squared error"""
        pass 

    @functools.cache
    def rmse(self, X: np.array, y: np.array, pred_y: np.array, p: int) -> Score: 
        """ root mean squared error"""
        return np.sqrt(self.mse(y,pred_y,p))


    @functools.cache
    def cvrmse(self,  X: np.array, y: np.array, pred_y: np.array, p: int) -> Score: 
        """ cv root mean squared error"""
        pass


# model dependent calcs... f and bounds should be static to satisfy curve_fit interface (?)
# each changepoint model subclasses this and provides implementations
# call into numpy 
class EnergyChangepointModelCalc(object): 


    @staticmethod 
    def f(X: np.array, *args) -> np.array: 
        # the function we wish to model. Must return  y array for curve_fit
        # NOTE that data possiblyneeds to be reshaped here into long(array) form... this is because sklearn interface only accepts [[],...] for X
        pass


    @staticmethod
    def bounds(X: np.array) -> Optional[Bound]: 
        # we model dependent bounds calculation for curve_fit
        # NOTE that data needs to be 
        pass


    def sort_X_y(self, X: np.array, y: np.array) -> Tuple[np.array]: 
        order = np.argsort(X.squeeze())  # this should work for 1d arrays
        return X[order], y[order]


    def load(self, X: np.array, y: np.array, predicted_y: np.array, total_y: float, total_pred_y: float, *coeffs) -> Load: # see above 
        # XXX this should probably not take 
        # model dependenant load calculation (heating or cooling load)
        # total_predicted_consumption: float, total_actual_consumption: float add these here... 
        pass 


    def tstat(self, X: np.array, y: np.array, predicted_y: np.array, *coeffs) -> Tstat: 
        pass


    # @staticmethod  # XXX keep this?  
    # def shape 


    # @staticmethod  # XXX keep this?  
    # def datapop


# things we need after fit to do things for bpl/bema: 
# Data: from estimator -> X, y, coeffs,
# Scores: ( model ind. ) -> r2, rmse, cvrmse --> XXX these get accessed alot... could make these a cached_property on EnergyChangepointModel
#         ( mdoel dep. ) -> tstat 
#         NOTE These require a call to estimator.predict on the fitted data. These could be in their own method? 
#         NOTE This precludes the use of cross_val_score or any other sklearn API. An interesting way to extend this within this API could be to sublcass EnergyChangepointModel and use train_test_split + cross_val_score from sklearn.model_selection
# Load: (model dep.) 

# def check_fitted(method): 

#     def _wrapper(*args, **kwargs): 

#         this = args[0]
#         if hasattr(this._estimator) and method.__name__ == 'fit': 
#             raise ValueError('EnergyChangepointModel is already fit')
#         return method(*args, **kwargs)

#     return _wrapper


def arg_cache(method): 
    cache = {}
    def _wrapper(*args, **kwargs): 
        pass



class EnergyChangepointModel(object): 
    """A container object for a changepoint model. After a model is fit you can access scores and 
    load calculations via propeties. 
    """

    def __init__(self, model: EnergyChangepointModelCalc, scorer: ScoreCalc=None): 
        self._model = model  
        self._scorer = scorer
    

    # better to remove reshape here and let pydantic handle the reshape in a post validation hook for Xdata ... avoid direct numpy calls by the caller.
    def fit(self, data: CurvefitEstimatorRegressorTargetData, sort_X_y: bool=True, **estimator_kwargs) -> None:
        """This is deisgned to protect the estimator and assure that fitting is idempotent. It sets the _pred_y variable 
        and extracts

        Args:
            data (CurvefitEstimatorInputData): [description]
            reshape (Optional[Tuple[int]], optional): [description]. Defaults to None.
        """
        if sort_X_y:
            X, y = self._model.sort_X_y(data.X, data.y)  # pretty sure this can go here.
        else:
            X, y = data.X, data.y        

        self._estimator = CurvefitEstimator(model_func=self._model.f, bounds=self._model.bounds, **estimator_kwargs)
        self.pred_y = self._estimator.fit(X, y, data.sigma, data.absolute_sigma).predict(data.X)  # call estimator fit  # XXX scipy/skl error in our own here.
        self._coeffs, self._X, self._y = self._estimator.popt_, self._estimator.X_, self._estimator.y_  # NOTE that X and y will be sorted here.


    def predict(self, data: CurvefitEstimatorRegressorData) -> np.array:  
        """Proxy a call to estimator.predict in order to use the model to generate a changepoint model with 
        different X vals on the fit estimator

        Args:
            data (CurvefitEstimatorRegressorData): [description]

        Returns:
            np.array : predicted y values
        """
        if self._estimator is None:
            raise ValueError('fit must be called before you can call predict')    
        return self._estimator.predict(data.X)  # NOTE shouildn't need to explicitly sort here.


    # Probably better to make these their own class that excepts a fit estimator? 
    @functools.cached_property
    def load(self) -> Load: 
        """ Generate a load response from the fit model

        Raises:
            TypeError: [description]

        Returns:
            Load : The response tuple for the load on the fit changepoint model
        """
        return self._model.load(self.X, self.pred_y, self.total_pred_y, self.total_y, *self.coeffs) # TODO 


    @functools.cached_property
    def tstat(self) -> Tstat: 
        """ Generate the Tstat for this model
        Raises:
            TypeError: [description]

        Returns:
            Tstat: [description]
        """
        return self._model.tstat()  # TODO


    @property 
    def X(self):
        return self._X


    @property 
    def y(self):
        return self._y

    @property 
    def coeffs(self): # tuple
        return self._coeffs


    @functools.cached_property 
    def total_pred_y(self): 
        return sum(self._pred_y)


    @functools.cached_property 
    def total_y(self): 
        return sum(self._y)

    
    def get_score(self, key: str) -> Score: 
        if self._scorer is not None: 
            try:
                scorer = getattr(self._score, 'key')
            except AttributeError: 
                return 
            return scorer.score(X=self.X, y=self.y, pred_y=self.pred_y, p=len(self.coeffs))


    # @functools.cached_property 
    # def r2(self): 
    #     return self._score.r2()  # TODO


    # @functools.cached_property 
    # def mse(self): 
    #     return self._score.mse() # TODO 


    # @functools.cached_property  
    # def rmse(self): 
    #     return self._score.rmse() # TODO


    # @functools.cached_property 
    # def cvrmse(self): 
    #     return self._score.cvrmse() # TODO



