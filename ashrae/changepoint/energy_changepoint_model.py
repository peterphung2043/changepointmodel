from typing_extensions import TypeAlias
import numpy as np
from nptyping import NDArray
from typing import Optional, Tuple
from curvefit_estimator import CurvefitEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from typing import NamedTuple 
from .energy_parameter_model import EnergyParameterModel, Load, Tstat
from .schemas import CurvefitEstimatorData
import functools



# TODO... some mechanism to freeze this object
class EnergyChangepointModel(object): 
    """A container object for a changepoint model. After a model is fit you can access scores and 
    load calculations via propeties. 
    """

    def __init__(self, model: EnergyParameterModel): 
        self._model = model  
        self._estimator = None  # set with call to fit

    
    def fit(self, data: CurvefitEstimatorData, sort: bool=True, **estimator_kwargs) -> None:
        """This is deisgned to protect the estimator and assure that fitting is idempotent. It sets the _pred_y variable 
        and extracts

        Args:
            data (CurvefitEstimatorInputData): [description]
            reshape (Optional[Tuple[int]], optional): [description]. Defaults to None.
        """
        if sort:
            X, y = self._model.sort(data.X, data.y)  # pretty sure this can go here.
        else:
            X, y = data.X, data.y        

        self._estimator = CurvefitEstimator(model_func=self._model.f, bounds=self._model.bounds, **estimator_kwargs)
        self._pred_y = self._estimator.fit(X, y, data.sigma, data.absolute_sigma).predict(data.X)  # call estimator fit  # XXX scipy/skl error in our own here.
        self._coeffs, self._X, self._y = self._estimator.popt_, self._estimator.X_, self._estimator.y_  # NOTE that X and y will be sorted here.


    def predict(self, data: CurvefitEstimatorData) -> NDArray[float]:  
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


    def adjust(self, other: 'EnergyChangepointModel', **kwargs):
        """ creates an adjusted prediction along with ashrae formulas... """
        pass 


    @functools.cached_property
    def score(self, scorer: Scorer) -> List[Score]: 
        return scorer.score(self.y, self.pred_y)  # this runs all the scores


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


    @property 
    def X(self):
        return self._X


    @property 
    def y(self):
        return self._y

    @property 
    def coeffs(self): # tuple
        return self._coeffs


    @property 
    def pred_y(self): 
        return self._pred_y

    @property 
    def n_params(self): 
        return len(self._coeffs)


    @functools.cached_property 
    def total_pred_y(self): 
        return sum(self._pred_y)


    @functools.cached_property 
    def total_y(self): 
        return sum(self._y)

    


