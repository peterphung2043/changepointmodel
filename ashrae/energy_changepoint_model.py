
from ashrae.filtering import ComparableFilter, ComparableFilters
from typing import List, Optional
from ashrae.scorer import Score, Scorer
from ashrae.base import AbstractEnergyParameterModel, Load
from curvefit_estimator import CurvefitEstimator

from .schemas import CurvefitEstimatorData
import functools



# TODO... some mechanism to freeze this object after fit?
class EnergyChangepointModel(object): 
    """A container object for a changepoint model. After a model is fit you can access scores and 
    load calculations via propeties. 
    """

    def __init__(self, model: AbstractEnergyParameterModel, scorer: Optional[Scorer]): 
        self._model = model  
        self._scorer = scorer
        self._estimator = None  # set with call to fit

    
    def fit(self, data: CurvefitEstimatorData, sort: bool=True, **estimator_kwargs) -> 'EnergyChangepointModel':
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

        return self


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


    # XXX todo... option-c stuff works in here.
    def adjust(self, other: 'EnergyChangepointModel', **kwargs):  # returns some adjusted type
        """ creates an adjusted prediction along with ashrae formulas... 
        This would treat this model as the `post`
        """
        # we need an api that excepts two fit models pre and post and defines entrypoints 

        pass 

    
    @classmethod 
    def filter_these(cls, models: List['EnergyChangepointModel'], on: List[ComparableFilter]) -> List['EnergyChangepointModel']:
        """ A class method that calls filter_on for a set of models and filters those that do 
        not meet the criteria.
        """
        return [m for m in models if m.filter_on(on)]


    ## XXX can maybe consider adding class methods that use sklearn - GridSearchCV to choose model
    # XXX can consider adding an instance method that uses sklearn cross_val_score  


    def filter_on(self, criteria: ComparableFilters) -> bool: 
        """ A custom hook that returns whether this instance meets the filter criteria. This is for custom 
        filtering and is not statistically driven with cross validation.

        Args:
            criteria (FilterCriteria): [description]

        Returns:
            bool: [description]
        """
        return all(criteria.check(self))


    @functools.cached_property
    def score(self) -> Optional[List[Score]]:
        """Generate a number of scores from a Scorer

        Args:
            scorer (Scorer): [description]

        Returns:
            List[Score]: [description]
        """
        if self._scorer: # if we don't give a scorer just short circuit this call
            return self._scorer.score(self.y, self.pred_y)  # this runs all the scores
            

    # Probably better to make these their own class that excepts a fit estimator? 
    @functools.cached_property
    def load(self) -> Load: 
        """ Generate a load response from the fit model

        Raises:
            TypeError: [description]

        Returns:
            Load : The response tuple for the load on the fit changepoint model
        """
        return self._model.load(self.X.squeeze(), self.pred_y, self.total_y, *self.coeffs) # TODO 

    @property 
    def name(self): 
        return self._model.name


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

    


