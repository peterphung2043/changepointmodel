import pytest
import numpy as np
from ashrae.savings import AshraeAdjustedSavingsCalculator, \
    AshraeNormalizedSavingsCalculator, AdjustedSavingsResult, NormalizedSavingsResult, _cvrmse_score

from ashrae.lib.savings import adjusted, weather_normalized

def test_adjusted_savings_calls_adjusted_correctly(mocker): 

    class DummyEstimator(object):

        def len_y(self):
            return len(self.y)

        def total_y(self): 
            return sum(self.y)

    pre = DummyEstimator()

    pre.X = np.array([1.,])
    pre.y = np.array([2., ])
    pre.pred_y = np.array([3.,])
    pre.coeffs = (1,1,) 

    post = DummyEstimator()

    post.X = np.array([10.,])
    post.y = np.array([20.,])
    post.coeffs = (1,1,1,)

    mocker.patch.object(_cvrmse_score, 'calc', return_value=0.42)
    mocker.patch('ashrae.savings._get_adjusted', return_value=np.array([42.,]))
    mock = mocker.patch('ashrae.lib.savings.adjusted', return_value=(1,1,1,1,))

    calc = AshraeAdjustedSavingsCalculator()
    calc.save(pre, post)

    mock.assert_called_once_with(42, 20, 0.42, 2, 1, 1, 0.8)


def test_normalized_savings_calls_weather_normalzied_correctly(mocker): 

    class DummyEstimator(object):

        def len_y(self):
            return len(self.y)

        def total_y(self): 
            return sum(self.y)

        def predict(self, X): 
            return X

    pre = DummyEstimator()

    pre.X = np.array([1.,])
    pre.y = np.array([2., ])
    pre.pred_y = np.array([3.,])
    pre.coeffs = (1,1,) 

    post = DummyEstimator()

    post.X = np.array([10.,])
    post.y = np.array([20.,])
    post.pred_y = np.array([30.,])
    post.coeffs = (1,1,1,)

    mocker.patch.object(_cvrmse_score, 'calc', side_effect=[0.42, 0.43])
    mocker.patch('ashrae.savings._get_adjusted', return_value=np.array([42.,]))
    mock = mocker.patch('ashrae.lib.savings.weather_normalized', return_value=(1,1,1,1,))

    Xnorm_pre = np.array([100.,])
    Xnorm_post = np.array([200.,])

    calc = AshraeNormalizedSavingsCalculator(Xnorm_pre, Xnorm_post)
    calc.save(pre, post)


    mock.assert_called_once_with(100, 200, 0.42, 0.43, 1, 1, 2, 3, 1, .8)


def test_that_normalized_norm_pre_and_post_must_match_len(): 

    with pytest.raises(ValueError): 
        AshraeNormalizedSavingsCalculator(np.array([1.,2.,]), np.array([1.,]))
