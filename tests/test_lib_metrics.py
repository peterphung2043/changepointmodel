import pytest
import numpy as np
from changepointmodel.core.calc import metrics as energymodelmetrics


def test_r2_score_forwards_arguments(mocker):
    y = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([4.0, 5.0, 6.0])
    sample_weight = np.array([7.0, 8.0, 9.0])

    mock = mocker.patch("sklearn.metrics.r2_score")
    energymodelmetrics.r2_score(y, y_pred, sample_weight=sample_weight)
    mock.assert_called_once_with(y, y_pred, sample_weight=sample_weight)


def test_rmse_score_forwards_arguments(mocker):
    y = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([4.0, 5.0, 6.0])
    sample_weight = np.array([7.0, 8.0, 9.0])

    mock = mocker.patch("sklearn.metrics.mean_squared_error")
    energymodelmetrics.rmse(y, y_pred, sample_weight=sample_weight)
    mock.assert_called_once_with(
        y, y_pred, sample_weight=sample_weight, squared=False
    )  # NOTE squared=False returns rmse according to skl docs


def test_cvrmse_score_forwards_arguments(mocker):
    y = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([4.0, 5.0, 6.0])
    sample_weight = np.array([7.0, 8.0, 9.0])

    mock = mocker.patch("sklearn.metrics.mean_squared_error")
    energymodelmetrics.cvrmse(y, y_pred, sample_weight=sample_weight)
    mock.assert_called_once_with(y, y_pred, sample_weight=sample_weight, squared=False)


def test_adjusted_r2_score():
    y_pred = np.array(
        [
            1279.60425108,
            1213.54261136,
            1118.12024288,
            1081.41933192,
            1021.4745107,
            935.8390518,
            906.74142535,
            998.81156387,
            1074.1416772,
            1080.83768728,
            1156.16780061,
            1186.29984595,
        ]
    )

    y = np.array(
        [
            1314.0,
            1117.0,
            1123.0,
            1205.0,
            976.0,
            915.0,
            975.0,
            916.0,
            1088.0,
            1045.0,
            1112.0,
            1267.0,
        ]
    )

    adj_r2 = energymodelmetrics.adjusted_r2_score(y=y, y_pred=y_pred, p=4)

    assert adj_r2 == 0.6324037889660822


# XXX removed this private method ...
# def test_cvrmse_from_rmse():
#     rmse = 2
#     y = np.array([1.0, 2.0, 3.0])
#     assert 1 == energymodelmetrics._cvrmse_from_rmse(rmse, y)
