from changepointmodel.core import estimator, schemas
from changepointmodel.core import pmodels as models
import numpy as np

from numpy.testing import assert_almost_equal


def test_2p(generated_2p_data):
    Xdata = np.array(generated_2p_data["x"])
    ydata = np.array(generated_2p_data["y"])
    model = models.twop_factory(name="2P")
    est = estimator.EnergyChangepointEstimator(model=model)
    data = schemas.CurvefitEstimatorDataModel(X=Xdata, y=ydata)
    X, y, original_ordering = data.sorted_X_y()

    est.original_ordering = original_ordering
    est.fit(X, y)

    # coeffs
    exp_yint = 250.56459
    exp_slopes = 49.70582
    assert_almost_equal(exp_yint, est.coeffs[0], decimal=4)
    assert_almost_equal(exp_slopes, est.coeffs[1], decimal=4)

    # loads
    loads_ = est.load()
    exp_baseload = 3006.775167021835
    exp_heating = 0
    exp_cooling = 41951.71341451602
    assert_almost_equal(exp_baseload, loads_.base, decimal=4)
    assert_almost_equal(exp_heating, loads_.heating, decimal=4)
    assert_almost_equal(exp_cooling, loads_.cooling, decimal=4)

    est.r2()
    est.cvrmse()
    left_tstat, right_tstat = est.tstat()
    assert left_tstat is None
    assert right_tstat  # slope > 0

    # dpop +
    heatnum, coolnum = est.dpop()
    assert coolnum == len(Xdata)
    assert heatnum == 0  # slope > 0 so cooling
    assert est.shape()


def test_3pc(generated_3pc_data):
    Xdata = np.array(generated_3pc_data["x"])
    ydata = np.array(generated_3pc_data["y"])
    model = models.threepc_factory(name="3PC")
    est = estimator.EnergyChangepointEstimator(model=model)
    data = schemas.CurvefitEstimatorDataModel(X=Xdata, y=ydata)
    X, y, original_ordering = data.sorted_X_y()

    est.original_ordering = original_ordering
    est.fit(X, y)

    # coeffs
    exp_yint = 286.31906
    exp_slopes = 50.9474
    exp_cp = 57.0
    assert_almost_equal(exp_yint, est.coeffs[0], decimal=4)
    assert_almost_equal(exp_slopes, est.coeffs[1], decimal=4)
    assert_almost_equal(exp_cp, est.coeffs[2], decimal=4)

    # loads
    loads_ = est.load()
    exp_baseload = 3435.82875
    exp_heating = 0
    exp_cooling = 11972.65983
    assert_almost_equal(exp_baseload, loads_.base, decimal=4)
    assert_almost_equal(exp_heating, loads_.heating, decimal=4)
    assert_almost_equal(exp_cooling, loads_.cooling, decimal=4)

    est.r2()
    est.cvrmse()

    # tstat
    left_tstat, right_tstat = est.tstat()
    assert left_tstat is None
    assert right_tstat

    # dpop
    heatnum, coolnum = est.dpop()
    assert coolnum != 0
    assert heatnum == 0

    assert est.shape()


def test_3ph(generated_3ph_data):
    Xdata = np.array(generated_3ph_data["x"])
    ydata = np.array(generated_3ph_data["y"])
    model = models.threeph_factory(name="3PH")
    est = estimator.EnergyChangepointEstimator(model=model)
    data = schemas.CurvefitEstimatorDataModel(X=Xdata, y=ydata)
    X, y, original_ordering = data.sorted_X_y()

    est.original_ordering = original_ordering
    est.fit(X, y)

    # coeffs
    exp_yint = 226.35034
    exp_slopes = -37.36379
    exp_cp = 57
    assert_almost_equal(exp_yint, est.coeffs[0], decimal=4)
    assert_almost_equal(exp_slopes, est.coeffs[1], decimal=4)
    assert_almost_equal(exp_cp, est.coeffs[2], decimal=4)

    # loads
    loads_ = est.load()
    exp_baseload = 2716.20408
    exp_heating = 2802.28450
    exp_cooling = 0
    assert_almost_equal(exp_baseload, loads_.base, decimal=4)
    assert_almost_equal(exp_heating, loads_.heating, decimal=4)
    assert_almost_equal(exp_cooling, loads_.cooling, decimal=4)

    est.r2()
    est.cvrmse()

    # tstat
    left_tstat, right_tstat = est.tstat()
    assert left_tstat
    assert right_tstat is None

    # dpop
    heatnum, coolnum = est.dpop()
    exp_heatnum = len([i for i in Xdata if i <= exp_cp])
    assert coolnum == 0
    assert heatnum == exp_heatnum

    assert est.shape()


def test_4p(generated_4p_data):
    Xdata = np.array(generated_4p_data["x"])
    ydata = np.array(generated_4p_data["y"])
    model = models.fourp_factory(name="4P")
    est = estimator.EnergyChangepointEstimator(model=model)
    data = schemas.CurvefitEstimatorDataModel(X=Xdata, y=ydata)
    X, y, original_ordering = data.sorted_X_y()

    est.original_ordering = original_ordering
    est.fit(X, y)

    # coeffs
    exp_yint = 326.26842
    exp_left_slopes = -34.14062
    exp_right_slopes = 49.756256
    exp_cp = 57
    assert_almost_equal(exp_yint, est.coeffs[0], decimal=4)
    assert_almost_equal(exp_left_slopes, est.coeffs[1], decimal=4)
    assert_almost_equal(exp_right_slopes, est.coeffs[2], decimal=4)
    assert_almost_equal(exp_cp, est.coeffs[3], decimal=4)

    # loads
    loads_ = est.load()
    exp_baseload = 3915.221061
    exp_heating = 2560.54721
    exp_cooling = 11692.72032
    assert_almost_equal(exp_baseload, loads_.base, decimal=4)
    assert_almost_equal(exp_heating, loads_.heating, decimal=4)
    assert_almost_equal(exp_cooling, loads_.cooling, decimal=4)

    est.r2()
    est.cvrmse()

    # tstat
    left_tstat, right_tstat = est.tstat()
    assert left_tstat
    assert right_tstat

    # dpop
    heatnum, coolnum = est.dpop()
    exp_heat = len([i for i in Xdata if i <= exp_cp])
    exp_cool = len([i for i in Xdata if i > exp_cp])
    assert heatnum == exp_heat
    assert coolnum == exp_cool

    assert est.shape() == False  # left slope < right slope


def test_5p(generated_5p_data):
    Xdata = np.array(generated_5p_data["x"])
    ydata = np.array(generated_5p_data["y"])
    model = models.fivep_factory(name="5P")
    est = estimator.EnergyChangepointEstimator(model=model)
    data = schemas.CurvefitEstimatorDataModel(X=Xdata, y=ydata)
    X, y, original_ordering = data.sorted_X_y()

    est.original_ordering = original_ordering
    est.fit(X, y)

    # coeffs
    exp_yint = 423.02913
    exp_left_slopes = -31.01931
    exp_right_slopes = 82.45359
    exp_left_cp = 57
    exp_right_cp = 86
    assert_almost_equal(exp_yint, est.coeffs[0], decimal=4)
    assert_almost_equal(exp_left_slopes, est.coeffs[1], decimal=4)
    assert_almost_equal(exp_right_slopes, est.coeffs[2], decimal=4)
    assert_almost_equal(exp_left_cp, est.coeffs[3], decimal=4)
    assert_almost_equal(exp_right_cp, est.coeffs[4], decimal=4)

    # loads
    loads_ = est.load()
    exp_baseload = 5076.34960
    exp_heating = 2326.44871
    exp_cooling = 3215.69026
    assert_almost_equal(exp_baseload, loads_.base, decimal=4)
    assert_almost_equal(exp_heating, loads_.heating, decimal=4)
    assert_almost_equal(exp_cooling, loads_.cooling, decimal=4)

    est.r2()
    est.cvrmse()

    # tstat
    left_tstat, right_tstat = est.tstat()
    assert left_tstat
    assert right_tstat

    # dpop
    heatnum, coolnum = est.dpop()
    exp_heat = len([i for i in Xdata if i <= exp_left_cp])
    exp_cool = len([i for i in Xdata if i > exp_right_cp])
    assert heatnum == exp_heat
    assert coolnum == exp_cool

    assert est.shape() == False  #  left slope < right slope
