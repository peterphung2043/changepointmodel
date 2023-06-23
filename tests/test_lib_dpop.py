from changepointmodel.core.calc import dpop
import numpy as np


def test_dpop_good(dummy_result_for_filter_extra_good_for_dpop_test):
    data = dummy_result_for_filter_extra_good_for_dpop_test

    twop_data = data[0]
    threepc_data = data[1]
    threeph_data = data[2]
    fourp_data = data[3]
    fivep_data = data[4]

    # 2P testing goes here
    ...

    result = dpop.threepc(
        threepc_data.result.input_data.X, threepc_data.result.coeffs.changepoints[0]
    )
    assert result == 8

    result = dpop.threeph(
        threeph_data.result.input_data.X, threeph_data.result.coeffs.changepoints[0]
    )
    assert result == 5

    result = dpop.fourp(
        fourp_data.result.input_data.X, fourp_data.result.coeffs.changepoints[0]
    )
    assert result == (7, 5)

    result = dpop.fivep(
        fivep_data.result.input_data.X, fivep_data.result.coeffs.changepoints
    )
    assert result == (4, 5)


def test_dpop_bad(dummy_result_for_filter_extra_bad_for_dpop_test):
    data = dummy_result_for_filter_extra_bad_for_dpop_test

    twop_data = data[0]
    threepc_data = data[1]
    threeph_data = data[2]
    fourp_data = data[3]
    fivep_data = data[4]

    # 2P testing goes here
    ...

    result = dpop.threepc(
        threepc_data.result.input_data.X, threepc_data.result.coeffs.changepoints[0]
    )
    assert result == 11

    result = dpop.threeph(
        threeph_data.result.input_data.X, threeph_data.result.coeffs.changepoints[0]
    )
    assert result == 2

    result = dpop.fourp(
        fourp_data.result.input_data.X, fourp_data.result.coeffs.changepoints[0]
    )
    assert result == (10, 2)

    result = dpop.fivep(
        fivep_data.result.input_data.X, fivep_data.result.coeffs.changepoints
    )
    assert result == (7, 5)
