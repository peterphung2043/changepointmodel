from changepointmodel.core.calc import tstat
import numpy as np


def test_tstat_good(dummy_t_test_bad):
    data = dummy_t_test_bad

    twop_data = data[0]
    threepc_data = data[1]
    threeph_data = data[2]
    fourp_data = data[3]
    fivep_data = data[4]

    # 2P
    result = tstat.twop(
        X=twop_data.result.input_data.X,
        Y=twop_data.result.input_data.y,
        pred_y=twop_data.result.pred_y,
        slope=twop_data.result.coeffs.slopes[0],
    )
    assert result == np.inf

    # 3PC
    result = tstat.threepc(
        X=threepc_data.result.input_data.X,
        Y=threepc_data.result.input_data.y,
        pred_y=threepc_data.result.pred_y,
        slope=threepc_data.result.coeffs.slopes[0],
        changepoint=threepc_data.result.coeffs.changepoints[0],
    )
    assert result == 1.1224972160321822

    # 3PH
    result = tstat.threeph(
        X=threeph_data.result.input_data.X,
        Y=threeph_data.result.input_data.y,
        pred_y=threeph_data.result.pred_y,
        slope=threeph_data.result.coeffs.slopes[0],
        changepoint=threeph_data.result.coeffs.changepoints[0],
    )
    assert result == -0.4898979485566357

    # 4P
    result = tstat.fourp(
        X=fourp_data.result.input_data.X,
        Y=fourp_data.result.input_data.y,
        pred_y=fourp_data.result.pred_y,
        slopes=fourp_data.result.coeffs.slopes,
        changepoint=fourp_data.result.coeffs.changepoints[0],
    )
    assert result == (-0.7348469228349536, 1.1224972160321822)

    # 5P
    result = tstat.fivep(
        X=fivep_data.result.input_data.X,
        Y=fivep_data.result.input_data.y,
        pred_y=fivep_data.result.pred_y,
        slopes=fivep_data.result.coeffs.slopes,
        changepoints=fivep_data.result.coeffs.changepoints,
    )
    assert result == (-0.24494897427831785, 0.31622776601683794)


def test_tstat_bad(dummy_t_test_good):
    data = dummy_t_test_good

    twop_data = data[0]
    threepc_data = data[1]
    threeph_data = data[2]
    fourp_data = data[3]
    fivep_data = data[4]

    # 2P
    result = tstat.twop(
        X=twop_data.result.input_data.X,
        Y=twop_data.result.input_data.y,
        pred_y=twop_data.result.pred_y,
        slope=twop_data.result.coeffs.slopes[0],
    )
    assert result == np.inf

    # 3PC
    result = tstat.threepc(
        X=threepc_data.result.input_data.X,
        Y=threepc_data.result.input_data.y,
        pred_y=threepc_data.result.pred_y,
        slope=threepc_data.result.coeffs.slopes[0],
        changepoint=threepc_data.result.coeffs.changepoints[0],
    )
    assert result == np.inf

    # 3PH
    result = tstat.threeph(
        X=threeph_data.result.input_data.X,
        Y=threeph_data.result.input_data.y,
        pred_y=threeph_data.result.pred_y,
        slope=threeph_data.result.coeffs.slopes[0],
        changepoint=threeph_data.result.coeffs.changepoints[0],
    )
    assert result == -np.inf

    # 4P
    result = tstat.fourp(
        X=fourp_data.result.input_data.X,
        Y=fourp_data.result.input_data.y,
        pred_y=fourp_data.result.pred_y,
        slopes=fourp_data.result.coeffs.slopes,
        changepoint=fourp_data.result.coeffs.changepoints[0],
    )
    assert result == (-np.inf, np.inf)

    # 5P
    result = tstat.fivep(
        X=fivep_data.result.input_data.X,
        Y=fivep_data.result.input_data.y,
        pred_y=fivep_data.result.pred_y,
        slopes=fivep_data.result.coeffs.slopes,
        changepoints=fivep_data.result.coeffs.changepoints,
    )
    assert result == (-np.inf, np.inf)
