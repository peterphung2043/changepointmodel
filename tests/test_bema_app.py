from urllib import request, response
from changepointmodel.bema import app, base, results, exc, models

import pytest
import numpy as np


def test_BemaChangepointModeler(raw_energy_model_data):
    usage =raw_energy_model_data['pre_data']['usage']
    oat = raw_energy_model_data['pre_data']['oat']

    models = ["2P", "3PC", "3PH", "4P"]

    cpm = app.BemaChangepointModeler(
        oat=oat, usage=usage, models=models
    )

    results = cpm.run()
    assert len(results) == 4
    res = results.pop(0)
    assert isinstance(res, base.BemaChangepointResultContainer)

    res_pred_y = res.result.pred_y
    res_input_data_x = res.result.input_data.X
    res_input_data_y = res.result.input_data.y

    ex_pred_y = [i*res.result.coeffs.slopes[0] + res.result.coeffs.yint for i in oat]
    assert list(res_input_data_x) == oat
    assert list(res_input_data_y) == usage
    assert list(res_pred_y) == ex_pred_y

def test_BemaChangepointModeler_with_exception(raw_energy_model_data):
    usage =raw_energy_model_data['pre_data']['usage']
    oat = raw_energy_model_data['pre_data']['oat']

    models = ["5P"]

    cpm = app.BemaChangepointModeler(
        oat=oat, usage=usage, models=models
    )

    with pytest.raises(exc.BemaChangepointException):
        cpm.run()

def test_run_baseline(baseline_request):
    request = models.BaselineChangepointModelRequest(**baseline_request)
    response = app.run_baseline(request)

    assert len(response.results) == 1

    #wihtout filter
    req = baseline_request
    req['model_config'].pop('model_filter')
    request = models.BaselineChangepointModelRequest(**req)
    response = app.run_baseline(request)

def test_run_baseline_with_norms(baseline_request):
    request = models.BaselineChangepointModelRequest(**baseline_request)
    response = app.run_baseline(request)

    assert len(response.results) == 1

    #wihtout filter
    req = baseline_request
    req['norms'] = [1]*12 #adding norms to see if it runs
    req['model_config'].pop('model_filter')
    request = models.BaselineChangepointModelRequest(**req)
    response = app.run_baseline(request)
    

    assert len(response.results) == 4

def test_run_baseline_with_exception(baseline_request):
    req = baseline_request
    req['model_config']['models'].append('5P')
    request = models.BaselineChangepointModelRequest(**req)

    with pytest.raises(exc.BemaChangepointException):
        app.run_baseline(request)

def test_run_option_c_with_filter(option_c_request):
    req = option_c_request
    req['pre']['model_config']['models'] = ["2P", "3PC", "3PH", "4P"]
    request = models.SavingsRequest(**req)
    response = app.run_optionc(request)

    assert len(response.results) == 1 #only one best model for pre and post

def test_run_option_c_with_filter_with_norms(option_c_request):
    req = option_c_request
    req['pre']['model_config']['models'] = ["2P", "3PC", "3PH", "4P"]
    req['norms'] = [1]*12
    request = models.SavingsRequest(**req)
    response = app.run_optionc(request)

    assert len(response.results) == 1 #only one best model for pre and post
    res = response.results[0]

    assert len(res.normalized_savings.result.normalized_y_post) == 12
    assert len(res.normalized_savings.result.normalized_y_pre) == 12

def test_run_option_c_with_no_filter(option_c_request, raw_energy_model_data):
    req = option_c_request
    req['pre']['model_config']['models'] = ["2P", "3PC", "3PH", "4P"]
    req['pre']['model_config'].pop('model_filter')
    req['post']['model_config'].pop('model_filter')
    request = models.SavingsRequest(**req)
    response = app.run_optionc(request)

    assert len(response.results) == 20 #4 models for pre and 5 models for post

    res = response.results.pop(0)

    res_pred_y = res.pre.pred_y
    res_input_data_x = res.pre.input_data.X
    res_input_data_y = res.pre.input_data.y

    usage =raw_energy_model_data['pre_data']['usage']
    oat = raw_energy_model_data['pre_data']['oat']

    post_oat = raw_energy_model_data['post_data']['oat']


    cp_model = lambda x: x*res.pre.coeffs.slopes[0] + res.pre.coeffs.yint
    ex_pred_y = [cp_model(i) for i in oat]
    assert list(res_input_data_x) == oat
    assert list(res_input_data_y) == usage
    assert list(res_pred_y) == ex_pred_y

    assert res.pre.name == '2P'
    assert res.post.name == '2P'

    exp_adj_y = [cp_model(i) for i in post_oat]
    res_adj_y = res.adjusted_savings.result.adjusted_y

    assert list(res_adj_y) == exp_adj_y

def test_run_option_c_with_exception(option_c_request):
    req = option_c_request
    request = models.SavingsRequest(**req)

    with pytest.raises(exc.BemaChangepointException):
        app.run_optionc(request)




def test_argsort_unargsort_pair(): 
    x_ = [31.5549430847168, 41.46236038208008, 40.40647888183594, 
                48.85377502441406, 66.41313934326172, 72.4892578125, 79.4146957397461,
                80.5838623046875, 72.34444427490234, 59.53846740722656,
                45.7861213684082, 40.545326232910156]

    y_ = [2771.7328796075267, 2778.9696134985625, 2866.628408060809,
                2887.0038965747117, 3553.987575239154, 3980.2125375555543,
                4595.253582350758, 4622.158449907303, 4136.844967333332,
                3386.2120536070365, 2841.454855450504, 2862.490733376347]
    x = np.array(x_)
    y = np.array(y_)
    
    new_x, new_y, ordering = app.argsort_1d_idx(x, y)

    sorted_x =[i.pop() for i in [[31.5549430847168],
            [40.40647888183594],
            [40.545326232910156],
            [41.46236038208008],
            [45.7861213684082],
            [48.85377502441406],
            [59.53846740722656],
            [66.41313934326172],
            [72.34444427490234],
            [72.4892578125],
            [79.4146957397461],
            [80.5838623046875 ]]]

    sorted_y = [y_[i] for i in ordering]

    assert list(new_x) == sorted_x
    assert list(new_y) == sorted_y

    unorder_x = results.unargsort_1d_idx(new_x, ordering)
    unorder_y = results.unargsort_1d_idx(new_y, ordering)
    
    assert list(unorder_x) == x_
    assert list(unorder_y) == y_
    