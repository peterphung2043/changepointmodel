import numpy as np
from ashrae import energy_parameter_models as models 



def test_abstractparametermodel_yint(): 


    coeffs = models.EnergyParameterModelCoefficients(42, [1], None)

    class Dummy(models.AbstractEnergyParameterModel): 

        def parse_coeffs(self, *coeffs): ...

        def cooling_slope(self, coeffs):... 

        def heating_slope(self, coeffs):... 

        def cooling_changepoint(self, coeffs):...

        def heating_changepoint(self, coeffs): ...

    assert Dummy().yint(coeffs) == 42


def test_twop_energy_parmaeter_model_function_coeffs_parsing(): 

    model = models.TwoParameterEnergyChangepointModel()
    coeffs = model.parse_coeffs((42., 43.))
    assert coeffs == models.EnergyParameterModelCoefficients(42, [43], None)

def test_threepc_energy_parmaeter_model_function_coeffs_parsing(): 

    model = models.ThreeParameterCoolingEnergyChangepointModel()
    coeffs = model.parse_coeffs((42., 43., 44.))
    assert coeffs == models.EnergyParameterModelCoefficients(42, [43], [44])


def test_threeph_energy_parmaeter_model_function_coeffs_parsing(): 

    model = models.ThreeParameterHeatingEnergyChangepointModel()
    coeffs = model.parse_coeffs((42., 43., 44.))
    assert coeffs == models.EnergyParameterModelCoefficients(42, [43], [44])


def test_fourp_energy_parmaeter_model_function_coeffs_parsing(): 

    model = models.FourParameterEnergyChangepointModel()
    coeffs = model.parse_coeffs((42., 43., 44., 45.))
    assert coeffs == models.EnergyParameterModelCoefficients(42, [43, 44], [45])

def test_fivep_energy_parmaeter_model_function_coeffs_parsing(): 

    model = models.FiveParameterEnergyChangepointModel()
    coeffs = model.parse_coeffs((42., 43., 44., 45., 46.))
    assert coeffs == models.EnergyParameterModelCoefficients(42, [43, 44], [45, 46])



def test_twop_energy_parameter_model_slopes(): 
    model = models.TwoParameterEnergyChangepointModel()
    
    coeffs = models.EnergyParameterModelCoefficients(42, [1], None)
    assert model.cooling_slope(coeffs) == 1 
    assert model.heating_slope(coeffs) == 0

    coeffs = models.EnergyParameterModelCoefficients(42, [-1], None)
    assert model.cooling_slope(coeffs) == 0 
    assert model.heating_slope(coeffs) == -1



def test_threepc_energy_parameter_model_slopes(): 
    model = models.ThreeParameterCoolingEnergyChangepointModel()
    
    coeffs = models.EnergyParameterModelCoefficients(42, [1], [42])
    assert model.cooling_slope(coeffs) == 1 
    assert model.heating_slope(coeffs) == 0


def test_threeph_energy_parameter_model_slopes(): 
    model = models.ThreeParameterHeatingEnergyChangepointModel()
    
    coeffs = models.EnergyParameterModelCoefficients(42, [-1], [42])
    assert model.cooling_slope(coeffs) == 0 
    assert model.heating_slope(coeffs) == -1


def test_fourp_energy_parameter_model_slopes(): 
    model = models.FourParameterEnergyChangepointModel()
    
    coeffs = models.EnergyParameterModelCoefficients(42, [1,2], [42])
    assert model.cooling_slope(coeffs) == 2
    assert model.heating_slope(coeffs) == 1


def test_fivep_energy_parameter_model_slopes(): 
    model = models.FiveParameterEnergyChangepointModel()
    
    coeffs = models.EnergyParameterModelCoefficients(42, [1,2], [42,42])
    assert model.cooling_slope(coeffs) == 2 
    assert model.heating_slope(coeffs) == 1


def test_twop_energy_parameter_model_changepoints():
    model = models.TwoParameterEnergyChangepointModel()

    coeffs = models.EnergyParameterModelCoefficients(42, [1], None)
    assert model.cooling_changepoint(coeffs) is None 
    assert model.heating_changepoint(coeffs) is None  


def test_threepc_energy_parameter_model_changepoints():
    model = models.ThreeParameterCoolingEnergyChangepointModel()

    coeffs = models.EnergyParameterModelCoefficients(42, [1], [1])
    assert model.cooling_changepoint(coeffs) == 1
    assert model.heating_changepoint(coeffs) == 1  


def test_threeph_energy_parameter_model_changepoints():
    model = models.ThreeParameterHeatingEnergyChangepointModel()

    coeffs = models.EnergyParameterModelCoefficients(42, [1], [1])
    assert model.cooling_changepoint(coeffs) == 1 
    assert model.heating_changepoint(coeffs) == 1  


def test_fourp_energy_parameter_model_changepoints():
    model = models.FourParameterEnergyChangepointModel()

    coeffs = models.EnergyParameterModelCoefficients(42, [1, 2], [1])
    assert model.cooling_changepoint(coeffs) == 1 
    assert model.heating_changepoint(coeffs) == 1  


def test_fivep_energy_parameter_model_changepoints():
    model = models.FiveParameterEnergyChangepointModel()

    coeffs = models.EnergyParameterModelCoefficients(42, [1, 2], [1, 2])
    assert model.cooling_changepoint(coeffs) == 2
    assert model.heating_changepoint(coeffs) == 1   
