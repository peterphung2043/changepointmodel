from typing import Tuple
import numpy as np
from . import uncertainties


def adjusted(gross_adjusted_y: float, 
    gross_post_y: float, 
    pre_cvrmse: float, 
    pre_p: int,
    pre_n: int, 
    post_n: int, 
    confidence_interval: float=0.8) -> Tuple[float, float, float, float]: 
        
    total_savings = gross_adjusted_y - gross_post_y 
    percent_savings = total_savings - gross_post_y

    fractional_savings = np.absolute(uncertainties.fractional_avoided_energy_use(
        gross_adjusted_y, 
        gross_post_y))
    
    rel_unc = uncertainties.relative_uncertainty_avoided_energy_use(
        confidence_interval, 
        pre_cvrmse, 
        fractional_savings, 
        pre_p, 
        pre_n, 
        post_n)

    absolute_uncertainty_of_total_savings = rel_unc * np.absolute(total_savings)
    
    average_monthly_savings = total_savings/post_n
    relative_uncertainty_of_total_savings = absolute_uncertainty_of_total_savings /\
            np.absolute(total_savings)

    return total_savings, average_monthly_savings, percent_savings, relative_uncertainty_of_total_savings


def weather_normalized(gross_normalized_y_pre: float, 
    gross_normalized_y_post: float, 
    pre_cvrmse: float, 
    post_cvrmse: float, 
    pre_n: int, 
    post_n: int, 
    pre_p: int, 
    post_p: int, 
    confidence_interval: float=0.8):

    total_savings =  gross_normalized_y_pre - gross_normalized_y_post #this seems to return wrong value
    percent_savings = total_savings / gross_normalized_y_pre

    n_norm_months = 12 # formula makes this assumption... therefore there should be an assertion somewhere that the data being passed is correct.

    pre_rel_unc = uncertainties.relative_uncertainty_normalized_period(
        confidence_interval, 
        pre_n, 
        pre_cvrmse, 
        gross_normalized_y_pre, 
        pre_p, 
        n_norm_months=n_norm_months)  

    post_rel_unc = uncertainties.relative_uncertainty_normalized_period(
        confidence_interval, 
        post_n, 
        post_cvrmse,
        gross_normalized_y_post, 
        post_p, 
        n_norm_months=n_norm_months)

    absolute_savings_uncertainty = np.sqrt((pre_rel_unc)**2 + (post_rel_unc)**2)
    percent_savings_uncertainty = np.absolute(absolute_savings_uncertainty / total_savings)
    average_monthly_savings = total_savings / n_norm_months

    return total_savings, average_monthly_savings, percent_savings, percent_savings_uncertainty


