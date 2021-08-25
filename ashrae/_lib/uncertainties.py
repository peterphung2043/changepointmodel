""" ashrae formulas for option-c methodologies involving energy usage.
"""
import numpy as np 
from scipy import stats
from ..base import OneDimNDArray



def fractional_avoided_energy_use(total_annual_adjusted: float, total_annual_measured_reporting: float) -> float:
    
    """ASHRAE Guideline 14 2014 Annex B Eq. B-22 Fractional Avoided Energy Use
    Args:
        annual_adj_baseline (float): Sum of a year of adjusted baseline consumption.
        annual_measured_reporting (float): Sum of a year of measured reporting consumption.
    Returns:
        float: Savings as a fraction of annual adjusted baseline.
    """
    return (total_annual_adjusted - total_annual_measured_reporting) / total_annual_adjusted


def relative_uncertainty_avoided_energy_use(
    conf_interval: float, 
    cvrmse: float, 
    f: float, 
    p: int, 
    n_months_pre: int,  
    n_months_post: int) -> float: 
    """
     ASHARE Guideline 14 2014, Annex B
     Modified Eq. B-28
     Relative Uncertainty of Avoided Energy Use

    Args:
        conf_interval (float): desired confidence interval
        cvrmse (float): cvrmse of baseline model as a fraction 
        f (float): savings as a fraction of annual adjusted baseline
        p (int): number of model parameters. depends on model complexity and number of independent variables.
        len_months_pre (int): number of months of data used to create baseline model
        len_months_post (int): number of months in the reporting period.

    Returns:
        float: [description]
    """

    tstat = stats.t.interval(conf_interval, n_months_pre-p)[1]
    return 1.26 * tstat * cvrmse * np.sqrt((1/n_months_post) * (1 + 2/n_months_pre))/f


def relative_uncertainty_normalized_period(
    conf_interval: float, 
    n_months: int, 
    cvrmse: float, 
    gross_norm: float, 
    p: int, 
    n_norm_months=12) -> float: 
    
    """
     ASHARE Guideline 14 2014, Annex B
     Modified Eq. B-28
     Relative Uncertainty of Normalized Period
     This function is used twice in normalized savings uncertainty calculation.

    Args:
        conf_interval ([type]): desired confidence interval 
        n_months ([type]): number of months of data used to create the model
        cvrmse ([type]): model cvrmse as a fraction
        gross_norm ([type]): sum of normalized consumption for the number of cny months 
        p ([type]): number of parameters. Depends on model complexity and number of independent variables.
        n_norm_months ([type]): number of months in normal period (default to 12)

    Returns:
        float: uncertainty for sum of normalized consumption for that period
    """ 

    tstat = stats.t.interval(conf_interval, n_months - p)[1]
    return 1.26 * tstat * gross_norm * cvrmse * np.sqrt((1/n_norm_months) * (1 + 2/n_months))
