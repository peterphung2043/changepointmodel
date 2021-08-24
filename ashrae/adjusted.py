""" An API wrapper for doing adjusted calculations on models. (option-c methodology)
"""

from .base import AbstractEnergyParameterModel

class Adjusted(object): 
    
    def __init__(self, 
        pre: AbstractEnergyParameterModel, 
        post: AbstractEnergyParameterModel): 
        
        self._pre = pre 
        self._post = post

    
