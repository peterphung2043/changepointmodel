""" Exception wrapper for bplrpc. fastapi will register an exception handler to deal specifically with BplrpcException classes to reraise as a 409 http error.
"""

from typing import Dict
import sys
import traceback 


class BplrpcException(Exception): 

    def __init__(self, 
        info: Dict[str, str]=None, 
        message: str=""): 
        
        self.info = info
        self.message = message
        super().__init__(message)

    def __str__(self): 
        return f"{self.message}"

    def __repr__(self): 
        return str(vars(self))



def bplrpc_exception_wrapper(err: Exception, message: str, **info_kwargs) -> BplrpcException: 

    exc_type, exc_value, exc_traceback = sys.exc_info()
    info = {
        **info_kwargs, 
        'exc': err.__class__.__name__,  # giving name here but not stack trace... 
        'tb': repr(traceback.format_exception(exc_type, exc_value, exc_traceback))
    }

    e = BplrpcException(info=info, message=message)
    return e


