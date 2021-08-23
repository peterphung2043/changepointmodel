# XXX clean this up and integrate

import numpy as np


def calc_heat_load(x: np.array,y_predicted: np.array, slope: float, y_intercept: float, break_point = np.inf) -> float:
    if slope is None or slope > 0:
        return None

    predicted_loads = (x < break_point) * (y_predicted - y_intercept)
    return postive_sum(predicted_loads)

def postive_sum(arr: np.array) -> float:
    return np.sum(arr * (arr > 0))

def calc_cool_load(x: np.array, y_predicted: np.array, slope: float, y_intercept: float, break_point =  -np.inf) -> float:
    if  slope is None or slope < 0:
        return

    predicted_loads = (x > break_point) * (y_predicted - y_intercept)
    return postive_sum(predicted_loads)


# 1. zero out consumption less than zero 
# 2. calculate total consumption and total predicted consumption (scaled consumption for both)
# 3. check ycp for ycp less than zero (for 2p, all casues )
#     * 2P: ycp is min usage
#     * 3PC: median value of y with x less than change point
#     * 3PH: median value of y with x greater than change point
#     * 4P: median value of the first three y 
#     * 5P: median value of y with x between two change points
# 4.  predicted_load = y_predicted_scaled * (ycp - scalar)
#     true_load = (total_predicted_load / total_predicted_consumption) * total_consumption
# 4a. calculate heat load
#     * 2P: if no left slope, return none; use 4. to calculate predicted load for all value.
#             Then,  calculate true load
#     * 3PH, 4P: 5P: if left slope is greater than zero, return none; use 4. to calculate predicted load for x less than change point
#            Then, calculate true load using total of predicted load greater than 0


#     def heat_load(x,y_predicted, slope, y_intercept, break_point = np.inf):
#         if slope is None or slope > 0:
#             return None
       
#         predicted_loads = (x < break_point) * (y_predicted - y_intercept)
#         return postive_sum(y_predicted)
    
#     def postive_sum(arr):
#         return np.sum(arr * (arr > 0))

# 4b. calculate cool load
#     * 2P: if no right slope, return none; use 4. se 4. to calculate predicted load for all value.
#             Then,  calculate true load
#     * 3PC, 4P, 5P: if right slope is less than zero, return none; use 4. to calculate predicted load for x greater than change point
#            Then, calculate true load using total of predicted load greater than 0. Note for 5p, it is second changepoint

#     def cool_load(x, y_predicted, slope, y_intercept, break_point =  -np.inf):
#         if  slope is None or slope < 0:
#             return
        
#         predicted_loads = (x > break_point) * (y_predicted - y_intercept)
#         return postive_sum(arr)

# 5. baseload = total_consumption - heating load - cooling load