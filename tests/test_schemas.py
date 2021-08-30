from ashrae import schemas 

import numpy as np

def test_curvefitestimatordatamodel_handles_1d_xdata(): 

    xdata = np.array([1., 2., 3., 4., 5.])

    d = schemas.CurvefitEstimatorDataModel(X=xdata)

