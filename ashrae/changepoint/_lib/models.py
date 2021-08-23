# XXX clean this up and integrate


def two_p(x: np.ndarray, 
    ycp: float, 
    m: float) -> np.ndarray:
    """Function to fit a 2P model using numpy. 
    NOTE: changed value name to m to make this more clear. Subsequent modules will 
    place this in the appropriate right or left scope based on the whether the value is positive or 
    negative.
    Args:
        x (np.ndarray): The x vector. Should be a 1d vector of float64 or 32
        ycp (float): The y intercept
        m2 (float): The slope
    Returns:
        np.ndarray: Predicted dependent variables
    """
    return m * x + ycp


def three_pc(x: np.ndarray, 
    ycp: float, 
    m2: float, 
    xcp1: float) -> np.ndarray:
    """ Function to fit a 3PC model using numpy.
    Args:
        x (numpy.array): Independent variable.
        ycp (float): Y Intercept.
        m2 (float): Slope.
        xcp1 (float): Changepoint.
    Returns:
        np.ndarray: Predicted dependent variables
    """
    return(
    (x < xcp1) * (ycp) +
    (x >= xcp1) * (m2*(x - xcp1) + ycp))


def three_ph(x: np.ndarray, 
    ycp: float, 
    m1: float, 
    xcp1: float) -> np.ndarray:
    """ Function to fit a 3PH model using numpy.
    Args:
        x (numpy.array): Independent variable.
        ycp (float): Y Intercept.
        m2 (float): Slope.
        xcp1 (float): Changepoint.
    Returns:
        np.ndarray: Predicted dependent variables
    """
    return(
    (x < xcp1) * (m1*(x-xcp1) + ycp) +
    (x >= xcp1) * (ycp))


def four_p(x: np.ndarray, 
    ycp: float, 
    m1: float, 
    m2: float, 
    xcp1: float) -> np.ndarray:
    """ Function to fit a 4P model using numpy.
    Args:
        x (numpy.array): Independent variable.
        ycp (float): Y Intercept.
        m1 (float): Left slope.
        m2 (float): Right slope.
        xcp1 (float): Changepoint.
    Returns:
        np.ndarray: Predicted dependent variables
    """
    return(
    (x < xcp1) * (m1*(x-xcp1) + ycp) +
    (x >= xcp1) * (m2*(x - xcp1) + ycp))


def five_p(x: np.ndarray, 
    ycp: float, 
    m1: float, 
    m2: float, 
    xcp1: float, 
    xcp2: float) -> np.ndarray:
    """ Function to fit a 5P model using numpy.
     Args:
        x (numpy.array): Independent variable.
        ycp (float): Y Intercept.
        m1 (float): Left slope.
        m2 (float): Right slope.
        xcp1 (float): Left changepoint.
        xcp2 (float): Right changepoint.
    Returns:
        np.ndarray: Predicted dependent variables
    """
    return(
    (x < xcp1) * (m1*(x-xcp1) + ycp) +
    ((x < xcp2) &(x >= xcp1)) * (ycp) +
    (x >= xcp2) * (m2*(x - xcp2) + ycp))