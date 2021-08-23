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


def two_p_bound(x: np.ndarray) -> np.ndarray: #why is this a function? Why does it accept x?
    """ Bound function for a 2P model.
    Args:
        x (numpy.array): Any array.
    Returns:
        np.ndarray: Bounds for 2p scipy model to pass shape test.
    """
    return ((0, -np.inf),(np.inf, np.inf))


def three_pc_bound(x: np.ndarray) -> np.ndarray:
    """ Bound function for a 3PC model.
    Args:
        x (numpy.array): Independent variable, in ascending order.
    Returns:
        np.ndarray: Bounds for scipy model to pass data pop and shape test.
    """
    min_xcp1 = x[int(len(x)/4)]
    max_xcp1 = x[int(3 * len(x)/4)]
    return ((0,0,min_xcp1), (np.inf,np.inf, max_xcp1))


def three_ph_bound(x: np.ndarray) -> np.ndarray:
    """ Bound function for a 3PH model.
    Args:
        x (numpy.array): Independent variable, in ascending order.
    Returns:
        np.ndarray: Bounds for scipy model to pass data pop and shape test.
    """
    min_xcp1 = x[int(len(x)/4)]
    max_xcp1 = x[int(3 * len(x)/4)]
    return ((0,-np.inf,min_xcp1), (np.inf,0, max_xcp1))


def four_p_bound(x: np.ndarray) -> np.ndarray:
    """ Bound function for a 4P model.
    Args:
        x (numpy.array): Independent variable, in ascending order.
    Returns:
        np.ndarray: Bounds for scipy model to pass data pop and shape test.
    """
    min_xcp1 = x[int(len(x)/4)]
    max_xcp1 = x[int(3 * len(x)/4)]
    return ((0,-np.inf, -np.inf, min_xcp1), (np.inf,np.inf, np.inf, max_xcp1))


def five_p_bound(x: np.ndarray) -> np.ndarray:
    """ Bound function for a 5P model.
    Args:
        x (numpy.array): Independent variable, in ascending order.
    Returns:
        np.ndarray: Bounds for scipy model to pass data pop and shape test.
    """
    min_xcp1 = x[int((2/8) * len(x))]
    max_xcp1 = x[int((3/8) * len(x))]
    min_xcp2 = x[int((5/8) * len(x))]
    max_xcp2 = x[int((6/8) * len(x))]
    return ((0,-np.inf, 0, min_xcp1, min_xcp2), (np.inf,0, np.inf, max_xcp1, max_xcp2))