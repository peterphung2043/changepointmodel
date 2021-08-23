


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