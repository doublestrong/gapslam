import numpy as np


def is_symmetric(a: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """
    Check whether matrix a is symmetric
    :param a: the matrix to be tested
    :type: numpy.ndarray
    :param rtol: relative tolerance
    :type: float
    :param atol: absolute tolerance
    :type: float
    :return: whether matrix a is symmetric
    :rtype: bool
    :raise ValueError: when the numpy array is not a matrix
    """
    if len(a.shape) == 2:
        return np.allclose(a, a.T, rtol=rtol, atol=atol)
    else:
        raise ValueError("The input numpy array must be a matrix")


def is_pos_def(x: np.ndarray) -> bool:
    """
    Check if a matrix is positive definite
    """
    return np.all(np.linalg.eigvals(x) > 0)


def is_spd(x: np.ndarray) -> bool:
    """
    Check if a matrix is symmetric positve definite
    """
    return is_symmetric(x) and is_pos_def(x)
