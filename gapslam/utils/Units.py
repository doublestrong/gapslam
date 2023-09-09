import numpy as np
from typing import Union


_RAD_TO_DEG_FACTOR = 180.0 / np.pi
_DEG_TO_RAD_FACTOR = np.pi / 180.0
_TWO_PI = 2 * np.pi


def rad_to_deg(rad: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert angle unit from radians to degrees
    """
    return rad * _RAD_TO_DEG_FACTOR


def deg_to_rad(deg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert angle unit from degrees to radians
    """
    return deg * _DEG_TO_RAD_FACTOR



