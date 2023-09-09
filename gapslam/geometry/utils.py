import numpy as np
import math
from typing import List, Tuple, Union
from geometry.TwoDimension import Point2, SE2Pose

def product_manifold_dist(x1: np.ndarray, x2: np.ndarray, geom_list: Union[List[Point2],List[SE2Pose]]):
    dist = 0
    prev_dim = 0
    for i, m in enumerate(geom_list):
        dist += m.dist(x1[prev_dim:prev_dim+m.dim],x2[prev_dim:prev_dim+m.dim])**2
        prev_dim = prev_dim + m.dim
    return dist**.5
