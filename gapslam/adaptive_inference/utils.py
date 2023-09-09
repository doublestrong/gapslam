import warnings

import gtsam
import numpy as np

from factors.Factors import BinaryFactor
from slam.Variables import Variable


def systematic_resample(samples, weights, return_idx=False):
    """
    Systematic resampling in https://ieeexplore.ieee.org/document/4378824
    """
    N = len(weights)
    positions = (np.random.random() + np.arange(N)) / N

    if abs(np.sum(weights) - 1.) > .001:
        # Guarantee that the weights will sum to 1.
        # warnings.warn("Weights do not sum to 1 and have been renormalized.")
        weights = np.array(weights) / np.sum(weights)

    indices = np.zeros(N, dtype=int)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    if not return_idx:
        return samples[indices]
    else:
        return indices


def to_Key(var: Variable):
    """
    Translate our own variables to GTSAM symbol
    param:
        var: only its name will be used for GTSAM. We assume our name follows the convention of char + int (e.g., "X1" or "L1")
    return:
        note that we return the key value (int) which is the required input for gtsam factors
    """
    name = var.name
    return gtsam.Symbol(name[0], int(name[1:])).key()
