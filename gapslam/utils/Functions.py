import json
import os

import numpy as np
import pandas

from utils.Units import _TWO_PI
from typing import Dict, List
from slam.Variables import Variable
from scipy.stats import circmean
from scipy.stats import gaussian_kde

def sort_pair_lists(number_list, attached_list) -> "sorted_number_list, sorted_attached_list":
    sorted_number_list, sorted_attached_list = (list(t) for t in zip(*sorted(zip(number_list, attached_list))) )
    return sorted_number_list, sorted_attached_list


def none_to_zero(x) -> "x":
    return 0.0 if x is None else x


def theta_to_pipi(theta):
    return (theta + np.pi) % _TWO_PI - np.pi


def sample_dict_to_array(samples: Dict[Variable, np.ndarray],
                         ordering: List[Variable] = None):
    """
    Convert samples from a dictionary form to numpy array form
    """
    if ordering is None:
        ordering = list(samples.keys())
    elif set(ordering) != set(samples.keys()):
        raise ValueError("Variables in the ordering do not match those in "
                         "the dictionary")
    return np.hstack((samples[var] for var in ordering))

def sample_from_arr(arr: np.array, size: int = 1) -> np.array:
    return arr[np.random.choice(len(arr), size=size, replace=False)]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def array_order_to_dict(samples: np.ndarray, order: List[Variable])->Dict:
    res = {}
    cur_idx = 0
    for var in order:
        res[var] = samples[:,cur_idx:cur_idx+var.dim]
        cur_idx += var.dim
    return res

def kabsch_umeyama(A, B):
    # authored by:
    # https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB
    return R, c, t

def reject_outliers(data, iq_range=0.5):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    sr = pandas.Series(data)
    pcnt = (1 - iq_range) / 2
    qlow, median, qhigh = sr.dropna().quantile([pcnt, 0.50, 1-pcnt])
    iqr = qhigh - qlow
    return np.where(np.logical_and(data>=qlow-1.7*iqr, data<=qhigh+1.7*iqr))[0] # get 99.7% data
    # return sr[ (sr - median).abs() <= iqr]

def normalize_samples(samples: np.ndarray, circular_dim_list: List[bool] = None, regularization = False):
    """
    Normalize samples. Need circmean for samples in [-pi, pi]
    params:
        samples: each row is a sample while each column is a dimension
        circular_dim_list: a list of boolean values
    return:
        norm_samples: np.ndarray
        means: np.ndarray
        stds: np.ndarray
    """
    sample_dim = samples.shape[-1]
    norm_samples = np.zeros_like(samples)
    if circular_dim_list is None:
        circular_dim_list = np.zeros(sample_dim, dtype=bool)
    means = np.zeros(sample_dim)
    stds = np.zeros(sample_dim)
    circular_indices = np.where(circular_dim_list)[0]
    euclidean_indices = np.setdiff1d(np.arange(sample_dim), circular_indices)

    # normalizing samples in circular dim
    if len(circular_indices) > 0:
        means[circular_indices] = circmean(samples[:, circular_indices], high=np.pi, low=-np.pi, axis=0)
        # transform the data to [-pi, pi]
        norm_samples[:, circular_indices] = theta_to_pipi(samples[:, circular_indices] - means[circular_indices])
        stds[circular_indices] = np.std(norm_samples[:, circular_indices], axis=0)  # approximiate std of circular quantity

    #Euclidean dim
    if len(euclidean_indices) > 0:
        means[euclidean_indices] = np.mean(samples[:, euclidean_indices], axis=0)
        stds[euclidean_indices] = np.std(samples[:, euclidean_indices], axis=0)
        norm_samples[:, euclidean_indices] = samples[:, euclidean_indices] - means[euclidean_indices]

    # this may cause error when samples of a variable get narrowed down to a point
    if regularization:
        stds = np.clip(stds, a_min = 1e-5, a_max=None)
    norm_samples = norm_samples /  stds
    return norm_samples, means, stds

def sample2ked(samples: np.ndarray):
    """
    params:
        samples with the sample (# of samples, # of dim)
    return:
        list of kernel functions
    """
    kernels = [gaussian_kde(samples[:, i]) for i in range(samples.shape[1])]
    return kernels

def kill_files(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print("failed on filepath: %s" % file_path)


def kill_numba_cache():

    root_folder = os.path.realpath(__file__ + "/../../")

    for root, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            if dirname == "__pycache__":
                try:
                    kill_files(root + "/" + dirname)
                except Exception as e:
                    print("failed on %s", root)