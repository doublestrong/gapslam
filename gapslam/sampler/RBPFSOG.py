from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed

from adaptive_inference.utils import systematic_resample
from factors.Factors import SE2R2RangeGaussianLikelihoodFactor

from slam.Variables import Variable


@dataclass
class SOG:
    weights: np.ndarray
    means: np.ndarray
    covs: np.ndarray


def copyLmk2SOG(d: dict[Variable, SOG]):
    return {k: SOG(mysog.weights.copy(), mysog.means.copy(), mysog.covs.copy()) for k, mysog in d.items()}


def range_meas_jacob(lmkxy, rbtxy):
    return (lmkxy - rbtxy) / np.linalg.norm(lmkxy - rbtxy)


class RBPFSOG(object):
    """
    A range-only SLAM solver using RBPF where conditionals of landmarks are modeled via sum of Gaussians.
    This is an implementation of the method described in the following paper.
    The implementation is for planar cases.
    J. -L. Blanco, J. -A. Fernandez-Madrigal and J. Gonzalez, "Efficient probabilistic Range-Only SLAM," 2008 IEEE/RSJ International Conference on Intelligent Robots and Systems, Nice, France, 2008, pp. 1017-1022, doi: 10.1109/IROS.2008.4650650.
    https://ieeexplore.ieee.org/abstract/document/4650650
    """

    def __init__(self, num_sample: int, dim_cap: int = 200, dm=5.0, K=0.4, max_drop_mode_weight=0.8,
                 parrallel_n_jobs=1):
        # dm is the maximum distance between adjacent Gaussians in the paper
        self.n = num_sample
        self.vars = []
        self.lmk_vars = []
        self.var2slice = {}
        self.samples = np.zeros((self.n, dim_cap))
        self.weights = np.ones(self.n) / self.n
        self.dm = dm
        self.K = K  # the proportionality factor for scaling the tangential std of a mode
        self.pathID2lmkSOG = [{} for _ in range(self.n)]
        self.max_drop_mode_weight = max_drop_mode_weight
        self.all_factors = []
        self.parallel_n_jobs = parrallel_n_jobs

    @property
    def dim(self):
        return sum([v.dim for v in self.vars])

    def update_var_sample(self, s: np.ndarray, var: Variable):
        cur_dim = self.dim
        new_slice = cur_dim + np.arange(var.dim)
        if max(new_slice) >= self.samples.shape[1]:
            # expand the sample array to two-fold columns
            self.samples = np.hstack((self.samples, np.zeros((self.n, self.samples.shape[1]))))
        self.var2slice[var] = new_slice
        self.vars.append(var)
        self.samples[:, new_slice] = s

    def add_odom_factor(self, f):
        self.all_factors.append(f)
        var1, var2 = f.vars
        var1sample = self.samples[:, self.var2slice[var1]]
        s = f.sample(var1=var1sample)
        self.update_var_sample(s, var2)

    def rbt_samples_xy(self, rbt_var):
        return self.samples[:, self.var2slice[rbt_var][:2]]

    def init_sog(self, f):
        rbt_var, lmk_var = f.vars
        r = f.observation[0]  # range meas.
        std_r = f.sigma
        # only need x and y in rbt samples
        rbt_samples_xy = self.rbt_samples_xy(rbt_var)
        # number of modes for each path
        num_modes = int(2 * np.ceil(np.pi * r / self.dm))
        one_path_angles = 2 * np.pi * np.arange(num_modes) / num_modes
        one_path_cov_xy = np.zeros((len(one_path_angles), 2, 2))
        delta_t = 2 * np.pi / num_modes
        std_t = r * delta_t * self.K
        for i, rad in enumerate(one_path_angles):
            v_r = np.array([[np.cos(rad), np.sin(rad)]])
            v_t = np.array([[-np.sin(rad), np.cos(rad)]])
            tmp_cov = v_r.T @ v_r * std_r ** 2 + v_t.T @ v_t * std_t ** 2
            one_path_cov_xy[i] = tmp_cov
        all_angles = np.tile(one_path_angles, len(rbt_samples_xy))
        displaced = np.array([r * np.cos(all_angles), r * np.sin(all_angles)]).T
        lmk_means = np.repeat(rbt_samples_xy, repeats=num_modes, axis=0) + displaced

        for i in range(self.n):
            self.pathID2lmkSOG[i][lmk_var] = SOG(np.ones(num_modes) / num_modes,
                                                 lmk_means[i * num_modes:(i + 1) * num_modes, :],
                                                 one_path_cov_xy)

    def prune_low_weight_modes(self, sog):
        hi_weight = sog.weights > (1.0 - self.max_drop_mode_weight) / self.n
        sog.weights = sog.weights[hi_weight]
        sog.weights = sog.weights / np.sum(sog.weights)
        sog.means = sog.means[hi_weight]
        sog.covs = sog.covs[hi_weight]
        # if len(sog.weights) < len(hi_weight):
        #     print(f"pruned {len(hi_weight)-len(sog.weights)} modes")

    def update_single_sog(self, cur_sog, rbt_samples_xy, r, var_r):
        innovations = r - np.linalg.norm(rbt_samples_xy - cur_sog.means, axis=1)
        jocobs = (cur_sog.means - rbt_samples_xy) / np.linalg.norm(cur_sog.means - rbt_samples_xy,
                                                                   axis=1).reshape((-1, 1))
        cov_xy_innov_t = np.einsum("ik,ikj->ij", jocobs, cur_sog.covs)
        innov_covs = np.einsum("ij,ji->i", cov_xy_innov_t, jocobs.T) + var_r
        cov_xy_innov = cov_xy_innov_t.T

        Kalman_gain = cov_xy_innov / innov_covs
        tmp_mean = cur_sog.means.T + Kalman_gain * innovations
        tmp_mean = tmp_mean.T
        tmp_covs = cur_sog.covs - np.einsum("ij,jk->jik", Kalman_gain, cov_xy_innov_t)

        tmp_err = r - np.linalg.norm(tmp_mean - rbt_samples_xy, axis=1)
        tmp_mode_weights = cur_sog.weights * np.exp(-tmp_err ** 2 / 2.0 / innov_covs) / np.sqrt(innov_covs)

        obs_likelihood = np.sum(tmp_mode_weights)

        cur_sog.means = tmp_mean
        cur_sog.covs = tmp_covs
        cur_sog.weights = tmp_mode_weights / obs_likelihood
        # remove low-weight modes from the SOGs
        self.prune_low_weight_modes(cur_sog)
        return obs_likelihood

    def add_lmk_meas_factor(self, f: SE2R2RangeGaussianLikelihoodFactor):
        self.all_factors.append(f)
        var1, var2 = f.vars
        if var2 not in self.vars:
            # new landmark
            self.init_sog(f)
            self.vars.append(var2)
            self.lmk_vars.append(var2)
        else:
            # existing landmark
            # init temp weight list
            r = f.observation[0]
            rbt_samples_xy = self.rbt_samples_xy(var1)
            var_r = f.sigma ** 2

            if self.parallel_n_jobs == 1:
                obs_likelihoods = np.zeros(self.n)
                for i in range(self.n):
                    # update the SOG
                    cur_sog = self.pathID2lmkSOG[i][var2]
                    obs_likelihoods[i] = self.update_single_sog(cur_sog, rbt_samples_xy[i], r, var_r)
            else:
                p_results = Parallel(n_jobs=self.parallel_n_jobs)(
                    delayed(self.update_single_sog)(self.pathID2lmkSOG[i][var2], rbt_samples_xy[i], r, var_r) for i in
                    range(self.n))
                obs_likelihoods = np.array(p_results)

            # update the map (SOGs of the landmark)
            self.weights = self.weights * obs_likelihoods
            self.weights = self.weights / np.sum(self.weights)

    # def add_lmk_meas_factor(self, f: SE2R2RangeGaussianLikelihoodFactor):
    #     self.all_factors.append(f)
    #     var1, var2 = f.vars
    #     if var2 not in self.vars:
    #         # new landmark
    #         self.init_sog(f)
    #         self.vars.append(var2)
    #         self.lmk_vars.append(var2)
    #     else:
    #         # existing landmark
    #         # init temp weight list
    #         obs_likelihoods = np.zeros(self.n)
    #
    #         r = f.observation[0]
    #         rbt_samples_xy = self.rbt_samples_xy(var1)
    #         var_r = f.sigma ** 2
    #
    #         if self.parallel_n_jobs == 1:
    #             for i in range(self.n):
    #                 # update the SOG
    #                 cur_sog = self.pathID2lmkSOG[i][var2]
    #                 # batch update of EKFs
    #                 innovations = r - np.linalg.norm(rbt_samples_xy[i] - cur_sog.means, axis=1)
    #                 jocobs = (cur_sog.means - rbt_samples_xy[i]) / np.linalg.norm(cur_sog.means - rbt_samples_xy[i],
    #                                                                               axis=1).reshape((-1, 1))
    #                 cov_xy_innov_t = np.einsum("ik,ikj->ij", jocobs, cur_sog.covs)
    #                 innov_covs = np.einsum("ij,ji->i", cov_xy_innov_t, jocobs.T) + var_r
    #                 cov_xy_innov = cov_xy_innov_t.T
    #
    #                 Kalman_gain = cov_xy_innov / innov_covs
    #                 tmp_mean = cur_sog.means.T + Kalman_gain * innovations
    #                 tmp_mean = tmp_mean.T
    #                 tmp_covs = cur_sog.covs - np.einsum("ij,jk->jik", Kalman_gain, cov_xy_innov_t)
    #
    #                 tmp_err = r - np.linalg.norm(tmp_mean - rbt_samples_xy[i], axis=1)
    #                 tmp_mode_weights = cur_sog.weights * np.exp(-tmp_err ** 2 / 2.0 / innov_covs) / np.sqrt(innov_covs)
    #
    #                 obs_likelihoods[i] = np.sum(tmp_mode_weights)
    #
    #                 cur_sog.means = tmp_mean
    #                 cur_sog.covs = tmp_covs
    #                 cur_sog.weights = tmp_mode_weights / obs_likelihoods[i]
    #                 # remove low-weight modes from the SOGs
    #                 self.prune_low_weight_modes(cur_sog)
    #         else:
    #             pass
    #         # update the map (SOGs of the landmark)
    #         self.weights = self.weights * obs_likelihoods
    #         self.weights = self.weights / np.sum(self.weights)

    def resample(self):
        # resample rbt path and map
        # resample_idx = systematic_resample(self.samples, self.weights, return_idx=True)
        # self.samples = self.samples[resample_idx]
        # self.weights = np.ones(self.n) / self.n
        # new_lmk_sog = []
        # for i in range(self.n):
        #     new_lmk_sog.append(deepcopy(self.pathID2lmkSOG[resample_idx[i]]))
        # self.pathID2lmkSOG = new_lmk_sog

        resample_idx = systematic_resample(self.samples, self.weights, return_idx=True)
        self.samples = self.samples[resample_idx]
        self.weights = np.ones(self.n) / self.n

        # new_lmk_sog = ([deepcopy(self.pathID2lmkSOG[resample_idx[i]]) for i in range(self.n)])
        new_lmk_sog = [copyLmk2SOG(self.pathID2lmkSOG[resample_idx[i]]) for i in range(self.n)]
        # new_lmk_sog = pickle.loads(pickle.dumps(new_lmk_sog, -1))
        self.pathID2lmkSOG = new_lmk_sog

    def add_prior_factor(self, f):
        self.all_factors.append(f)
        var = f.vars[0]
        s = f.sample(self.n)
        self.update_var_sample(s, var)

    def sample_lmk(self, sample_num):
        num_per_path = np.random.multinomial(sample_num, self.weights)
        sample_per_path = []
        lmk2samples = {k: np.empty((0, k.dim)) for k in self.lmk_vars}
        rbt2samples = {k: np.empty((0, k.dim)) for k in self.vars if k not in self.lmk_vars}
        for i in range(self.n):
            if num_per_path[i] > 0:
                sample_on_path = {}
                for k, v in self.pathID2lmkSOG[i].items():
                    num_per_mode = np.random.multinomial(num_per_path[i], v.weights)
                    sample_on_path[k] = np.vstack([np.random.multivariate_normal(v.means[mod_id],
                                                                                 v.covs[mod_id],
                                                                                 num_per_mode[mod_id])
                                                   for mod_id in range(len(num_per_mode))])
                    lmk2samples[k] = np.vstack([lmk2samples[k], sample_on_path[k]])
                sample_per_path.append(sample_on_path)
            else:
                sample_per_path.append([])

            for k, v in rbt2samples.items():
                rbt2samples[k] = np.vstack([v, np.tile(self.samples[i, self.var2slice[k]], (num_per_path[i], 1))])
        return sample_per_path, lmk2samples, rbt2samples

    def getAvgModeNum(self):
        res = {}
        for lmk2sog in self.pathID2lmkSOG:
            for lmk, sog in lmk2sog.items():
                if lmk.name not in res:
                    res[lmk.name] = [len(sog.weights)]
                else:
                    res[lmk.name].append(len(sog.weights))
        for lmk, v in res.items():
            res[lmk] = np.mean(v)
        return res
