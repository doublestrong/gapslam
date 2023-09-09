import numpy as np
import gtsam
from abc import ABCMeta
from typing import List, Union
# from numba import jit

from geometry.ThreeDimension import SE3Pose
from geometry.TwoDimension import SE2Pose
from slam.Variables import Variable

manifoldstr2dim = {"R2Variable": 2,
                   "R3Variable": 3,
                   "SE2Variable": 3,
                   "SE3Variable": 6}

class GaussianSolverWrapper(metaclass=ABCMeta):
    def update_factor_value(self, factors, values, *args, **kwargs):
        pass
    def update_estimates(self, *args, **kwargs):
        pass
    def update_point_estimate(self, *args, **kwargs):
        pass
    def update_covariance_estimate(self, *args, **kwargs):
        pass
    def reinitialize(self, var, value, factor_indices, *args, **kwargs):
        pass
    def get_point_estimate(self, var, vartype, *args, **kwargs):
        pass
    def get_joint_covariance(self, vars, vartypes, *args, **kwargs):
        pass
    def get_single_covariance(self, var, *args, **kwargs):
        pass
    def get_samples(self, vars, vartypes: List[str], sample_num, parallel, *args, **kwargs):
        pass
    def get_factor_size(self):
        pass
    def remove_factors(self, factor_indices):
        pass

class gtsamWrapper(GaussianSolverWrapper):
    def __init__(self):
        # parameters = gtsam.ISAM2Params()
        # parameters.setRelinearizeThreshold(0.01)
        # parameters.relinearizeSkip = 1
        # self._solver = gtsam.ISAM2(parameters)
        self._solver = gtsam.ISAM2()
        self._point_estimate = None
        self._marginals = None

    @property
    def isam(self):
        return self._solver

    @property
    def point(self):
        return self._point_estimate

    @property
    def marg(self):
        return self._marginals

    @property
    def var_ordering(self):
        return self._point_estimate.keys()

    def update_factor_value(self, factors: gtsam.NonlinearFactorGraph,
                            values: gtsam.Values,
                            *args, **kwargs):
        self._solver.update(factors, values)

    def update_estimates(self):
        self.update_point_estimate()
        self.update_covariance_estimate()

    def update_point_estimate(self):
        try:
            # it is possible that some exceptinos may happen such as indeterminant systems
            self._point_estimate = self._solver.calculateEstimate()
        except Exception as e:
            print(str(e))

    def update_covariance_estimate(self):
        # check alternatives if the default Cholesky method prones to be unstable
        # https://gtsam.org/doxygen/4.0.0/a03711.html#ac5829b5b43587ab6316671def4a9d491
        try:
            self._marginals = gtsam.Marginals(self._solver.getFactorsUnsafe(), self._point_estimate)
        except Exception as e:
            print(str(e))

    def reinitialize(self, var: gtsam.Symbol,
                     value: Union[gtsam.Pose2, gtsam.Point2],
                     factor_indices: List,
                     *args,
                     **kwargs):
        """
        This function will re-initialize a variable, lmk, by the given new value, reinit_val.
        params:
            var: the variable key
            factor_indices: indices of factors in isam that connect to the var
            value: a gtsam geomtry object, e.g., gtsam.Pose2, gtsam.Point2, etc.
        return
            a list of the updated indices of factors that connect to the var
        """
        return gtsam_reinit(self._solver, var, factor_indices, value)

    def get_single_covariance(self, var: gtsam.Symbol, *args, **kwargs):
        # remember the returned matrix lives in a product manifold
        return self._marginals.marginalCovariance(var)

    def get_joint_covariance(self, vars: List[gtsam.Symbol], vartypes: List[str] = None, reorder=False,
                             *args, **kwargs):
        # 1. remember the returned matrix lives in a product manifold
        # 2. note that if vars is all variables, the variable ordering
        # in the resulting matrix follows the ordering in the point estimate,
        # rather than the ordering in vars
        if len(vars) == len(self.var_ordering) or reorder:
            if vartypes is None:
                raise ValueError("Vartype is required to get full joint covariance.")
            dims = [manifoldstr2dim[vt] for vt in vartypes]
            postidx = np.cumsum(dims)
            preidx = np.zeros_like(postidx, dtype=int)
            preidx[1:] = postidx[:-1]
            dft_cov = self._marginals.jointMarginalCovariance(vars).fullMatrix()
            dft_vars = list(self.var_ordering)
            dft_dims = [manifoldstr2dim[vartypes[vars.index(v)]] for v in self.var_ordering]
            dft_postidx = np.cumsum(dft_dims)
            dft_preidx = np.zeros_like(dft_postidx, dtype=int)
            dft_preidx[1:] = dft_postidx[:-1]

            cov = np.zeros((sum(dims), sum(dims)))
            for i, vi in enumerate(vars):
                for j, vj in enumerate(vars):
                    dft_i = dft_vars.index(vi)
                    dft_j = dft_vars.index(vj)
                    cov[preidx[i]: postidx[i], preidx[j]: postidx[j]] = \
                        dft_cov[dft_preidx[dft_i]: dft_postidx[dft_i], dft_preidx[dft_j]: dft_postidx[dft_j]]
            return cov
        else:
            return self._marginals.jointMarginalCovariance(vars).fullMatrix()

    def get_point_estimate(self, var: gtsam.Symbol, vartype: str, *args, **kwargs):
        if vartype == "SE2Variable":
            return self._point_estimate.atPose2(var)
        elif vartype == "R2Variable":
            return self._point_estimate.atPoint2(var)
        elif vartype == "SE3Variable":
            return self._point_estimate.atPose3(var)
        elif vartype == "R3Variable":
            return self._point_estimate.atPoint3(var)
        else:
            raise NotImplementedError(f"Unknown type {vartype}")


    def get_samples_dict(self, vars: List[Variable], sample_num):
        """
        return a dictonary of samples. The shape of the samples is (# of samples, # of dimension)
        """
        vars_keys = [v.key for v in vars]
        vartypes = [v.__class__.__name__ for v in vars]
        res_dict = {}

        var_len = len(vars)
        dims = [v.dim for v in vars]
        dim = sum(dims)
        joint = self.get_joint_covariance(vars_keys, vartypes)
        noise = np.random.multivariate_normal(mean=np.zeros(dim),
                                              cov=joint, size=(sample_num,))
        postidx = np.cumsum(dims)
        preidx = np.zeros_like(postidx, dtype=int)
        preidx[1:] = postidx[:-1]

        poseidx = [i for i in range(var_len) if vartypes[i][:2] == "SE"]
        otheridx = list(set(range(var_len)) - set(poseidx))
        for i in otheridx:
            if vartypes[i] == "R2Variable":
                mean = self._point_estimate.atPoint2(vars[i].key)
                res_dict[vars[i]] = mean + noise[:,preidx[i]: postidx[i]]
            elif vartypes[i] == "R3Variable":
                mean = self._point_estimate.atPoint3(vars[i].key)
                res_dict[vars[i]] = mean + noise[:,preidx[i]: postidx[i]]
            else:
                raise NotImplementedError(f"Unknown manifold type {vartypes[i]}")

        for i in poseidx:
            if vartypes[i] == "SE2Variable":
                mean = self._point_estimate.atPose2(vars[i].key)
                perturbed = np.zeros((sample_num, 3))
                for j in range(sample_num):
                    tmp_pose = mean * gtsam.Pose2.Expmap(noise[j, preidx[i]: postidx[i]])
                    perturbed[j] = tmp_pose.x(), tmp_pose.y(), tmp_pose.theta()
                res_dict[vars[i]] = perturbed
            elif vartypes[i] == "SE3Variable":
                mean = self._point_estimate.atPose3(vars[i].key)
                perturbed = np.zeros((sample_num, 4, 4))
                for j in range(sample_num):
                    tmp_pose = mean * gtsam.Pose3.Expmap(noise[j, preidx[i]: postidx[i]])
                    perturbed[j] = tmp_pose.matrix()
                res_dict[vars[i]] = perturbed
            else:
                raise NotImplementedError(f"Unknown manifold type {vartypes[i]}")
        return res_dict

    # @jit(nopython=True,cache=False)
    def get_samples(self, vars: List[gtsam.Symbol], vartypes: List[str], sample_num, *args, **kwargs):
        """
        return an array of samples. The shape of the samples is (# of samples, # of dimension)
        """
        var_len = len(vars)
        dims = [manifoldstr2dim[vt] for vt in vartypes]
        dim = sum(dims)
        res = np.zeros((sample_num, dim))
        joint = self.get_joint_covariance(vars, vartypes)
        noise = np.random.multivariate_normal(mean=np.zeros(dim),
                                              cov=joint, size=(sample_num,))
        postidx = np.cumsum(dims)
        preidx = np.zeros_like(postidx, dtype=int)
        preidx[1:] = postidx[:-1]

        poseidx = [i for i in range(var_len) if vartypes[i][:2] == "SE"]
        otheridx = list(set(range(var_len)) - set(poseidx))
        for i in otheridx:
            if vartypes[i] == "R2Variable":
                mean = self._point_estimate.atPoint2(vars[i])
                res[:, preidx[i]: postidx[i]] = mean + noise[:,preidx[i]: postidx[i]]
            elif vartypes[i] == "R3Variable":
                mean = self._point_estimate.atPoint3(vars[i])
                res[:, preidx[i]: postidx[i]] = mean + noise[:,preidx[i]: postidx[i]]
            else:
                raise NotImplementedError(f"Unknown manifold type {vartypes[i]}")

        for j in range(sample_num):
            for i in poseidx:
                if vartypes[i] == "SE2Variable":
                    mean = self._point_estimate.atPose2(vars[i])
                    mean = SE2Pose(mean.x(), mean.y(), mean.theta())
                    # TODO: vectorize this step to get rid of the for loop
                    res[j, preidx[i]: postidx[i]] = (mean * SE2Pose.by_exp_map(noise[j,preidx[i]: postidx[i]])).array
                elif vartypes[i] == "SE3Variable":
                    mean = self._point_estimate.atPose3(vars[i])
                    mean = SE3Pose(mean.matrix())
                    # TODO: vectorize this step to get rid of the for loop
                    res[j, preidx[i]: postidx[i]] = (mean * SE3Pose.by_exp_map(noise[j,preidx[i]: postidx[i]])).array
                else:
                    raise NotImplementedError(f"Unknown manifold type {vartypes[i]}")
        return res

    def get_factor_size(self):
        return self.isam.getFactorsUnsafe().size()

    def remove_factors(self, factor_indices: List[int]):
        self._solver.update(gtsam.NonlinearFactorGraph(), gtsam.Values(), factor_indices)

def gtsam_reinit(isam: gtsam.ISAM2, lmk: gtsam.Symbol, factor_indices: List, reinit_val):
    """
    This function will manually re-initialize a variable, lmk, by the given new value, reinit_val.
    params:
        isam: the solver
        lmk: the variable key
        factor_indices: indices of factors in isam that connect to the lmk
        reinit_val: a gtsam geomtry object, e.g., gtsam.Pose2, gtsam.Point2, etc.
    """
    temp_graph = gtsam.NonlinearFactorGraph()
    for f_idx in factor_indices:
        temp_graph.add(isam.getFactorsUnsafe().at(f_idx))
    new_val = gtsam.Values()
    new_val.insert(lmk, reinit_val)

    # remove factors
    isam.update(gtsam.NonlinearFactorGraph(), gtsam.Values(), factor_indices)

    # re-add factors back with new initial values
    isam.update(temp_graph, new_val)
    factor_indices = list(range(isam.getFactorsUnsafe().size() - temp_graph.size(), isam.getFactorsUnsafe().size()))
    return factor_indices