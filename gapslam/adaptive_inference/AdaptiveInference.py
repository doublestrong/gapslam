import numpy as np
import gtsam

from adaptive_inference.GaussianSolverWrapper import gtsamWrapper
from adaptive_inference.GTSAM_help import to_gtsam_factor, init_gtsam_var, boostrap_init_factors, reg_factor
from adaptive_inference.utils import systematic_resample, to_Key
from slam.Variables import R2Variable, SE2Variable, VariableType, Variable
from factors.Factors import UnaryFactor, BinaryFactor, SE2R2RangeGaussianLikelihoodFactor, \
    UnaryR2RangeGaussianPriorFactor, UnaryFactorMixture
from typing import Union, Dict
from scipy.stats import circstd

from utils.Functions import theta_to_pipi

default_ambiguous_factor_class = {'SE2R2RangeGaussianLikelihoodFactor',
                                  'SE2R2BearingLikelihoodFactor'}
default_bootstrap_init_class = set(boostrap_init_factors.keys())

class AdaptiveInferenceSolver:
    """
    A SLAM solver that combines both parameteric and non-parametric approaches to represent poseteriors.
    """

    def __init__(self, ambiguous_factor_class=None,
                 bootstrap_init_class=None,
                 lmk_factor_size: int = np.inf,
                 nonGaussian_args=None,
                 reg_scale=0):
        """
        params:
            ambiguous_factor_class: multi-modal or underdetermined factor classes
            bootstrap_init_class: factor classes in ambiguous_factor_class that will be read into gtsam from the beginning
            lmk_factor_size: maximal size of landmark factors involved in updating non-Gaussian belief of the landmark
            nonGaussian_method: methods for updating non-Gaussian landmark belief
        """
        if ambiguous_factor_class is None:
            self.ambiguous_factor_class = default_ambiguous_factor_class
        else:
            self.ambiguous_factor_class = ambiguous_factor_class
        if bootstrap_init_class is None:
            self.bootstrap_init_class = default_bootstrap_init_class
        else:
            self.bootstrap_init_class = bootstrap_init_class
        if nonGaussian_args == None:
            self.nonGaussian_args = {"method": "importance sampling", "ifr_sample_num": 1000}
        else:
            self.nonGaussian_args = nonGaussian_args
        if not self.ambiguous_factor_class.issubset(
                default_ambiguous_factor_class) or not self.bootstrap_init_class.issubset(default_bootstrap_init_class):
            set_diff1 = self.ambiguous_factor_class - default_ambiguous_factor_class
            set_diff2 = self.bootstrap_init_class - default_bootstrap_init_class
            raise NotImplementedError(
                f"Unknown ambiguous factors: {set_diff1}, unknown boostrap_init_factors: {set_diff2}")
        self.gaussian_solver = gtsamWrapper()
        # largest size of landmark factors for updating non-Gaussian belief
        self.lmk_factor_size = lmk_factor_size
        self.lmk_update_fun = None
        if self.nonGaussian_args["method"] == "importance sampling":
            self.lmk_update_fun = self.importance_sampling_update
            self.posterior_fun = self.importance_sampling_posterior
        elif self.nonGaussian_args["method"] == "rpf":
            self.lmk_update_fun = self.rpf_lmk_update

        # cache factor graph and initial value in case of indeterminant linear systems
        self.cached_lmk2values = {}
        self.cached_lmk2graph = {}

        self.ambiguous_lmk = []

        self.uninit_lmk = []
        # factors that have not been used for non-Gaussian inference
        self.lmk2new_factors = {}
        # factors that have been used for non-Gaussian inference
        self.lmk2old_factors = {}
        # all factors to a landmark
        self.lmk2factors = {}
        # store samples and logratio for post-processing and incremental update
        self.lmk2sample_logratio = {}
        # store posterior samples
        self.lmk2post_samples = {}
        # store isam marginals of robot poses used for updating landmark belief last time
        self.lmk2rbt_points = {}
        # store lmk factor indices in isam if it is a variable in isam
        self.lmk2isam_factor_indices = {}
        # variables in isam
        self.gaussian_vars = []

        # regularization cov scale
        self.reg_scale = reg_scale
        self.lmk_reg_prior = {}

        self.all_factors = set()

    def get_gs_factor_size(self):
        return self.gaussian_solver.get_factor_size()

    def add_factor(self, factor: Union[UnaryFactor, BinaryFactor], do_point_estimate=True):
        self.all_factors.add(factor)
        # flags for non-Gaussian inference
        flag_ambiguous_lmk, flag_uninit_lmk = False, False

        f_vars = factor.vars
        new_vars = list(set(f_vars) - set(self.gaussian_vars))
        old_vars = list(set(f_vars).intersection(set(self.gaussian_vars)))
        f_class = factor.__class__.__name__

        if len(new_vars) == 0:
            fg = gtsam.NonlinearFactorGraph()
            values = gtsam.Values()
            fg.add(to_gtsam_factor(factor))
            self.gaussian_solver.update_factor_value(fg, values)
            if f_class in self.ambiguous_factor_class:
                # existing ambiguous landmarks should have been in self.ambiguous_lmk
                ambi_lmks = list(set(f_vars).intersection(self.ambiguous_lmk))
                assert len(ambi_lmks) == 1
                # track factor indices in isam
                self.lmk2isam_factor_indices[ambi_lmks[0]].append(self.get_gs_factor_size()-1)
                # add for non-Gaussian inference
                if ambi_lmks[0] in self.lmk2new_factors:
                    self.lmk2new_factors[ambi_lmks[0]].append(factor)
                else:
                    self.lmk2new_factors[ambi_lmks[0]] = [factor]
                self.lmk2factors[ambi_lmks[0]].append(factor)
                flag_ambiguous_lmk = True
        elif len(new_vars) == 1:
            new_var = new_vars[0]
            if f_class in self.bootstrap_init_class:
                # find initial values and add factors for gtsam
                # Bootstrapped initialization
                if new_var in self.cached_lmk2values:
                    # this var wasn't added to the gaussian factor graph due to exceptions
                    # this var has showed up before but failed to be updated in isam
                    gs_f = to_gtsam_factor(factor)
                    self.cached_lmk2graph[new_var].add(gs_f)
                    update_flag = False
                    try:
                        self.gaussian_solver.update_factor_value(self.cached_lmk2graph[new_var],
                                                                 self.cached_lmk2values[new_var])
                        # manage add new_var and cached factors to the gaussian solver
                        update_flag = True
                    except Exception as e:
                        # the new factor still cannot initialize the new_var
                        print(str(e))
                    if update_flag:
                        self.gaussian_vars.append(new_var)
                        if f_class in self.ambiguous_factor_class and new_var.type == VariableType.Landmark:
                            self.lmk2isam_factor_indices[new_var] = \
                                list(range(self.get_gs_factor_size() - self.cached_lmk2graph[new_var].size(),
                                           self.get_gs_factor_size()))
                        # clear cached gaussian factor graphs
                        del self.cached_lmk2graph[new_var]
                        del self.cached_lmk2values[new_var]
                    else:
                        # note that the factors and values are added to gtsam
                        # remove them from gtsam to prevent unexpected errors in subsequent computations
                        self.gaussian_solver.remove_factors(
                            list(range(self.get_gs_factor_size() - self.cached_lmk2graph[new_var].size(),
                                       self.get_gs_factor_size())))
                        flag_uninit_lmk = True
                else:
                    # a totally new factor
                    old_var = None
                    if len(old_vars) == 1:
                        old_var = old_vars[0]
                        # we require that old_var must be a robot pose;
                        # in practice, an odom factor must be read in before a robot-landmark factor
                        assert old_var.type is not VariableType.Landmark
                    init_val = init_gtsam_var(factor, self.gaussian_solver, old_var)
                    new_key = to_Key(new_var)
                    fg = gtsam.NonlinearFactorGraph()
                    values = gtsam.Values()
                    values.insert(new_key, init_val)
                    fg.add(to_gtsam_factor(factor))
                    update_flag = False
                    try:
                        self.gaussian_solver.update_factor_value(fg,
                                                                 values)
                        # manage add new_var and cached factors to the gaussian solver
                        update_flag = True
                    except Exception as e:
                        # the new factor still cannot initialize the new_var
                        print(str(e))
                    if update_flag:
                        self.gaussian_vars.append(new_var)
                        if f_class in self.ambiguous_factor_class and new_var.type == VariableType.Landmark:
                            self.lmk2isam_factor_indices[new_var] = [self.get_gs_factor_size()-1]
                    else:
                        assert f_class in self.ambiguous_factor_class
                        if self.reg_scale != 0:
                            # regularize landmark with an uninformative prior
                            self.lmk_reg_prior[new_var] = reg_factor(new_key, init_val, self.reg_scale, new_var)
                            self.gaussian_solver.remove_factors([self.get_gs_factor_size()-1])
                            fg.add(self.lmk_reg_prior[new_var])
                            self.gaussian_solver.update_factor_value(fg, values)
                            self.lmk2isam_factor_indices[new_var] = [self.get_gs_factor_size()-2,
                                                                     self.get_gs_factor_size()-1]
                            self.gaussian_vars.append(new_var)
                        else:
                            # remove indeterminant factors from gtsam
                            self.gaussian_solver.remove_factors([self.get_gs_factor_size()-1])
                            self.cached_lmk2graph[new_var] = fg
                            self.cached_lmk2values[new_var] = values
                            flag_uninit_lmk = True
            elif f_class in self.ambiguous_factor_class and new_var.type == VariableType.Landmark:
                if new_var not in self.uninit_lmk:
                    self.uninit_lmk.append(new_var)
                flag_uninit_lmk = True
            else:
                raise NotImplementedError(f"Initialization of {factor.__repr__()} has not been considered.")

            # bookkeeping ambiguous landmarks
            if f_class in self.ambiguous_factor_class and new_var.type == VariableType.Landmark:
                if new_var not in self.ambiguous_lmk:
                    self.ambiguous_lmk.append(new_var)
                if new_var not in self.lmk2new_factors:
                    self.lmk2new_factors[new_var] = [factor]
                else:
                    self.lmk2new_factors[new_var].append(factor)
                if new_var not in self.lmk2factors:
                    self.lmk2factors[new_var] = [factor]
                else:
                    self.lmk2factors[new_var].append(factor)
                # sanity check: if a var is in uninit_lmk, its isam_factor_indices is supposed to be empty
                flag_ambiguous_lmk = True
        else:
            raise ValueError(f"The input factor {factor.__repr__()} is disconnected from the factor graph.")

        if do_point_estimate:
            self.gaussian_solver.update_point_estimate()
        return flag_ambiguous_lmk, flag_uninit_lmk

    def update_logratio(self, f, lmk, lmk_samples, logratio):
        if f.var2 == lmk:
            rbt_var = f.var1
            rbt_key = to_Key(rbt_var)
            rbt_samples = self.gaussian_solver.get_samples(vars=[rbt_key],
                                                           vartypes=[rbt_var.__class__.__name__],
                                                           sample_num=self.nonGaussian_args["ifr_sample_num"])
            tmp_samples = np.hstack((rbt_samples, lmk_samples))
        else:
            rbt_var = f.var2
            rbt_key = to_Key(rbt_var)
            rbt_samples = self.gaussian_solver.get_samples(vars=[rbt_key],
                                                           vartypes=[rbt_var.__class__.__name__],
                                                           sample_num=self.nonGaussian_args["ifr_sample_num"])
            tmp_samples = np.hstack((lmk_samples, rbt_samples))
        logratio += f.log_pdf(tmp_samples)
        return logratio

    def init_samples(self, f, lmk):
        if f.var2 == lmk:
            rbt_var = f.var1
            rbt_key = to_Key(rbt_var)
            rbt_samples = self.gaussian_solver.get_samples(vars=[rbt_key],
                                                           vartypes=[rbt_var.__class__.__name__],
                                                           sample_num=self.nonGaussian_args["ifr_sample_num"])
            lmk_samples = f.sample(var1=rbt_samples)
        else:
            rbt_var = f.var2
            rbt_key = to_Key(rbt_var)
            rbt_samples = self.gaussian_solver.get_samples(vars=[rbt_key],
                                                           vartypes=[rbt_var.__class__.__name__],
                                                           sample_num=self.nonGaussian_args["ifr_sample_num"])
            lmk_samples = f.sample(var2=rbt_samples)
        return lmk_samples, np.zeros(self.nonGaussian_args["ifr_sample_num"])

    def importance_sampling_update(self, incremental_update = True):
        if incremental_update:
            for lmk, factors in self.lmk2new_factors.items():
                if lmk in self.lmk2sample_logratio:
                    lmk_samples, logratio = self.lmk2sample_logratio[lmk]
                    for f in factors:
                        logratio = self.update_logratio(f, lmk, lmk_samples, logratio)
                    # substract a mean to avoid very large logratio
                    # self.lmk2sample_logratio[lmk][1] = logratio - np.mean(logratio)
                else:
                    lmk_samples, logratio = self.init_samples(factors[0], lmk)
                    for f in factors[1:]:
                        logratio = self.update_logratio(f, lmk, lmk_samples, logratio)
                    # substract a mean to avoid very large logratio
                    # self.lmk2sample_logratio[lmk] = [lmk_samples, logratio - np.mean(logratio)]
                weights = np.exp(logratio)
                weights = weights / np.sum(weights)
                lmk_samples = systematic_resample(lmk_samples, weights)
                self.lmk2sample_logratio[lmk] = [lmk_samples, np.zeros(len(lmk_samples))]

                if lmk in self.lmk2old_factors:
                    self.lmk2old_factors[lmk] += factors
                else:
                    self.lmk2old_factors[lmk] = factors
            self.lmk2new_factors = {}
        else:
            raise NotImplementedError("Not implemented yet.")

    def importance_sampling_posterior(self):
        vars = []
        samples = np.empty((self.nonGaussian_args['ifr_sample_num'],0))
        for lmk, (lmk_samples, logratio) in self.lmk2sample_logratio.items():
            weights = np.exp(logratio)
            weights = weights / np.sum(weights)
            if np.isnan(np.sum(weights)):
                weights = np.nan_to_num(weights, nan=1.0)
                weights = weights / np.sum(weights)
            tmp_samples = systematic_resample(lmk_samples, weights)
            samples = np.hstack((samples, tmp_samples))
            vars.append(lmk)
            self.lmk2post_samples[lmk] = tmp_samples
        keys = []
        vartypes = []
        for var in self.gaussian_vars:
            if var not in self.lmk2sample_logratio:
                vars.append(var)
                keys.append(to_Key(var))
                vartypes.append(var.__class__.__name__)
        samples = np.hstack((samples, self.gaussian_solver.get_samples(keys, vartypes, sample_num=samples.shape[0])))
        return samples, vars

    def reinit_check(self, reinit_trans_tol = 1.0, reinit_orient_tol = .3, reinit_weight_tol = 1.5):
        # perform reinit for ambiguous landmarks or init for uninit landmarks
        reinit_lmk_list = []
        for lmk, (lmk_samples, logratio) in self.lmk2sample_logratio.items():
            weights = np.exp(logratio)
            max_weight = max(weights)
            max_idx = np.argmax(weights)
            max_sample = lmk_samples[max_idx]
            if lmk in self.gaussian_vars:
                # check if reinit and landmarkactually
                lmk_key = to_Key(lmk)
                lmk_point = self.gaussian_solver.get_point_estimate(lmk_key,lmk.__class__.__name__)
                if isinstance(lmk, R2Variable):
                    nearest_idx = np.linalg.norm(lmk_samples - lmk_point, axis=1).argmin()
                    nearest_weight = weights[nearest_idx]
                    distance = np.linalg.norm(lmk_point - max_sample)
                    if distance > reinit_trans_tol and max_weight/nearest_weight > reinit_weight_tol:
                        # the weight tol is to prevent frequent reinitialization
                        # in cases with many equally weighted samples
                        self.lmk2isam_factor_indices[lmk] = \
                            self.gaussian_solver.reinitialize(lmk_key,
                                                              gtsam.Point2(*max_sample),
                                                              self.lmk2isam_factor_indices[lmk])
                        reinit_lmk_list.append(lmk)
                elif isinstance(lmk, SE2Variable):
                    trans_err = np.linalg.norm([max_sample[0]-lmk_point.x(),
                                                max_sample[1]-lmk_point.y()])
                    orient_err = abs(theta_to_pipi(max_sample[2] - lmk_point.theta()))
                    if trans_err > reinit_trans_tol or orient_err > reinit_orient_tol:
                        self.lmk2isam_factor_indices[lmk] = \
                            self.gaussian_solver.reinitialize(lmk_key,
                                                              gtsam.Pose2(*max_sample),
                                                              self.lmk2isam_factor_indices[lmk])
                        reinit_lmk_list.append(lmk)
                else:
                    raise NotImplementedError("Not implemented variable types for reinitialization.")
            else:
                assert (lmk in self.uninit_lmk or lmk in self.cached_lmk2values) and (
                        lmk not in self.lmk2isam_factor_indices or len(self.lmk2isam_factor_indices[lmk]) == 0)
                if lmk in self.lmk2post_samples:
                    equal_samples = self.lmk2post_samples[lmk]
                else:
                    weights = np.exp(logratio)
                    weights = weights / np.sum(weights)
                    equal_samples = systematic_resample(lmk_samples, weights)
                if isinstance(lmk, R2Variable):
                    xstd, ystd = np.std(equal_samples, axis=0)
                    if xstd < reinit_trans_tol and ystd < reinit_trans_tol:
                        # indicate the belief of the landmark location is fairly concentrated
                        # it is ready to be initialized
                        factors = self.lmk2factors[lmk]
                        fg = gtsam.NonlinearFactorGraph()
                        for i, f in enumerate(factors):
                            fg.add(to_gtsam_factor(f))
                            # self.lmk2isam_factor_indices[lmk].append(i + self.gaussian_solver.get_factor_size())
                        values = gtsam.Values()
                        lmk_key = to_Key(lmk)
                        values.insert(lmk_key, gtsam.Point2(*max_sample))

                        update_flag = False
                        try:
                            self.gaussian_solver.update_factor_value(fg,
                                                                     values)
                            # manage add new_var and cached factors to the gaussian solver
                            update_flag = True
                        except Exception as e:
                            # the new factor still cannot initialize the new_var
                            print(str(e))
                        if update_flag:
                            reinit_lmk_list.append(lmk)
                            self.gaussian_vars.append(lmk)
                            self.lmk2isam_factor_indices[lmk] = \
                                list(range(self.get_gs_factor_size() - fg.size(),
                                           self.get_gs_factor_size()))
                            if lmk in self.uninit_lmk:
                                self.uninit_lmk.remove(lmk)
                            if lmk in self.cached_lmk2values:
                                assert self.cached_lmk2graph[lmk].size() <= fg.size()
                                del self.cached_lmk2values[lmk]
                                del self.cached_lmk2graph[lmk]
                        else:
                            # remove indeterminant factors from gtsam
                            self.gaussian_solver.remove_factors(
                                list(range(self.get_gs_factor_size() - fg.size(), self.get_gs_factor_size()))
                            )
                elif isinstance(lmk, SE2Variable):
                    xstd, ystd = np.std(equal_samples[:,2], axis=0)
                    th_std = circstd(theta_to_pipi(equal_samples[:,2]), high=np.pi, low=-np.pi)
                    if xstd < reinit_trans_tol and ystd < reinit_trans_tol and th_std < reinit_orient_tol:
                        # indicate the belief of the landmark location is fairly concentrated
                        # it is ready to be initialized
                        factors = self.lmk2factors
                        fg = gtsam.NonlinearFactorGraph()
                        for i, f in enumerate(factors):
                            fg.add(f)
                        values = gtsam.Values()
                        lmk_key = to_Key(lmk)
                        values.insert(lmk_key, gtsam.Pose2(*max_sample))
                        update_flag = False
                        try:
                            self.gaussian_solver.update_factor_value(fg,
                                                                     values)
                            # manage add new_var and cached factors to the gaussian solver
                            update_flag = True
                        except Exception as e:
                            # the new factor still cannot initialize the new_var
                            print(str(e))
                        if update_flag:
                            reinit_lmk_list.append(lmk)
                            self.gaussian_vars.append(lmk)
                            self.lmk2isam_factor_indices[lmk] = \
                                list(range(self.get_gs_factor_size() - fg.size(),
                                           self.get_gs_factor_size()))
                            if lmk in self.uninit_lmk:
                                self.uninit_lmk.remove(lmk)
                            if lmk in self.cached_lmk2values:
                                # we expect the cached factor graph is the same as
                                # the factors involved in non-Gaussian inference
                                assert self.cached_lmk2graph[lmk].size() <= fg.size()
                                del self.cached_lmk2values[lmk]
                                del self.cached_lmk2graph[lmk]
                        else:
                            # remove indeterminant factors from gtsam
                            self.gaussian_solver.remove_factors(
                                list(range(self.get_gs_factor_size() - fg.size(), self.get_gs_factor_size()))
                            )
                else:
                    raise NotImplementedError("Not implemented variable types for initializing unint_lmk.")
        return reinit_lmk_list

    def update_lmk_belief(self, update_covariance_estimate=True):
        if update_covariance_estimate:
            self.gaussian_solver.update_covariance_estimate()
        self.lmk_update_fun()
        return

    def get_sum_mixture_proposal(self, lmk: Variable, rbt_var2sample: Dict[Variable, np.ndarray]):
        lmk_fs = self.lmk2factors[lmk]
        f_n = len(lmk_fs)
        ury_fs = []
        for f in lmk_fs:
            if f.var1 == lmk:
                rbt_var = f.var2
            else:
                rbt_var = f.var1
            if isinstance(f, SE2R2RangeGaussianLikelihoodFactor):
                assert isinstance(lmk, R2Variable)
                ury_fs.append(UnaryR2RangeGaussianPriorFactor(lmk, rbt_var2sample[rbt_var][:2],
                                                              f.observation[0], f.sigma))
            else:
                raise NotImplementedError
        return UnaryFactorMixture(lmk, np.ones(f_n)/f_n, ury_fs)

    def posterior_samples(self):
        """
        return an array of samples and a list of vars
        """
        return self.posterior_fun()

