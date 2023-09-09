import logging
import random
import time
from typing import List

import numpy as np
import gtsam

from adaptive_inference.GaussianSolverWrapper import gtsamWrapper
from adaptive_inference.GTSAM_help import to_gtsam_factor, init_gtsam_var, boostrap_init_factors, reg_factor
from geometry.ThreeDimension import SE3Pose
from slam.Variables import R2Variable, SE2Variable, VariableType, Variable, R3Variable, SE3Variable
from factors.Factors import UnaryFactor, BinaryFactor, OdomFactor, Factor

default_ambiguous_factor_class = {'SE2R2RangeGaussianLikelihoodFactor',
                                  'SE2R2BearingLikelihoodFactor',
                                  'BinaryFactorWithNullHypo',
                                  'CameraProjectionFactor'}
default_bootstrap_init_class = set(boostrap_init_factors.keys())


class GAPSLAM:
    """
    A SLAM solver that combines both parameteric and non-parametric approaches to represent poseteriors.
    """

    def __init__(self, ambiguous_factor_class=None,
                 bootstrap_init_class=None,
                 prior_cov_scale=100,
                 lmk_sample_num=500,
                 rd_seed=0,
                 reinit_tol=0.1,
                 bw_method=0.1,
                 proposal_mixture=5,
                 remove_reg_factors=True,
                 parallel_n_jobs=1
                 ):
        """
        params:
            ambiguous_factor_class: multi-modal or underdetermined factor classes
            bootstrap_init_class: factor classes in ambiguous_factor_class that will be read into gtsam from the beginning
            lmk_factor_size: maximal size of landmark factors involved in updating non-Gaussian belief of a landmark
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

        if not self.ambiguous_factor_class.issubset(
                default_ambiguous_factor_class) or not self.bootstrap_init_class.issubset(default_bootstrap_init_class):
            set_diff1 = self.ambiguous_factor_class - default_ambiguous_factor_class
            set_diff2 = self.bootstrap_init_class - default_bootstrap_init_class
            raise NotImplementedError(
                f"Unknown ambiguous factors: {set_diff1}, unknown boostrap_init_factors: {set_diff2}")
        self.gaussian_solver = gtsamWrapper()
        # largest size of landmark factors for updating non-Gaussian belief

        # landmarks that still have non-Gaussian distributions
        self.nglmk = []

        # non-Gaussian landmarks whose marginals are almost unimodal gaussian
        # so we don't do re-initialization
        # unimodal landmarks
        self.unimlmk = []

        # all rbt poses related to a landmark
        self.nglmk2rbt = {}
        # all binary factors related to a landmark
        self.nglmk2bf = {}
        self.nglmk2gsbf = {}  # these binary factors in the gaussian solver
        # all unary factors to a landmark
        self.lmk2uf = {}
        self.lmk2gsuf = {}  # these unary factors in the gasussian solver

        # store lmk factor indices in isam if it is a variable in isam
        self.lmk2isam_factor_indices = {}
        # variables in isam
        self.gs_vars = []

        # regularization cov scale
        self.reg_scale = prior_cov_scale
        # factors for preventing indeterminant system errors in the Gaussian solver
        self.lmk_reg_prior = {}
        self.lmk2reg_factor_indices = {}

        # sampling params
        self.lmk_sample_num_per_path = lmk_sample_num
        self.rd_seed = rd_seed
        self.rng = np.random.default_rng(seed=self.rd_seed)  # this rng is newer and faster than the np.random
        self.reinit_tol = reinit_tol
        self.parallel_n_jobs = parallel_n_jobs

        self.all_factors = set()
        self.marginal_samples = {}
        self.bw_method = bw_method
        self.proposal_mixture_num = proposal_mixture
        self.lmk_samples = {}
        self.remove_reg_factors = remove_reg_factors

    def get_gs_factor_size(self):
        """
        Return the size of factor vector in the Gaussian solver.
        Note that some elements in the vector can be null.
        """
        return self.gaussian_solver.get_factor_size()

    def add_gs_factor(self, factor: Factor, new_var=None, init_val=None):
        """
        Add a factor to the Gaussian solver (GS)
        """
        fg = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        if new_var is None and init_val is None:
            gs_factor = to_gtsam_factor(factor)
            fg.add(gs_factor)
            self.gaussian_solver.update_factor_value(fg, values)
        else:
            new_key = new_var.key
            values.insert(new_key, init_val)
            gs_factor = to_gtsam_factor(factor)
            fg.add(gs_factor)
            self.gaussian_solver.update_factor_value(fg, values)
        return gs_factor

    def add_gs_prior(self, factor: UnaryFactor):
        """
        Add a prior to the Gaussian solver
        """
        # adding a prior
        if factor.vars[0] not in self.gs_vars:
            # initialize a new variable to gaussian solver
            init_val = init_gtsam_var(factor, self.gaussian_solver, factor.vars[0])
            gs_factor = self.add_gs_factor(factor, factor.vars[0], init_val)
            self.gs_vars.append(factor.vars[0])
        else:
            gs_factor = self.add_gs_factor(factor)
        return gs_factor

    def add_gs_binary_factor(self, factor: BinaryFactor):
        """
        Add a binary factor to the Gaussian solver
        :param factor:
        :return:
        """
        f_vars = factor.vars
        # can be optimized for speed
        new_vars = list(set(f_vars) - set(self.gs_vars))
        old_vars = list(set(f_vars).intersection(set(self.gs_vars)))

        # adding a binary factor
        if len(new_vars) == 0:
            gs_factor = self.add_gs_factor(factor)
        elif len(new_vars) == 1:
            init_val = init_gtsam_var(factor, self.gaussian_solver, old_vars[0])
            self.add_gs_factor(factor, new_vars[0], init_val)
            gs_factor = self.gs_vars.append(new_vars[0])
        else:
            raise ValueError("Adding a binary factor which is disconnected to the graph.")
        return gs_factor

    def add_odom_factor(self, factor: OdomFactor):
        """
        Add an odom factor to the GAPSLAM solver
        :param factor:
        :return:
        """
        assert isinstance(factor, OdomFactor)
        gs_factor = self.add_gs_binary_factor(factor)
        self.all_factors.add(factor)
        # since odom are supposed to be gaussian so followed no non-gaussian treatments
        return gs_factor

    def add_prior_factor(self, factor: UnaryFactor):
        """
        Add a prior factor to the GAPSLAM solver
        """
        assert isinstance(factor, UnaryFactor)
        gs_factor = self.add_gs_prior(factor)
        self.all_factors.add(factor)

        # only track landmark priors in case of non-Gaussian landmarks
        if factor.vars[0].type == VariableType.Landmark:
            # add to unary factor set
            if factor.vars[0] not in self.lmk2uf:
                self.lmk2uf[factor.vars[0]] = [factor]
                self.lmk2gsuf[factor.vars[0]] = [gs_factor]
            else:
                self.lmk2uf[factor.vars[0]].append(factor)
                self.lmk2gsuf[factor.vars[0]].append(gs_factor)
            # keep factor indices in isam
            if factor.vars[0] not in self.lmk2isam_factor_indices:
                self.lmk2isam_factor_indices[factor.vars[0]] = [self.get_gs_factor_size() - 1]
            else:
                self.lmk2isam_factor_indices[factor.vars[0]].append(self.get_gs_factor_size() - 1)
        return gs_factor

    def productFactorPDF(self, factor_list, sample_dict: dict, sample_num):
        densities = np.ones(sample_num)
        for factor in factor_list:
            densities *= factor.samples2pdf(rbt_samples=sample_dict[factor.var1], lmk_samples=sample_dict[factor.var2])
        return densities

    def productFactorLogPDF(self, factor_list, sample_dict: dict, sample_num):
        densities = np.zeros(sample_num)
        for factor in factor_list:
            densities += factor.samples2logpdf(rbt_samples=sample_dict[factor.var1],
                                               lmk_samples=sample_dict[factor.var2])
        return densities

        # turns out joblib parallel does not help here
        # if self.parallel_n_jobs == 1:
        #     densities = np.ones(sample_num)
        #     for factor in factor_list:
        #         densities *= factor.samples2pdf(rbt_samples=sample_dict[factor.var1], lmk_samples=sample_dict[factor.var2])
        # else:
        #     p_result = Parallel(n_jobs=self.parallel_n_jobs)(delayed(factor.samples2pdf)(sample_dict[factor.var1], sample_dict[factor.var2]) for factor in factor_list)
        #     densities = np.prod(np.array(p_result), axis=0)
        # return densities

    def factor2rbtlmk(self, factor: BinaryFactor):
        if factor.vars[0].type == VariableType.Pose:
            rbt_var = factor.vars[0]
            lmk_var = factor.vars[1]
        else:
            lmk_var = factor.vars[0]
            rbt_var = factor.vars[1]
        assert lmk_var.type == VariableType.Landmark and rbt_var.type == VariableType.Pose
        assert rbt_var in self.gs_vars
        return rbt_var, lmk_var

    def sample_lmk_given_rbt_paths(self, lmk_var: Variable, rbt_paths: dict, path_num: int, sample_per_path: int,
                                   rbt_dim):
        lmk_bfs = self.nglmk2bf[lmk_var]
        if len(lmk_bfs) > self.proposal_mixture_num:
            proposal_bfs = random.sample(lmk_bfs, self.proposal_mixture_num)
        else:
            proposal_bfs = lmk_bfs
        total_sample_num = sample_per_path * path_num

        if rbt_dim == 3:
            rbt_samples_for_proposal = {k: np.tile(v, (sample_per_path, 1)) for k, v in rbt_paths.items()}
            samples_for_pdf = {k: np.tile(v, (sample_per_path * len(proposal_bfs), 1)) for k, v in rbt_paths.items()}
            lmk_samples = np.zeros((sample_per_path * path_num * len(proposal_bfs), 2))
        elif rbt_dim == 6:
            rbt_samples_for_proposal = {k: np.tile(v, (sample_per_path, 1, 1)) for k, v in rbt_paths.items()}
            samples_for_pdf = {
                k: np.tile([SE3Pose(tmp_v).inverse().mat for tmp_v in v], (sample_per_path * len(proposal_bfs), 1, 1))
                for k, v
                in rbt_paths.items()}
            lmk_samples = np.zeros((sample_per_path * path_num * len(proposal_bfs), 3))
        else:
            raise NotImplementedError(f"Unknown robot variable dim {rbt_dim}.")

        # landmark samples from all binary factors
        for i, bf in enumerate(proposal_bfs):
            lmk_samples[i * total_sample_num: (i + 1) * total_sample_num] = bf.sample_lmk_from_rbt(
                rbt_samples_for_proposal[bf.var1])

        # evaluate the proposal density
        proposal_pdf = np.zeros(lmk_samples.shape[0])

        for bf in proposal_bfs:
            proposal_pdf += bf.samples2pdf(samples_for_pdf[bf.var1], lmk_samples)

        if lmk_var in self.lmk2uf:
            lmk_fs = lmk_bfs + self.lmk2uf[lmk_var]
        else:
            lmk_fs = lmk_bfs
        samples_for_pdf[lmk_var] = lmk_samples

        # target_pdf = self.productFactorPDF(lmk_fs, samples_for_pdf, lmk_samples.shape[0])
        # if np.sum(target_pdf) == 0.0:
        #     print(np.sum(proposal_pdf))
        # weights = target_pdf / proposal_pdf

        target_logpdf = self.productFactorLogPDF(lmk_fs, samples_for_pdf, lmk_samples.shape[0])
        logdiff = target_logpdf - np.log(proposal_pdf)

        logdiff -= np.median(logdiff)

        # for exp overflow
        tmp_max = logdiff.max()
        if tmp_max > 650:
            logdiff += 650 - tmp_max

        weights = np.exp(logdiff)

        weights = weights / np.sum(weights)

        if np.isnan(weights).any():
            print("Warning: using uniform weights in underflow")
            weights = np.ones(weights) / len(weights)

        # apply naive perturbation for now
        # TODOTODO
        if isinstance(lmk_var, (R2Variable, R3Variable)):
            # proposal_cov = np.cov(proposal, rowvar=False, aweights=weights)
            # lowtriangle = np.linalg.cholesky(proposal_cov)
            # equal_samples = systematic_resample(lmk_samples, weights)
            # # gaus_kde = scipy.stats.gaussian_kde(equal_samples.T, bw_method=self.bw_method)
            # # can specify seed for the resampling
            # # resamples = gaus_kde.resample(sample_num)
            # resamples = equal_samples[np.random.choice(np.arange(len(equal_samples)), total_sample_num,
            #                                            replace=False)] + np.random.random(
            #     (total_sample_num, equal_samples.shape[-1])) * self.bw_method
            # return resamples, samples_for_pdf

            # directly draw samples using the weight; no need to do the resampling
            samples_idx = self.rng.choice(np.arange(len(lmk_samples)), size=total_sample_num, replace=True, p=weights)
            equal_samples = lmk_samples[samples_idx].copy()
            equal_samples += self.rng.random(equal_samples.shape) * self.bw_method
            return equal_samples, samples_for_pdf
        else:
            raise NotImplementedError(f"Resampling for variable {lmk_var.__str__()}")

    def reinit_lmk_old(self, lmk_var: Variable, rbt_var: Variable, factor: BinaryFactor):
        """
        Reinitialize lmk_var in the Gaussian solver and add the factor to the Gaussian solver
        :param factor:
        :param lmk_var:
        :return:
        """
        # get connected robot variables and factors to the landmark
        rbt_vars = self.nglmk2rbt[lmk_var]
        lmk_key = lmk_var.key
        lmk_mean = self.gaussian_solver.get_point_estimate(lmk_key, lmk_var.__class__.__name__)
        rbt_path = {v: self.gaussian_solver.get_point_estimate(v.key, v.__class__.__name__) for v in rbt_vars}
        if isinstance(rbt_var, SE2Variable):
            rbt_path = {v: np.array([k.x(), k.y(), k.theta()]) for v, k in
                        rbt_path.items()}
            rbt_dim = 3
        elif isinstance(rbt_var, SE3Variable):
            rbt_path = {v: np.array([k.matrix()]) for v, k in
                        rbt_path.items()}
            rbt_dim = 6
        else:
            raise NotImplementedError(rbt_var.__str__())

        lmk_samples, _ = self.sample_lmk_given_rbt_paths(lmk_var, rbt_path, 1,
                                                         self.lmk_sample_num_per_path, rbt_dim)
        lmk_samples = np.vstack([lmk_samples, lmk_mean.reshape((1, -1))])
        if rbt_dim == 3:
            samples_for_pdf = {k: np.tile(v, (lmk_samples.shape[0], 1)) for k, v in rbt_path.items()}
        elif rbt_dim == 6:
            samples_for_pdf = {
                k: np.tile([SE3Pose(tmp_v).inverse().mat for tmp_v in v], (lmk_samples.shape[0], 1, 1)) for k, v
                in rbt_path.items()}
        else:
            raise NotImplementedError(f"Unknown robot variable dim {rbt_dim}.")
        samples_for_pdf[lmk_var] = lmk_samples

        if lmk_var in self.lmk2uf:
            lmk_factors = self.nglmk2bf[lmk_var] + self.lmk2uf[lmk_var]
        else:
            lmk_factors = self.nglmk2bf[lmk_var]

        target_pdf = self.productFactorPDF(lmk_factors + [factor], samples_for_pdf, lmk_samples.shape[0])

        max_idx = np.argmax(target_pdf)
        reinit_val = lmk_samples[max_idx]

        reinit_flag = False
        if np.linalg.norm(reinit_val - lmk_mean) > self.reinit_tol:
            reinit_flag = True

        fg = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        gs_factor = to_gtsam_factor(factor)
        fg.add(gs_factor)
        # update the Gaussian solver
        self.gaussian_solver.update_factor_value(fg, values)
        # bookkeeping
        self.lmk2isam_factor_indices[lmk_var].append(self.get_gs_factor_size() - 1)
        if reinit_flag:
            logging.info(f"Reinit {lmk_var.name}")
            updated_factor_indices = self.gaussian_solver.reinitialize(lmk_var.key, reinit_val,
                                                                       self.lmk2isam_factor_indices[lmk_var])
            if lmk_var in self.lmk2reg_factor_indices:
                # updating the index for artificial prior factors
                if self.lmk2reg_factor_indices[lmk_var] in self.lmk2isam_factor_indices[lmk_var]:
                    artif_factor_idx = self.lmk2isam_factor_indices[lmk_var].index(self.lmk2reg_factor_indices[lmk_var])
                    self.lmk2reg_factor_indices[lmk_var] = updated_factor_indices[artif_factor_idx]
            self.lmk2isam_factor_indices[lmk_var] = updated_factor_indices
        return gs_factor

    def reinit_lmk(self, lmk_var: Variable, rbt_var: Variable, factor: BinaryFactor):
        """
        Reinitialize lmk_var in the Gaussian solver and add the factor to the Gaussian solver
        :param factor:
        :param lmk_var:
        :return:
        """
        # get connected robot variables and factors to the landmark
        rbt_vars = self.nglmk2rbt[lmk_var]
        lmk_key = lmk_var.key
        lmk_mean = self.gaussian_solver.get_point_estimate(lmk_key, lmk_var.__class__.__name__)
        rbt_path = {v: self.gaussian_solver.get_point_estimate(v.key, v.__class__.__name__) for v in rbt_vars}
        if isinstance(rbt_var, SE2Variable):
            rbt_path = {v: np.array([k.x(), k.y(), k.theta()]) for v, k in
                        rbt_path.items()}
            rbt_dim = 3
        elif isinstance(rbt_var, SE3Variable):
            rbt_path = {v: np.array([k.matrix()]) for v, k in
                        rbt_path.items()}
            rbt_dim = 6
        else:
            raise NotImplementedError(rbt_var.__str__())

        if lmk_var in self.lmk_samples:
            lmk_samples = self.lmk_samples[lmk_var]
        else:
            lmk_samples, _ = self.sample_lmk_given_rbt_paths(lmk_var, rbt_path, 1,
                                                             self.lmk_sample_num_per_path, rbt_dim)
        lmk_samples = np.vstack([lmk_samples, lmk_mean.reshape((1, -1))])
        if rbt_dim == 3:
            samples_for_pdf = {k: np.tile(v, (lmk_samples.shape[0], 1)) for k, v in rbt_path.items()}
        elif rbt_dim == 6:
            samples_for_pdf = {
                k: np.tile([SE3Pose(tmp_v).inverse().mat for tmp_v in v], (lmk_samples.shape[0], 1, 1)) for k, v
                in rbt_path.items()}
        else:
            raise NotImplementedError(f"Unknown robot variable dim {rbt_dim}.")
        samples_for_pdf[lmk_var] = lmk_samples

        if lmk_var in self.lmk2uf:
            lmk_factors = self.nglmk2bf[lmk_var] + self.lmk2uf[lmk_var]
        else:
            lmk_factors = self.nglmk2bf[lmk_var]

        target_pdf = self.productFactorPDF(lmk_factors + [factor], samples_for_pdf, lmk_samples.shape[0])

        max_idx = np.argmax(target_pdf)
        reinit_val = lmk_samples[max_idx]

        reinit_flag = False
        if np.linalg.norm(reinit_val - lmk_mean) > self.reinit_tol:
            reinit_flag = True

        fg = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        gs_factor = to_gtsam_factor(factor)
        fg.add(gs_factor)
        # update the Gaussian solver
        self.gaussian_solver.update_factor_value(fg, values)
        # bookkeeping
        self.lmk2isam_factor_indices[lmk_var].append(self.get_gs_factor_size() - 1)
        if reinit_flag:
            logging.info(f"Reinit {lmk_var.name}")
            updated_factor_indices = self.gaussian_solver.reinitialize(lmk_var.key, reinit_val,
                                                                       self.lmk2isam_factor_indices[lmk_var])
            if lmk_var in self.lmk2reg_factor_indices:
                # updating the index for artificial prior factors
                if self.lmk2reg_factor_indices[lmk_var] in self.lmk2isam_factor_indices[lmk_var]:
                    artif_factor_idx = self.lmk2isam_factor_indices[lmk_var].index(self.lmk2reg_factor_indices[lmk_var])
                    self.lmk2reg_factor_indices[lmk_var] = updated_factor_indices[artif_factor_idx]
            self.lmk2isam_factor_indices[lmk_var] = updated_factor_indices
        return gs_factor

    def init_nglmk(self, factor: BinaryFactor, rbt_var: Variable, lmk_var: Variable):
        """
        Add the new non-Gaussian landmark as well as factors to the Gaussian solver
        :param factor:
        :param rbt_var:
        :param lmk_var:
        :return:
        """
        init_val = init_gtsam_var(factor, self.gaussian_solver, rbt_var)
        fg = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        new_key = lmk_var.key
        values.insert(new_key, init_val)
        gs_factor = to_gtsam_factor(factor)
        fg.add(gs_factor)
        # add an uninformative prior to the landmark for preventing indeterminant system errors
        self.lmk_reg_prior[lmk_var] = reg_factor(new_key, init_val, self.reg_scale, lmk_var)
        fg.add(self.lmk_reg_prior[lmk_var])
        logging.info("Added a factor for regularizing the underdetermined linear system. It will be removed later.")
        # update the Gaussian solver
        self.gaussian_solver.update_factor_value(fg, values)

        # bookkeeping
        self.lmk2isam_factor_indices[lmk_var] = [self.get_gs_factor_size() - 2,
                                                 self.get_gs_factor_size() - 1]
        self.lmk2reg_factor_indices[lmk_var] = self.get_gs_factor_size() - 1
        self.gs_vars.append(lmk_var)
        return gs_factor

    def add_lmk_meas_factor(self, factor: BinaryFactor, reinit_time: List = None, enable_reinit=True):
        """
        Add a rbt-lmk factor to the GAPSLAM solver
        :param factor:
        :return:
        """
        assert isinstance(factor, BinaryFactor)
        self.all_factors.add(factor)
        # start non-gaussian treatments
        f_class = factor.__class__.__name__
        if f_class not in self.ambiguous_factor_class:
            gs_factor = self.add_gs_binary_factor(factor)
        else:
            rbt_var, lmk_var = self.factor2rbtlmk(factor)
            # A binary factor needs non-Gaussian treatments
            if lmk_var not in self.nglmk:
                # brand-new landmarks
                self.nglmk.append(lmk_var)
                self.nglmk2bf[lmk_var] = [factor]
                self.nglmk2rbt[lmk_var] = [rbt_var]
                gs_factor = self.init_nglmk(factor, rbt_var, lmk_var)
                self.nglmk2gsbf[lmk_var] = [gs_factor]
                if not enable_reinit:
                    self.unimlmk.append(lmk_var)
            else:
                # old landmarkss
                if rbt_var not in self.nglmk2rbt[lmk_var]:
                    self.nglmk2rbt[lmk_var].append(rbt_var)
                if lmk_var not in self.unimlmk and enable_reinit:
                    tmp_start = time.time()
                    gs_factor = self.reinit_lmk(lmk_var, rbt_var, factor)
                    if reinit_time is not None:
                        reinit_time.append(time.time() - tmp_start)
                else:
                    gs_factor = self.add_gs_binary_factor(factor)
                    self.lmk2isam_factor_indices[lmk_var].append(self.get_gs_factor_size() - 1)
                self.nglmk2bf[lmk_var].append(factor)
                self.nglmk2gsbf[lmk_var].append(gs_factor)
        return gs_factor

    def update_gs_estimate(self):
        self.gaussian_solver.update_estimates()

    def sample_landmark(self, lmk_var, downsample=2000, sample_per_path=200, path_num: int = 200):
        rbt_vars = self.nglmk2rbt[lmk_var]
        if path_num > 1:
            rbt_joint_sampless = self.gaussian_solver.get_samples([v.key for v in rbt_vars],
                                                                  [v.__class__.__name__ for v in rbt_vars], path_num)
            cur_dim = 0
            rbt_path_samples = {}
            for v in rbt_vars:
                if isinstance(v, SE3Variable):
                    rbt_path_samples[v] = [SE3Pose.by_trans_rotvec(rbt_val).mat for rbt_val in
                                           rbt_joint_sampless[:, cur_dim:cur_dim + v.dim]]
                elif isinstance(v, SE2Variable):
                    rbt_path_samples[v] = rbt_joint_sampless[:, cur_dim:cur_dim + v.dim]
                else:
                    raise NotImplementedError(f"Unknown variable {v}")
                cur_dim += v.dim
        else:
            rbt_path_samples = {}
            for v in rbt_vars:
                k = self.gaussian_solver.get_point_estimate(v.key, v.__class__.__name__)
                if isinstance(v, SE2Variable):
                    rbt_path_samples[v] = np.array([[k.x(), k.y(), k.theta()]])
                elif isinstance(v, SE3Variable):
                    rbt_path_samples[v] = [k.matrix()]
                else:
                    raise NotImplementedError(v.__str__())
        lmk_samples, _ = self.sample_lmk_given_rbt_paths(lmk_var, rbt_path_samples, path_num, sample_per_path,
                                                         rbt_vars[0].dim)
        if len(lmk_samples) > downsample:
            lmk_samples = lmk_samples[:downsample]
        return lmk_samples

    def get_path_sample_dict(self, rbt_vars, path_num):
        # var odering in rbt_vars must be the same as ordering in gtsam
        var_idx = [self.gs_vars.index(v) for v in rbt_vars]
        res = all(i < j for i, j in zip(var_idx[:-1], var_idx[1:]))
        assert res
        if path_num > 1:
            rbt_path_samples = self.gaussian_solver.get_samples_dict(rbt_vars, path_num)
        else:
            rbt_path_samples = {}
            for v in rbt_vars:
                k = self.gaussian_solver.get_point_estimate(v.key, v.__class__.__name__)
                if isinstance(v, SE2Variable):
                    rbt_path_samples[v] = np.array([[k.x(), k.y(), k.theta()]])
                elif isinstance(v, SE3Variable):
                    rbt_path_samples[v] = [k.matrix()]
                else:
                    raise NotImplementedError(v.__str__())
        return rbt_path_samples

    def lmk_posterior_samples(self, sample_num=2000, sample_per_path=200, path_num: int = 200):
        lmk2samples = {}

        gaussian_lmks = [lmk for lmk in self.nglmk if lmk in self.unimlmk]
        ng_lmks = list(set(self.nglmk) - set(gaussian_lmks))

        if len(gaussian_lmks) > 0:
            g_lmk_samples_dict = self.gaussian_solver.get_samples_dict(gaussian_lmks, sample_num)
            lmk2samples = {**lmk2samples, **g_lmk_samples_dict}

        if len(ng_lmks) > 0:
            lmks = ng_lmks
            tmp_rbt_vars = [self.nglmk2rbt[tmp_lmk] for tmp_lmk in lmks]
            tmp_rbt_vars = list(set().union(*tmp_rbt_vars))
            rbt_vars = []
            for tmp_v in self.gs_vars:
                if tmp_v in tmp_rbt_vars:
                    rbt_vars.append(tmp_v)
            rbt_path_samples = self.get_path_sample_dict(rbt_vars, path_num)

            total_sample_num = sample_per_path * path_num
            rbt_dim = rbt_vars[0].dim

            for lmk_var in lmks:
                lmk_bfs = self.nglmk2bf[lmk_var]
                if len(lmk_bfs) > self.proposal_mixture_num:
                    proposal_bfs = random.sample(lmk_bfs, self.proposal_mixture_num)
                else:
                    proposal_bfs = lmk_bfs

                if rbt_dim == 3:
                    rbt_samples_for_proposal = {k: np.tile(rbt_path_samples[k], (sample_per_path, 1)) for k in
                                                self.nglmk2rbt[lmk_var]}
                    samples_for_pdf = {k: np.tile(rbt_path_samples[k], (sample_per_path * len(proposal_bfs), 1)) for k
                                       in
                                       self.nglmk2rbt[lmk_var]}
                    lmk_samples = np.zeros((sample_per_path * path_num * len(proposal_bfs), 2))
                elif rbt_dim == 6:
                    rbt_samples_for_proposal = {k: np.tile(rbt_path_samples[k], (sample_per_path, 1, 1)) for k in
                                                self.nglmk2rbt[lmk_var]}
                    samples_for_pdf = {
                        k: np.tile([SE3Pose(tmp_v).inverse().mat for tmp_v in rbt_path_samples[k]],
                                   (sample_per_path * len(proposal_bfs), 1, 1)) for k in self.nglmk2rbt[lmk_var]}
                    lmk_samples = np.zeros((sample_per_path * path_num * len(proposal_bfs), 3))
                else:
                    raise NotImplementedError(f"Unknown robot variable dim {rbt_dim}.")

                # landmark samples from all binary factors
                for i, bf in enumerate(proposal_bfs):
                    lmk_samples[i * total_sample_num: (i + 1) * total_sample_num] = bf.sample_lmk_from_rbt(
                        rbt_samples_for_proposal[bf.var1])

                # if self.parallel_n_jobs == 1:
                #     for i, bf in enumerate(proposal_bfs):
                #         lmk_samples[i * total_sample_num: (i + 1) * total_sample_num] = bf.sample_lmk_from_rbt(
                #             rbt_samples_for_proposal[bf.var1])
                # else:
                #     p_results = Parallel(n_jobs=self.parallel_n_jobs)(
                #         delayed(bf.sample_lmk_from_rbt)(rbt_samples_for_proposal[bf.var1]) for bf in proposal_bfs)
                #     lmk_samples = np.vstack(p_results)

                # evaluate the proposal density
                proposal_pdf = np.zeros(lmk_samples.shape[0])

                for bf in proposal_bfs:
                    proposal_pdf += bf.samples2pdf(samples_for_pdf[bf.var1], lmk_samples)

                if lmk_var in self.lmk2uf:
                    lmk_fs = lmk_bfs + self.lmk2uf[lmk_var]
                else:
                    lmk_fs = lmk_bfs
                samples_for_pdf[lmk_var] = lmk_samples
                target_pdf = self.productFactorPDF(lmk_fs, samples_for_pdf, lmk_samples.shape[0])

                weights = target_pdf / proposal_pdf
                weights = weights / np.sum(weights)

                # apply naive perturbation for now
                # TODOTODO
                if isinstance(lmk_var, (R2Variable, R3Variable)):
                    # proposal_cov = np.cov(proposal, rowvar=False, aweights=weights)
                    # lowtriangle = np.linalg.cholesky(proposal_cov)
                    # equal_samples = systematic_resample(lmk_samples, weights)
                    # # gaus_kde = scipy.stats.gaussian_kde(equal_samples.T, bw_method=self.bw_method)
                    # # can specify seed for the resampling
                    # # resamples = gaus_kde.resample(sample_num)
                    # if total_sample_num < downsample:
                    #     resamples = equal_samples[np.random.choice(np.arange(len(equal_samples)), total_sample_num,
                    #                                                replace=False)] + np.random.random(
                    #         (total_sample_num, equal_samples.shape[-1])) * self.bw_method
                    # else:
                    #     resamples = equal_samples[np.random.choice(np.arange(len(equal_samples)), downsample,
                    #                                                replace=False)] + np.random.random(
                    #         (downsample, equal_samples.shape[-1])) * self.bw_method
                    # lmk2samples[lmk_var] = resamples
                    samples_idx = self.rng.choice(np.arange(len(lmk_samples)), size=min(total_sample_num, sample_num),
                                                  replace=True,
                                                  p=weights)
                    equal_samples = lmk_samples[samples_idx].copy()
                    equal_samples += self.rng.random(equal_samples.shape) * self.bw_method
                    lmk2samples[lmk_var] = equal_samples
                else:
                    raise NotImplementedError(f"Resampling for variable {lmk_var.__str__()}")
        return lmk2samples

    def sample_selected_landmarks(self, lmks, downsample=2000, sample_per_path=200, path_num: int = 200, rbt_vars=None,
                                  rbt_path_samples=None, mixture_comp_num=None):
        lmk2samples = {}
        # drawing robot path samples
        if rbt_vars is None:
            tmp_rbt_vars = [self.nglmk2rbt[tmp_lmk] for tmp_lmk in lmks]
            tmp_rbt_vars = list(set().union(*tmp_rbt_vars))
            rbt_vars = []
            for tmp_v in self.gs_vars:
                if tmp_v in tmp_rbt_vars:
                    rbt_vars.append(tmp_v)
        if rbt_path_samples is None:
            rbt_path_samples = self.get_path_sample_dict(rbt_vars, path_num)

        total_sample_num = sample_per_path * path_num
        rbt_dim = rbt_vars[0].dim

        for lmk_var in lmks:
            lmk_bfs = self.nglmk2bf[lmk_var]
            if mixture_comp_num is None:
                comp_num = self.proposal_mixture_num
            else:
                comp_num = mixture_comp_num

            if len(lmk_bfs) > comp_num:
                proposal_bfs = random.sample(lmk_bfs, comp_num)
            else:
                proposal_bfs = lmk_bfs

            if rbt_dim == 3:
                rbt_samples_for_proposal = {k: np.tile(rbt_path_samples[k], (sample_per_path, 1)) for k in
                                            self.nglmk2rbt[lmk_var]}
                samples_for_pdf = {k: np.tile(rbt_path_samples[k], (sample_per_path * len(proposal_bfs), 1)) for k in
                                   self.nglmk2rbt[lmk_var]}
                lmk_samples = np.zeros((sample_per_path * path_num * len(proposal_bfs), 2))
            elif rbt_dim == 6:
                rbt_samples_for_proposal = {k: np.tile(rbt_path_samples[k], (sample_per_path, 1, 1)) for k in
                                            self.nglmk2rbt[lmk_var]}
                samples_for_pdf = {
                    k: np.tile([SE3Pose(tmp_v).inverse().mat for tmp_v in rbt_path_samples[k]],
                               (sample_per_path * len(proposal_bfs), 1, 1)) for k in self.nglmk2rbt[lmk_var]}
                lmk_samples = np.zeros((sample_per_path * path_num * len(proposal_bfs), 3))
            else:
                raise NotImplementedError(f"Unknown robot variable dim {rbt_dim}.")

            # landmark samples from all binary factors
            for i, bf in enumerate(proposal_bfs):
                lmk_samples[i * total_sample_num: (i + 1) * total_sample_num] = bf.sample_lmk_from_rbt(
                    rbt_samples_for_proposal[bf.var1])

            # if self.parallel_n_jobs == 1:
            #     for i, bf in enumerate(proposal_bfs):
            #         lmk_samples[i * total_sample_num: (i + 1) * total_sample_num] = bf.sample_lmk_from_rbt(
            #             rbt_samples_for_proposal[bf.var1])
            # else:
            #     p_results = Parallel(n_jobs=self.parallel_n_jobs)(
            #         delayed(bf.sample_lmk_from_rbt)(rbt_samples_for_proposal[bf.var1]) for bf in proposal_bfs)
            #     lmk_samples = np.vstack(p_results)

            # evaluate the proposal density
            proposal_pdf = np.zeros(lmk_samples.shape[0])

            for bf in proposal_bfs:
                proposal_pdf += bf.samples2pdf(samples_for_pdf[bf.var1], lmk_samples)

            if lmk_var in self.lmk2uf:
                lmk_fs = lmk_bfs + self.lmk2uf[lmk_var]
            else:
                lmk_fs = lmk_bfs
            samples_for_pdf[lmk_var] = lmk_samples
            target_pdf = self.productFactorPDF(lmk_fs, samples_for_pdf, lmk_samples.shape[0])

            weights = target_pdf / proposal_pdf
            weights = weights / np.sum(weights)

            # apply naive perturbation for now
            # TODOTODO
            if isinstance(lmk_var, (R2Variable, R3Variable)):
                # proposal_cov = np.cov(proposal, rowvar=False, aweights=weights)
                # lowtriangle = np.linalg.cholesky(proposal_cov)
                # equal_samples = systematic_resample(lmk_samples, weights)
                # # gaus_kde = scipy.stats.gaussian_kde(equal_samples.T, bw_method=self.bw_method)
                # # can specify seed for the resampling
                # # resamples = gaus_kde.resample(sample_num)
                # if total_sample_num < downsample:
                #     resamples = equal_samples[np.random.choice(np.arange(len(equal_samples)), total_sample_num,
                #                                                replace=False)] + np.random.random(
                #         (total_sample_num, equal_samples.shape[-1])) * self.bw_method
                # else:
                #     resamples = equal_samples[np.random.choice(np.arange(len(equal_samples)), downsample,
                #                                                replace=False)] + np.random.random(
                #         (downsample, equal_samples.shape[-1])) * self.bw_method
                # lmk2samples[lmk_var] = resamples
                samples_idx = self.rng.choice(np.arange(len(lmk_samples)), size=min(total_sample_num, downsample),
                                              replace=True,
                                              p=weights)
                equal_samples = lmk_samples[samples_idx].copy()
                equal_samples += self.rng.random(equal_samples.shape) * self.bw_method
                lmk2samples[lmk_var] = equal_samples
            else:
                raise NotImplementedError(f"Resampling for variable {lmk_var.__str__()}")
        return lmk2samples

    # def sample_selected_landmarks(self, lmks, downsample=2000, sample_per_path=200, path_num: int = 200):
    #     lmk2samples = {}
    #     # drawing robot path samples
    #     rbt_vars = [self.nglmk2rbt[tmp_lmk] for tmp_lmk in lmks]
    #     rbt_vars = list(set().union(*rbt_vars))
    #     if path_num > 1:
    #         rbt_joint_sampless = self.gaussian_solver.get_samples([v.key for v in rbt_vars],
    #                                                               [v.__class__.__name__ for v in rbt_vars], path_num)
    #         cur_dim = 0
    #         rbt_path_samples = {}
    #         for v in rbt_vars:
    #             if isinstance(v, SE3Variable):
    #                 rbt_path_samples[v] = [SE3Pose.by_trans_rotvec(rbt_val).mat for rbt_val in
    #                                        rbt_joint_sampless[:, cur_dim:cur_dim + v.dim]]
    #             elif isinstance(v, SE2Variable):
    #                 rbt_path_samples[v] = rbt_joint_sampless[:, cur_dim:cur_dim + v.dim]
    #             else:
    #                 raise NotImplementedError(f"Unknown variable {v}")
    #             cur_dim += v.dim
    #     else:
    #         rbt_path_samples = {}
    #         for v in rbt_vars:
    #             k = self.gaussian_solver.get_point_estimate(v.key, v.__class__.__name__)
    #             if isinstance(v, SE2Variable):
    #                 rbt_path_samples[v] = np.array([[k.x(), k.y(), k.theta()]])
    #             elif isinstance(v, SE3Variable):
    #                 rbt_path_samples[v] = [k.matrix()]
    #             else:
    #                 raise NotImplementedError(v.__str__())
    #
    #     total_sample_num = sample_per_path * path_num
    #     rbt_dim = rbt_vars[0].dim
    #
    #     for lmk_var in lmks:
    #         lmk_bfs = self.nglmk2bf[lmk_var]
    #         if len(lmk_bfs) > self.proposal_mixture_num:
    #             proposal_bfs = random.sample(lmk_bfs, self.proposal_mixture_num)
    #         else:
    #             proposal_bfs = lmk_bfs
    #
    #         if rbt_dim == 3:
    #             rbt_samples_for_proposal = {k: np.tile(rbt_path_samples[k], (sample_per_path, 1)) for k in self.nglmk2rbt[lmk_var]}
    #             samples_for_pdf = {k: np.tile(rbt_path_samples[k], (sample_per_path * len(proposal_bfs), 1)) for k in self.nglmk2rbt[lmk_var]}
    #             lmk_samples = np.zeros((sample_per_path * path_num * len(proposal_bfs), 2))
    #         elif rbt_dim == 6:
    #             rbt_samples_for_proposal = {k: np.tile(rbt_path_samples[k], (sample_per_path, 1, 1)) for k in self.nglmk2rbt[lmk_var]}
    #             samples_for_pdf = {
    #                 k: np.tile([SE3Pose(tmp_v).inverse().mat for tmp_v in rbt_path_samples[k]], (sample_per_path * len(proposal_bfs), 1, 1)) for k in self.nglmk2rbt[lmk_var]}
    #             lmk_samples = np.zeros((sample_per_path * path_num * len(proposal_bfs), 3))
    #         else:
    #             raise NotImplementedError(f"Unknown robot variable dim {rbt_dim}.")
    #
    #         # landmark samples from all binary factors
    #         for i, bf in enumerate(proposal_bfs):
    #             lmk_samples[i * total_sample_num: (i + 1) * total_sample_num] = bf.sample_lmk_from_rbt(
    #                 rbt_samples_for_proposal[bf.var1])
    #
    #         # evaluate the proposal density
    #         proposal_pdf = np.zeros(lmk_samples.shape[0])
    #
    #         for bf in proposal_bfs:
    #             proposal_pdf += bf.samples2pdf(samples_for_pdf[bf.var1], lmk_samples)
    #
    #         if lmk_var in self.lmk2uf:
    #             lmk_fs = lmk_bfs + self.lmk2uf[lmk_var]
    #         else:
    #             lmk_fs = lmk_bfs
    #         samples_for_pdf[lmk_var] = lmk_samples
    #         target_pdf = self.productFactorPDF(lmk_fs, samples_for_pdf, lmk_samples.shape[0])
    #
    #         weights = target_pdf / proposal_pdf
    #
    #         # apply naive perturbation for now
    #         # TODOTODO
    #         if isinstance(lmk_var, (R2Variable, R3Variable)):
    #             # proposal_cov = np.cov(proposal, rowvar=False, aweights=weights)
    #             # lowtriangle = np.linalg.cholesky(proposal_cov)
    #             equal_samples = systematic_resample(lmk_samples, weights)
    #             # gaus_kde = scipy.stats.gaussian_kde(equal_samples.T, bw_method=self.bw_method)
    #             # can specify seed for the resampling
    #             # resamples = gaus_kde.resample(sample_num)
    #             resamples = equal_samples[np.random.choice(np.arange(len(equal_samples)), total_sample_num,
    #                                                        replace=False)] + np.random.random(
    #                 (total_sample_num, equal_samples.shape[-1])) * self.bw_method
    #             if len(resamples) > downsample:
    #                 resamples = resamples[:downsample]
    #             lmk2samples[lmk_var] = resamples
    #         else:
    #             raise NotImplementedError(f"Resampling for variable {lmk_var.__str__()}")
    #     return lmk2samples

    def update_lmk_samples(self, lmk_var, downsample=1000, sample_per_path=200, path_num: int = 1, eig_threshold=0.5):
        if lmk_var not in self.unimlmk:
            rbt_vars = self.nglmk2rbt[lmk_var]
            if path_num > 1:
                rbt_joint_sampless = self.gaussian_solver.get_samples([v.key for v in rbt_vars],
                                                                      [v.__class__.__name__ for v in rbt_vars],
                                                                      path_num)
                cur_dim = 0
                rbt_path_samples = {}
                for v in rbt_vars:
                    if isinstance(v, SE3Variable):
                        rbt_path_samples[v] = [SE3Pose.by_trans_rotvec(rbt_val).mat for rbt_val in
                                               rbt_joint_sampless[:, cur_dim:cur_dim + v.dim]]
                    elif isinstance(v, SE2Variable):
                        rbt_path_samples[v] = rbt_joint_sampless[:, cur_dim:cur_dim + v.dim]
                    else:
                        raise NotImplementedError(f"Unknown variable {v}")
                    cur_dim += v.dim
            else:
                rbt_path_samples = {}
                for v in rbt_vars:
                    k = self.gaussian_solver.get_point_estimate(v.key, v.__class__.__name__)
                    if isinstance(v, SE2Variable):
                        rbt_path_samples[v] = np.array([[k.x(), k.y(), k.theta()]])
                    elif isinstance(v, SE3Variable):
                        rbt_path_samples[v] = [k.matrix()]
                    else:
                        raise NotImplementedError(v.__str__())
            lmk_samples, _ = self.sample_lmk_given_rbt_paths(lmk_var, rbt_path_samples, path_num, sample_per_path,
                                                             rbt_vars[0].dim)
            if len(lmk_samples) > downsample:
                self.lmk_samples[lmk_var] = lmk_samples[:downsample]
            else:
                self.lmk_samples[lmk_var] = lmk_samples
            ng_lmk_cov = np.cov(self.lmk_samples[lmk_var], rowvar=False)
            ng_lmk_eig, _ = np.linalg.eig(ng_lmk_cov)
            if np.sqrt(max(ng_lmk_eig)) < eig_threshold:
                logging.info(f"Found unimodal lmk {lmk_var.name}")
                self.unimlmk.append(lmk_var)
                if self.remove_reg_factors:
                    if lmk_var in self.lmk2reg_factor_indices:
                        logging.info("Removing the factor for regularization...")
                        self.gaussian_solver.remove_factors([self.lmk2reg_factor_indices[lmk_var]])
        return self.lmk_samples

    def get_ng_lmk_samples(self, downsample=1000, sample_per_path=200, path_num: int = 20, eig_threshold=0.5):
        ng_vars = [v for v in self.nglmk if v not in self.unimlmk]
        ng_vars = list(ng_vars)
        for lmk_var in ng_vars:
            rbt_vars = self.nglmk2rbt[lmk_var]
            rbt_joint_sampless = self.gaussian_solver.get_samples([v.key for v in rbt_vars],
                                                                  [v.__class__.__name__ for v in rbt_vars], path_num)
            cur_dim = 0
            rbt_path_samples = {}
            for v in rbt_vars:
                if isinstance(v, SE3Variable):
                    rbt_path_samples[v] = [SE3Pose.by_trans_rotvec(rbt_val).mat for rbt_val in
                                           rbt_joint_sampless[:, cur_dim:cur_dim + v.dim]]
                elif isinstance(v, SE2Variable):
                    rbt_path_samples[v] = rbt_joint_sampless[:, cur_dim:cur_dim + v.dim]
                else:
                    raise NotImplementedError(f"Unknown variable {v}")
                cur_dim += v.dim
            lmk_samples, _ = self.sample_lmk_given_rbt_paths(lmk_var, rbt_path_samples, path_num, sample_per_path,
                                                             rbt_vars[0].dim)
            if len(lmk_samples) > downsample:
                self.lmk_samples[lmk_var] = lmk_samples[:downsample]
            else:
                self.lmk_samples[lmk_var] = lmk_samples
            ng_lmk_cov = np.cov(self.lmk_samples[lmk_var], rowvar=False)
            ng_lmk_eig, _ = np.linalg.eig(ng_lmk_cov)
            if np.sqrt(max(ng_lmk_eig)) < eig_threshold:
                logging.info(f"Found unimodal lmk {lmk_var.name}")
                self.unimlmk.append(lmk_var)
                if self.remove_reg_factors:
                    if lmk_var in self.lmk2reg_factor_indices:
                        logging.info("Removing the factor for regularization...")
                        self.gaussian_solver.remove_factors([self.lmk2reg_factor_indices[lmk_var]])
        return self.lmk_samples

    def get_gs_marginals(self, get_cov=True):
        # TODO add SE3 R3 vars
        mean_list = []
        cov_list = []
        for v in self.gs_vars:
            mean = self.gaussian_solver.get_point_estimate(v.key, v.__class__.__name__)
            if isinstance(v, (R2Variable, R3Variable)):
                mean_list.append(np.array(mean))
            elif isinstance(v, SE2Variable):
                mean = [mean.x(), mean.y(), mean.theta()]
                mean_list.append(np.array(mean))
            elif isinstance(v, SE3Variable):
                mean_list.append(np.array(SE3Pose(mean.matrix()).array))
            if get_cov:
                cov = self.gaussian_solver.get_single_covariance(v.key)
                cov_list.append(np.array(cov))
        return mean_list, cov_list

    def vars2points(self, query_vars):
        point_estimates = []
        for tmp_i, lmk_var in enumerate(query_vars):
            point_estimates.append(
                self.gaussian_solver.get_point_estimate(var=lmk_var.key, vartype=lmk_var.__class__.__name__))
        return point_estimates
