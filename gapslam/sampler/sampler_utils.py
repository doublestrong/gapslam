from abc import ABCMeta
from typing import List, Dict

import numpy as np

from factors.Factors import Factor
from factors.utils import unpack_prior_binary_nh_da_factors
from slam.Variables import Variable


class JointFactor(Factor, metaclass=ABCMeta):
    def __init__(self, factors: List[Factor], vars: List[Variable]) -> None:
        """
        :param factors
        :type: List[Factor]
        :param vars: list of variables
        :type: List[Variable]
        """
        self._vars = vars
        self._factors = factors
        super().__init__()
        self._var_to_indices = {}
        current_index = 0
        for var in vars:
            next_index = current_index + var.dim
            self._var_to_indices[var] = list(range(current_index,
                                                   next_index))
            current_index = next_index
        self._factor_to_indices = {}
        for factor in self._factors:
            indices = []
            for var in factor.vars:
                indices += self._var_to_indices[var]
            self._factor_to_indices[factor] = indices
        self._x = None
        self._w = None
        self._is_gaussian = all([factor.is_gaussian for factor in factors])

    @property
    def is_gaussian(self) -> bool:
        return self._is_gaussian

    @property
    def is_unary(self) -> bool:
        return len(self._vars) == 1

    @property
    def is_binary(self) -> bool:
        return len(self._vars) == 2

    def set_quadrature(self, x, w):
        self._x = x
        self._w = w

    def quadrature(self, qtype, qparams, mass, *args, **kwargs):
        if qtype == 0:
            return self._x, self._w

    @property
    def vars(self) -> List[Variable]:
        return self._vars

    @property
    def var_indices(self) -> Dict[Variable, List[int]]:
        return self._var_to_indices

    @property
    def factor_to_indices(self) -> Dict[Factor, List[int]]:
        return self._factor_to_indices

    def pdf(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        :param x: the positions at which densities are evaluated
        :type: numpy.ndarray
               each row is a position
               the number of rows is the number of positions
        :param kwargs:
        :return: the densities at given locations
        :rtype: 1-dim np.ndarray
        """
        densities = np.ones(x.shape[0])
        for factor in self._factors:
            densities *= factor.pdf(x[:, self._factor_to_indices[factor]])
        return densities

    def log_pdf(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        :param x: the positions at which logarithm of densities are evaluated
        :type: numpy.ndarray
               each row is a position
               the number of rows is the number of positions
        :param kwargs:
        :return: the logarithm densities at given locations
        :rtype: 1-dim np.ndarray
        """
        densities = np.zeros(x.shape[0])
        for factor in self._factors:
            densities += factor.log_pdf(x[:, self._factor_to_indices[factor]])
        return densities

    def grad_x_log_pdf(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        :param x: the positions at which densities are evaluated
        :type: numpy.ndarray
               each row is a position
               the number of rows is the number of positions
        :param kwargs:
        :return: the gradients of log of densities at given locations
        :rtype: 2-dim np.ndarray
                each row is a position
                each column is a dimension
        """
        gradients = np.zeros(x.shape)
        for factor in self._factors:
            related_indices = self._factor_to_indices[factor]
            gradients[:, related_indices] += factor.grad_x_log_pdf(
                x[:, related_indices])
        return gradients

    def hess_x_log_pdf(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        :param x: the positions at which densities are evaluated
        :type: numpy.ndarray
               each row is a position
               the number of rows is the number of positions
        :param kwargs:
        :return: the Hessian of log of densities at given locations
        :rtype: 3-dim np.ndarray
                the first index is for samples
                the next two indices are for dimensions
        """
        hessians = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
        for factor in self._factors:
            related_indices = self._factor_to_indices[factor]
            hessians[np.ix_(range(x.shape[0]), related_indices,
                            related_indices)] += \
                factor.hess_x_log_pdf(x[:, related_indices])
        return hessians

class StructuredJointFactorForSLAM:
    def __init__(self, factors: List[Factor], variable_pattern: List[Variable], manually_partitioned_llh_factors: List[Factor] = None):
        # likelihood_factors are binary factors while da_factors connect with three or more variables
        self.manually_partitioned_llh_factors = manually_partitioned_llh_factors
        self.prior_factors, binary_factors, nh_factors, da_factors = \
            unpack_prior_binary_nh_da_factors(factors)

        self.vars = variable_pattern
        self.var_to_indices = {}
        current_index = 0
        for var in variable_pattern:
            next_index = current_index + var.dim
            self.var_to_indices[var] = list(range(current_index,
                                                  next_index))
            current_index = next_index
        self.dim = current_index

        # store prior and likelihood factors that are going to supply likelihood functions for nested sampling
        self.factors_with_all_ends_sampled = []

        self.factor_to_indices = {}
        sampled_vars = set()
        true_priors = []
        for factor in self.prior_factors:
            # prior vars
            if len(set.intersection(sampled_vars, factor.vars)) != 0:
                self.factors_with_all_ends_sampled.append(factor)
            else:
                true_priors.append(factor)
                sampled_vars.update(factor.vars)
            # circular dim list
            cur_var = 0
            factor_indices = []
            for var in factor.vars:
                cur_var = cur_var + var.dim
                factor_indices += self.var_to_indices[var]
            self.factor_to_indices[factor] = factor_indices
        self.prior_factors = true_priors
        # looking for the likelihood factors which can be absorbed into prior
        self.binary_factors_with_one_unsampled_end = []
        self.is_var1_sampled = {}
        added_nh_factors = False
        while binary_factors or nh_factors:
            if not added_nh_factors and len(binary_factors) == 0:
                binary_factors = nh_factors
                added_nh_factors = True

            factor = binary_factors.pop(0)

            var_intersection = set.intersection(set(factor.vars),
                                                sampled_vars)
            intersect_len = len(var_intersection)
            var1, var2 = factor.vars
            factor_indices = self.var_to_indices[var1] + self.var_to_indices[var2]
            if intersect_len == 1:
                self.binary_factors_with_one_unsampled_end.append(factor)
                if next(iter(var_intersection)) == var1:  # only var1 has been sampled
                    if var1.dim < var2.dim:
                        #prevent sampling a SE3 pose from a a R2 landmark
                        if len(binary_factors) == 0:
                            #the only remaining factor can't be sampled
                            raise ValueError("The only remaining factor in this clique requires sampling from landmark to pose")
                        binary_factors.append(factor)
                        continue
                    else:
                        self.is_var1_sampled[factor] = True
                        sampled_vars.add(var2)
                else:  # only var2 has been sampled
                    if var2.dim < var1.dim:
                        #prevent sampling a SE3 pose from a a R2 landmark
                        if len(binary_factors) == 0:
                            #the only remaining factor can't be sampled
                            raise ValueError("The only remaining factor in this clique requires sampling from landmark to pose")
                        binary_factors.append(factor)
                        continue
                    else:
                        self.is_var1_sampled[factor] = False
                        sampled_vars.add(var1)
                self.factor_to_indices[factor] = factor_indices
            elif intersect_len == 2:  # both vars have been sampled
                self.factors_with_all_ends_sampled.append(factor)
                self.factor_to_indices[factor] = factor_indices
            elif intersect_len == 0:  # both vars have not been sampled so push this factor back
                binary_factors.append(factor)
            else:
                raise ValueError("Oops! The number of variable"
                                 " intersection is " + str(intersect_len))
        assert len(sampled_vars) == len(variable_pattern)
        for factor in da_factors:
            da_vars = set(factor.vars)
            if da_vars.issubset(sampled_vars):
                factor_indices = []
                for var in factor.vars:
                    factor_indices += self.var_to_indices[var]
                self.factors_with_all_ends_sampled.append(factor)
                self.factor_to_indices[factor] = factor_indices
            else:
                unsampled_vars = da_vars - sampled_vars
                raise ValueError("Some variables of the data association have not been sampled: ".join([var.name for var in unsampled_vars]))
        if manually_partitioned_llh_factors is not None:
            for factor in manually_partitioned_llh_factors:
                f_vars = set(factor.vars)
                if f_vars.issubset(sampled_vars):
                    factor_indices = []
                    for var in factor.vars:
                        factor_indices += self.var_to_indices[var]
                    self.factors_with_all_ends_sampled.append(factor)
                    self.factor_to_indices[factor] = factor_indices
                else:
                    unsampled_vars = f_vars - sampled_vars
                    raise ValueError("Some variables of the likelihood factor have not been sampled: ".join([var.name for var in unsampled_vars]))

    def sample(self, num_sample: int):
        x = np.zeros((num_sample, self.dim))
        # sampling prior factors
        for factor in self.prior_factors:
            # u_to_sample() in GaussianPriorFactor, SE2PriorFactor or FlowsPriorFactor
            x[:, self.factor_to_indices[factor]] = \
                factor.sample(num_sample)
        # sampling priors from likelihood factors
        for factor in self.binary_factors_with_one_unsampled_end:
            var1_indices = self.factor_to_indices[factor][0:factor.var1.dim]
            var2_indices = self.factor_to_indices[factor][factor.var1.dim:]
            if self.is_var1_sampled[factor]:
                x[:, var2_indices] = factor.sample(var1=x[:, var1_indices], var2=None)
            else:
                x[:, var1_indices] = factor.sample(var1=None, var2=x[:, var2_indices])
        return x

    @property
    def ifDirectSampling(self):
        return len(self.factors_with_all_ends_sampled) == 0

    @property
    def circular_dim_list(self):
        circ_list = []
        for var in self.vars:
            circ_list = circ_list + var.circular_dim_list
        return  circ_list


class JointFactorForParticleFilter(StructuredJointFactorForSLAM):
    def __init__(self, factors: List[Factor], variable_pattern: List[Variable], manually_partitioned_llh_factors: List[Factor] = None):
        """
        This class prepares proposal samples and weight functions for particle filters.
        """

        super().__init__(factors=factors, variable_pattern=variable_pattern,
                         manually_partitioned_llh_factors=manually_partitioned_llh_factors)

    def naive_proposal(self, num_sample):
        """
        sampling from self.priors and self.binary_factors_with_one_unsampled_end
        :params num_sample: int
        :return: np.ndarray with shape (num_sample, dim)
        """
        x = np.zeros((num_sample, self.dim))
        # sampling prior factors
        for factor in self.prior_factors:
            # u_to_sample() in GaussiianPriorFactor, Pose2PriorFactor or FlowsPriorFactor
            x[:, self.factor_to_indices[factor]] = \
                factor.sample(num_sample)
        # sampling priors from likelihood factors
        for factor in self.binary_factors_with_one_unsampled_end:
            var1_indices = self.factor_to_indices[factor][:factor.var1.dim]
            var2_indices = self.factor_to_indices[factor][factor.var1.dim:]
            if self.is_var1_sampled[factor]:
                x[:, var2_indices] = factor.sample(var1=x[:, var1_indices])
            else:
                x[:, var1_indices] = factor.sample(var2=x[:, var2_indices])
        return x

    def loglike(self, x):
        """
        :params x: samples with shape (num_sample, dim)
        """
        log_like = np.zeros(x.shape[0])
        for factor in self.factors_with_all_ends_sampled:
            log_like += factor.log_pdf(x[:, self.factor_to_indices[factor]])
        return log_like

class JointFactorForNestedSampler(StructuredJointFactorForSLAM):
    def __init__(self, factors: List[Factor], variable_pattern: List[Variable], manually_partitioned_llh_factors: List[Factor] = None):
        super().__init__(factors=factors, variable_pattern=variable_pattern,
                         manually_partitioned_llh_factors=manually_partitioned_llh_factors)

    def ptform(self, u):
        """Transforms the uniform random variables `u ~ Unif[0., 1.)`
        to the parameters of interest.
        u is 1*dim numpy array.
        sampling from self.priors and self.binary_factors_with_one_unsampled_end
        """
        x = np.empty_like(u)
        # sampling prior factors
        for factor in self.prior_factors:
            # u_to_sample() in GaussiianPriorFactor, Pose2PriorFactor or FlowsPriorFactor
            x[self.factor_to_indices[factor]] = \
                factor.unif_to_sample(u[self.factor_to_indices[factor]])
        # sampling priors from likelihood factors
        for factor in self.binary_factors_with_one_unsampled_end:
            var1_indices = self.factor_to_indices[factor][:factor.var1.dim]
            var2_indices = self.factor_to_indices[factor][factor.var1.dim:]
            if self.is_var1_sampled[factor]:
                x[var2_indices] = factor.unif_to_sample(u=u[var2_indices],
                                                        var1=x[var1_indices])
            else:
                x[var1_indices] = factor.unif_to_sample(u=u[var1_indices],
                                                        var2=x[var2_indices])
        return x

    def loglike(self, x):
        """
        x is 1*dim; return a float
        """
        log_like = 0.0
        for factor in self.factors_with_all_ends_sampled:
            log_like += factor.evaluate_loglike(x[self.factor_to_indices[factor]])
        return log_like

    def grad_x_loglike(self, x):
        grad_x = np.zeros_like(x)
        for factor in self.factors_with_all_ends_sampled:
            idx = self.factor_to_indices[factor]
            grad_x[idx] += factor.grad_x_log_pdf(np.array([x[idx]]))[0]
        return grad_x

    def grad_u_loglike(self, x):
        return np.dot(self.grad_x_loglike(x), self.jac_u(x))

    def jac_u(self, x):
        jac = np.zeros((len(x), len(x)))
        for factor in self.prior_factors:
            f_idx = self.factor_to_indices[factor]
            ix_msh = np.ix_(f_idx, f_idx)
            jac[ix_msh] = factor.dvardu(x[f_idx])
        for factor in self.binary_factors_with_one_unsampled_end:
            var1_indices = self.factor_to_indices[factor][:factor.var1.dim]
            var2_indices = self.factor_to_indices[factor][factor.var1.dim:]
            if self.is_var1_sampled[factor]:
                dvar2dvar1, dvar2du = factor.dvar2du(var1=x[var1_indices], var2=x[var2_indices])
                var2msh = np.ix_(var2_indices, var2_indices)
                jac[var2msh] = dvar2du
                jac[var2_indices] += dvar2dvar1 @ jac[var1_indices]
            else:
                dvar1dvar2, dvar1du = factor.dvar1du(var1=x[var1_indices], var2=x[var2_indices])
                var1msh = np.ix_(var1_indices, var1_indices)
                jac[var1msh] = dvar1du
                jac[var1_indices] += dvar1dvar2 @ jac[var2_indices]
        return jac


# TODO: generalize it from 2D to 3D
class JointLikelihoodForNestedSampler(object):
    def __init__(self, factors: List[Factor],
                 variable_pattern: List[Variable], x_lim: list, y_lim: list):
        """
        Prior samples are drawn from uniform distributions within x_lim and y_lim. theta_lim are fixed to [-pi, pi]
        """
        self.vars = variable_pattern
        self.factors = factors
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.var_to_indices = {}
        current_index = 0
        for var in variable_pattern:
            next_index = current_index + var.dim
            self.var_to_indices[var] = list(range(current_index,
                                                  next_index))
            current_index = next_index
        self.dim = current_index

        self.factor_to_indices = {}
        for factor in factors:
            factor_indices = []
            for var in factor.vars:
                factor_indices += self.var_to_indices[var]
            self.factor_to_indices[factor] = factor_indices

        self.circular_dim_list = []
        for var in variable_pattern:
            self.circular_dim_list = self.circular_dim_list + var.circular_dim_list
        assert len(self.circular_dim_list) == self.dim

        dxdu = np.zeros(self.dim)
        for var in self.vars:
            indices = self.var_to_indices[var]
            dxdu[indices[0]] = self.x_lim[1] - self.x_lim[0]
            dxdu[indices[1]] = self.y_lim[1] - self.y_lim[0]
            if len(indices) == 3 and var.circular_dim_list[2]:
                dxdu[indices[2]] = 2 * np.pi
        self.dxdu = np.diag(dxdu)

    def ptform(self, u):
        """Transforms the uniform random variables `u ~ Unif[0., 1.)`
        to the parameters of interest.
        u is 1*dim numpy array.
        sampling from self.priors and self.binary_factors_with_one_unsampled_end
        """
        x = np.empty_like(u)
        # sampling prior factors
        for factor in self.factors:
            # u_to_sample() in GaussiianPriorFactor, Pose2PriorFactor or FlowsPriorFactor
            vars = factor.vars
            for var in vars:
                indices = self.var_to_indices[var]
                x[indices[0]] = self.x_lim[0] + u[indices[0]] * (self.x_lim[1]-self.x_lim[0])
                x[indices[1]] = self.y_lim[0] + u[indices[1]] * (self.y_lim[1] - self.y_lim[0])
                if len(indices) == 3 and var.circular_dim_list[2]:
                    x[indices[2]] = -np.pi + u[indices[2]] * 2 * np.pi
        return x

    def loglike(self, x):
        """
        x is 1*dim; return a float
        """
        log_like = 0.0
        for factor in self.factors:
            log_like += factor.evaluate_loglike(x[self.factor_to_indices[factor]])
        return log_like

    def grad_x_loglike(self, x):
        grad_x = np.zeros_like(x)
        for factor in self.factors:
            idx = self.factor_to_indices[factor]
            grad_x[idx] += factor.grad_x_log_pdf(np.array([x[idx]]))[0]
        return grad_x

    def grad_u_loglike(self, x):
        return np.dot(self.grad_x_loglike(x), self.dxdu)

    @property
    def ifDirectSampling(self):
        return False

# TODO: fill in methods for SMC sampling that exploits SLAM structure
class JointFactorForSMCSampler(StructuredJointFactorForSLAM):
    def __init__(self, factors: List[Factor], variable_pattern: List[Variable],  manually_partitioned_llh_factors: List[Factor] = None):
        super().__init__(factors=factors, variable_pattern=variable_pattern,
                         manually_partitioned_llh_factors=manually_partitioned_llh_factors)


