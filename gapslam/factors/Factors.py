import math
from abc import ABCMeta
from copy import deepcopy

import gtsam
import numpy as np
import stats.Distributions as dists
import stats.Likelihoods as likes
import TransportMaps.Distributions as dist
import TransportMaps.Likelihoods as like
import TransportMaps.Maps as maps
from typing import List, Tuple, Union, Iterable, Dict
from scipy import stats as scistats
from scipy.spatial.transform import Rotation as sciR
import scipy
from geometry.ThreeDimension import SE3Pose
from slam.Variables import Variable, R2Variable, VariableType, R1Variable, SE2Variable, Bearing2DVariable, SE3Variable, \
    R3Variable, Bearing3DVariable
from geometry.TwoDimension import Point2, Rot2, SE2Pose
from utils.Units import _TWO_PI
from utils.Functions import sample_from_arr, theta_to_pipi


class Factor(metaclass=ABCMeta):
    @property
    def vars(self) -> List[Variable]:
        """
        All poses and landmarks connected to the factor
        """
        raise NotImplementedError

    @property
    def dim(self) -> int:
        """
        Dimensionality of measurement
        """
        return sum([var.dim for var in self.vars])

    @property
    def var_dim(self) -> List[Tuple[Variable, int]]:
        return [(var, var.dim) for var in self.vars]

    def __str__(self) -> str:
        raise NotImplementedError

    @classmethod
    def construct_from_text(cls, line: str, variables: Iterable[Variable]
                            ) -> "Factor":
        line = line.strip().split()
        if line[0] != "Factor":
            raise ValueError("The input string does not represent a factor")
        if line[1] == cls.__name__:
            raise ValueError(f"{cls.__name__} is an abstract class")
        factor = eval(line[1]).construct_from_text(line=" ".join(line[1:]),
                                                   variables=variables)
        return factor


class UnaryFactor(Factor, metaclass=ABCMeta):
    @property
    def var(self) -> Variable:
        raise NotImplementedError


class BinaryFactor(Factor, metaclass=ABCMeta):
    @property
    def vars(self) -> List[Variable]:
        raise NotImplementedError

    @property
    def var1(self) -> Variable:
        return self.vars[0]

    @property
    def var2(self) -> Variable:
        return self.vars[1]


class UndefinedFactor(Factor, metaclass=ABCMeta):
    def __init__(self, vars: List[Variable]) -> None:
        """
        :param vars: a list of variables
        :type: list of Variable objects
        """
        super().__init__()
        self._vars = vars

    @property
    def vars(self) -> List[Variable]:
        return self._vars

    def __str__(self) -> str:
        return "Factor " + self.__class__.__name__ + " " + \
               " ".join([var.name for var in self._vars])


class PriorFactor(Factor, metaclass=ABCMeta):
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: the positions at which densities are evaluated
        :type: numpy.ndarray
               each row is a position
               the number of rows is the number of positions
        :param kwargs:
        :return: the logarithm of densities at given locations
        :rtype: 1-dim np.ndarray
        """
        raise NotImplementedError

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: the positions at which logarithm of densities are evaluated
        :type: numpy.ndarray
               each row is a position
               the number of rows is the number of positions
        :param kwargs:
        :return: the logarithm of densities at given locations
        :rtype: 1-dim np.ndarray
        """
        raise NotImplementedError

    def grad_x_log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: the positions at which gradients are evaluated
        :type: numpy.ndarray
               each row is a position
               the number of rows is the number of positions
        :param kwargs:
        :return: the gradients of log of densities at given locations
        :rtype: 2-dim np.ndarray
                each row is a position
                each column is a dimension
        """
        raise NotImplementedError

    def sample(self, num_samples: int, **kwargs) -> np.ndarray:
        """
        :param num_samples: number of samples
        :type: int
        :return: samples
        :rtype: numpy.ndarray
               each row is a sample
               the number of columns is the number of dim
        """
        raise self.distribution.rvs(num_samples)

    def unif_to_sample(self, u: np.ndarray) -> np.ndarray:
        """
        Transform uniform distribution samples to samples from this prior factor
        :param u: single sample from uniform distribution
        :type: 1d numpy array
        :return: sample from the prior factor
        :rtype: 1d numpy array
        """
        raise NotImplementedError


class LikelihoodFactor(Factor, metaclass=ABCMeta):
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: the positions at which densities are evaluated
        :type: numpy.ndarray
               each row is a position
               the number of rows is the number of positions
        :param kwargs:
        :return: the densities at given locations
        :rtype: 1-dim np.ndarray
        """
        raise NotImplementedError

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: the positions at which logarithm of densities are evaluated
        :type: numpy.ndarray
               each row is a position
               the number of rows is the number of positions
        :param kwargs:
        :return: the logarithm of densities at given locations
        :rtype: 1-dim np.ndarray
        """
        raise NotImplementedError

    def grad_x_log_pdf(self, x: np.ndarray) -> np.ndarray:
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
        raise NotImplementedError

    @property
    def vars(self) -> List[Variable]:
        raise NotImplementedError

    def sample(self, index: Union[int, None], x: np.ndarray) -> np.ndarray:
        """
        Sample the index-th dimension, with all other dimensions given
        """
        raise NotImplementedError

    def unif_to_sample(self, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @property
    def observation(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def measurement_dim(self) -> int:
        return self.observation.shape[0]


class ExplicitPriorFactor(PriorFactor, metaclass=ABCMeta):
    def __init__(self, vars: List[Variable],
                 distribution: Union[dist.Distribution,
                                     dists.Distribution]) -> None:
        """
        :param vars: a list of variables
        :type: list of Variable objects
        :param distribution: the unnormalized density
        :type: TransportMaps.Distribution
        """
        super().__init__()
        self._vars = vars
        self._distribution = distribution
        # all dims are supposed to be on Euclidean space
        # self._circular_dim_list = [False] * self.dim

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._distribution.pdf(x)

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        return self._distribution.log_pdf(x)

    def grad_x_log_pdf(self, x: np.ndarray) -> np.ndarray:
        return self._distribution.grad_x_log_pdf(x)

    def sample(self, num_samples: int, **kwargs) -> np.ndarray:
        """
        :param num_samples: number of samples
        :type: int
        :return: samples
        :rtype: numpy.ndarray
               each row is a sample
               the number of columns is the number of dim
        """
        return self._distribution.rvs(num_samples)

    @property
    def vars(self) -> List[Variable]:
        return self._vars

    @property
    def distribution(self) -> Union[dist.Distribution, dists.Distribution]:
        return self._distribution

    @property
    def circular_dim_list(self) -> List[bool]:
        # list of True or False indicating circular dimension for variables in prior
        res = []
        for var in self.vars:
            res += var.circular_dim_list
        return res


class ExplicitLikelihoodFactor(LikelihoodFactor, metaclass=ABCMeta):
    def __init__(self, vars: List[Variable],
                 log_likelihood: Union[
                     like.LogLikelihood, likes.LogLikelihood, None]) -> None:
        """
        :param vars: a list of Variable objects
        :param log_likelihood: the likelihood
        :type: TransportMaps.LogLikelihood
        """
        super().__init__()
        self._vars = vars
        self._log_likelihood = log_likelihood

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: the positions at which densities are evaluated
        :type: numpy.ndarray
               each row is a position
               the number of rows is the number of positions
        :param kwargs:
        :return: the densities at given locations
        :rtype: 1-dim np.ndarray
        """
        return np.exp(self._log_likelihood.evaluate(x))

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: the positions at which logarithm of densities are evaluated
        :type: numpy.ndarray
               each row is a position
               the number of rows is the number of positions
        :param kwargs:
        :return: the logarithm of densities at given locations
        :rtype: 1-dim np.ndarray
        """
        return self._log_likelihood.evaluate(x)

    def grad_x_log_pdf(self, x: np.ndarray) -> np.ndarray:
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
        return self._log_likelihood.grad_x(x)

    @property
    def vars(self) -> List[Variable]:
        return self._vars

    @property
    def observation_var(self):
        raise NotImplementedError("Need to specify variable types (i.e., manifold) in likelihood factors.")


# consider using Deprecated decorator
class GaussianPriorFactor(ExplicitPriorFactor, metaclass=ABCMeta):
    def __init__(self,
                 var: Variable,
                 mean: np.ndarray,
                 covariance: np.ndarray = None,
                 precision: np.ndarray = None):
        super().__init__(
            vars=[var], distribution=dist.GaussianDistribution(
                mu=mean, sigma=covariance, precision=precision))
        self._mean = mean
        if covariance is not None:
            self._covariance = covariance
            self._precision = np.linalg.inv(covariance)
            self._cov_sqrt = scipy.linalg.sqrtm(covariance)
        elif precision is not None:
            self._covariance = np.linalg.inv(precision)
            self._precision = precision
            self._cov_sqrt = scipy.linalg.sqrtm(self._covariance)
        else:
            raise ValueError("None of cov and info. were defined.")
        self._lnorm = -0.5 * (np.log(_TWO_PI) * var.dim +
                              np.log(np.linalg.det(self._covariance)))  # ln(normalization)

    def unif_to_sample(self, u) -> np.array:
        normal_var = scistats.norm.ppf(u)  # convert to standard normal
        noise_sample = np.dot(self._cov_sqrt, normal_var) + self._mean
        return noise_sample

    def evaluate_loglike(self, x):
        delta = (x - self._mean.flatten())
        return -0.5 * np.dot(delta, np.dot(self._precision, delta)) + self._lnorm


class UnaryR2GaussianPriorFactor(ExplicitPriorFactor, UnaryFactor,
                                 metaclass=ABCMeta):
    measurement_variable_type = R2Variable

    def __init__(self, var: R2Variable, mu: np.ndarray,
                 covariance: np.ndarray = None,
                 precision: np.ndarray = None) -> None:
        """
        Params:
        sigma: Covariance matrix
        """
        self._distribution = dist.GaussianDistribution(
            mu=mu, sigma=covariance, precision=precision)
        super().__init__([var], distribution=self._distribution)
        if covariance is not None:
            self._covariance = covariance
            self._precision = np.linalg.inv(covariance)
            self._cov_sqrt = scipy.linalg.sqrtm(covariance)
        elif precision is not None:
            self._covariance = np.linalg.inv(precision)
            self._precision = precision
            self._cov_sqrt = scipy.linalg.sqrtm(self._covariance)
        self._lnorm = -0.5 * (np.log(_TWO_PI) * var.dim +
                              np.log(np.linalg.det(self._covariance)))  # ln(normalization)

    @property
    def vars(self) -> List[R2Variable]:
        return self._vars

    @property
    def mu(self) -> np.ndarray:
        return self._distribution.mu

    @property
    def covariance(self) -> np.ndarray:
        return self._distribution.sigma

    @property
    def precision(self) -> np.ndarray:
        return self._distribution.precision

    @property
    def observation(self):
        return self.mu

    def __str__(self) -> str:
        line = ["Factor", self.__class__.__name__, str(self.vars[0].name),
                str(self.mu[0]), str(self.mu[1]), "covariance",
                str(self.covariance[0, 0]), str(self.covariance[0, 1]),
                str(self.covariance[1, 0]), str(self.covariance[1, 1])]
        return " ".join(line)

    @classmethod
    def construct_from_text(cls, line: str, variables
                            ) -> "UnaryR2GaussianPriorFactor":
        line = line.strip().split()
        name_to_var = {var.name: var for var in variables}
        if line[0] == cls.__name__:
            var = name_to_var[line[1]]
            mu = np.array([float(line[2]), float(line[3])])
            key = line[4]
            if key == "covariance":
                key = "covariance"
            elif key != "precision":
                raise ValueError("Must specify either covariance or precision")
            mat = np.array([[float(line[5]), float(line[6])],
                            [float(line[7]), float(line[8])]])
            factor = UnaryR2GaussianPriorFactor(
                var=var, mu=mu, **{key: mat})
        else:
            raise ValueError("The factor name is incorrect")
        return factor

    def unif_to_sample(self, u) -> np.array:
        # u is a 1*2 numpy array
        # return a sample on R2
        normal_var = scistats.norm.ppf(u)  # convert to standard normal
        noise_sample = np.dot(self._cov_sqrt, normal_var) + self.mu
        return noise_sample

    def evaluate_loglike(self, x):
        delta = (x - self.observation.flatten())
        return -0.5 * np.dot(delta, np.dot(self._precision, delta)) + self._lnorm

    @property
    def is_gaussian(self):
        return True


class UnaryR2RangeGaussianPriorFactor(ExplicitPriorFactor, UnaryFactor,
                                      metaclass=ABCMeta):
    measurement_variable_type = R1Variable

    def __init__(self, var: R2Variable, center: np.ndarray,
                 mu: float,
                 sigma: float) -> None:
        """
        Params:
        sigma: float
        """
        self._distribution = dists.GaussianRangeDistribution(
            center=center, mu=mu, sigma=sigma ** 2)
        super().__init__([var], distribution=self._distribution)
        self._covariance = sigma ** 2
        self._precision = 1.0 / self._covariance
        self._cov_sqrt = sigma
        self._lnorm = -0.5 * (np.log(_TWO_PI) +
                              np.log(self._covariance))  # ln(normalization)

    @property
    def vars(self) -> List[R2Variable]:
        return self._vars

    @property
    def mu(self) -> float:
        return self._distribution.mean

    @property
    def covariance(self) -> float:
        return self._distribution.sigma

    @property
    def center(self) -> np.ndarray:
        return self._distribution.center

    def __str__(self) -> str:
        line = ["Factor", self.__class__.__name__, str(self.vars[0].name),
                "center", str(self.center[0]), str(self.center[1]),
                "mu", str(self.mu),
                "sigma", str(self._cov_sqrt)]
        return " ".join(line)

    @classmethod
    def construct_from_text(cls, line: str, variables
                            ) -> "UnaryR2RangeGaussianPriorFactor":
        line = line.strip().split()
        name_to_var = {var.name: var for var in variables}
        if line[0] == cls.__name__:
            var = name_to_var[line[1]]
            center = np.array([float(line[2]), float(line[3])])
            mu = float(line[4])
            sigma = float(line[5])
            factor = cls(
                var=var, center = center, mu=mu, sigma=sigma)
        else:
            raise ValueError("The factor name is incorrect")
        return factor

    def unif_to_sample(self, u) -> np.array:
        # u is a (2,))numpy array
        # return a sample on R2
        # for R2 this u.shape is (2,)
        assert len(u) == 2
        u_for_dist = u[0]
        u_for_angle = u[1]
        normal_var = scistats.norm.ppf(u_for_dist)  # convert to standard normal
        dist_sample = (self._cov_sqrt * normal_var + self.mu)
        angle_sample = (u_for_angle - 0.5) * _TWO_PI

        return self.center + np.array([dist_sample * np.cos(angle_sample),
                                       dist_sample * np.sin(angle_sample)])

    @property
    def observation(self):
        return self.mu

    def evaluate_loglike(self, x):
        delta = (np.linalg.norm(x - self.center.flatten() - self.observation))
        return -0.5 * np.dot(delta, np.dot(self._precision, delta)) + self._lnorm

    @property
    def is_gaussian(self):
        return False

class UncertainUnaryR2RangeGaussianPriorFactor(ExplicitPriorFactor, UnaryFactor,
                                      metaclass=ABCMeta):
    measurement_variable_type = R1Variable

    def __init__(self, var: R2Variable, center: np.ndarray,
                 mu: float,
                 sigma: float,
                 observed_flag: bool = False, unobserved_sigma: float = .3
                 ) -> None:
        """
        Params:
        sigma: float
        """
        self._center = center
        self._sigma = sigma
        self._observation = mu
        self._observed_flag = observed_flag
        self._unobserved_sigma = unobserved_sigma

        self._new_var = self._sigma **2 * self._unobserved_sigma **2 / (self._sigma **2 + self._unobserved_sigma **2)
        self._new_mu = self._unobserved_sigma **2 * self._observation / (self._sigma **2 + self._unobserved_sigma **2)
        self._new_cov_sqrt = np.sqrt(self._new_var)

        self._distribution = dists.GaussianRangeDistribution(
            center=center, mu=self._new_mu, sigma=self._new_var)

        super().__init__([var], distribution=self._distribution)

    @property
    def vars(self) -> List[R2Variable]:
        return self._vars

    @property
    def center(self) -> np.ndarray:
        return self._center

    def __str__(self) -> str:
        line = ["Factor", self.__class__.__name__, str(self.vars[0].name),
                "center", str(self.center[0]), str(self.center[1]),
                "mu", str(self._observation),
                "sigma", str(self._sigma),
                "observed_flag", str(int(self._observed_flag)),
                "unobserved_sigma",str(self._unobserved_sigma)]
        return " ".join(line)

    @classmethod
    def construct_from_text(cls, line: str, variables
                            ) -> "UncertainUnaryR2RangeGaussianPriorFactor":
        line = line.strip().split()
        name_to_var = {var.name: var for var in variables}
        if line[0] == cls.__name__:
            var = name_to_var[line[1]]
            center = np.array([float(line[3]), float(line[4])])
            mu = float(line[6])
            sigma = float(line[8])
            flag =  bool(int(line[10]))
            unobsv_sigma = float(line[12])
            factor = cls(
                var=var, center=center, mu=mu, sigma=sigma, observed_flag=flag, unobserved_sigma=unobsv_sigma)
        else:
            raise ValueError("The factor name is incorrect")
        return factor

    def unif_to_sample(self, u) -> np.array:
        # u is a (2,))numpy array
        # return a sample on R2
        # for R2 this u.shape is (2,)

        # we don't allow unobserved cases involved in drawing samples
        # This is a choice of design for this particular factor.

        assert self._observed_flag == True
        assert len(u) == 2
        u_for_dist = u[0]
        u_for_angle = u[1]
        normal_var = scistats.norm.ppf(u_for_dist)  # convert to standard normal
        dist_sample = (self._new_cov_sqrt * normal_var + self._new_mu)
        angle_sample = (u_for_angle - 0.5) * _TWO_PI

        return self.center + np.array([dist_sample * np.cos(angle_sample),
                                       dist_sample * np.sin(angle_sample)])

    def evaluate_loglike(self, x):
        if self._observed_flag == False:
            delta = np.linalg.norm(x - self.center.flatten())
            return np.log(1- np.exp(-0.5 * delta ** 2 / self._unobserved_sigma **2))
        else:
            delta = np.linalg.norm(x - self.center.flatten()) - self._new_mu
            return -0.5 * delta ** 2 / self._new_var

    @property
    def is_gaussian(self):
        return False

class UnarySE2ApproximateGaussianMixturePriorFactor(ExplicitPriorFactor, UnaryFactor):
    """ This class models a multimodal prior factor with manifold mixture noise.
        currently it is only implemented for draw samples and fit manifold mixture model
    """

    def __init__(self,
                 var: Variable,
                 prior_poses: List[SE2Pose],
                 weights: List[float],
                 covariances: List[np.ndarray]):
        # [np.zeros(var.dim) for i in weights]
        means = np.zeros((len(weights), var.dim))
        super().__init__([var], distribution=dists.GaussianMixtureDistribution(weights=weights,
                                                                               means=means,
                                                                               sigmas=covariances))
        dim = var.dim
        assert dim == 3 and dim == prior_poses[0].dim and dim == covariances[0].shape[0]
        self._translation_dim = 2

        self._dim = dim
        self._prior_poses = prior_poses

    @property
    def observation(self):
        return self._prior_poses

    @property
    def covariance(self) -> np.ndarray:
        return self._distribution.sigmas

    def sample(self, num_samples: int, **kwargs) -> np.ndarray:
        samples = np.zeros((num_samples, self._dim))
        comp2indices = {}
        for i in range(num_samples):
            comp = np.random.choice(
                range(self._distribution._num_components), p=self._distribution._weights)
            samples[[i], :] = (self._prior_poses[comp] *
                               SE2Pose.by_exp_map(self._distribution._components[comp].rvs(1).flatten())).array
            comp2indices[comp] = [i] if comp not in comp2indices else comp2indices[comp] + [i]
        return samples, comp2indices

    def __str__(self) -> str:
        line = ["Factor", self.__class__.__name__, str(self.vars[0].name)]
        line += [pose.__str__() for pose in self._prior_poses]
        line.append(np.array_str(np.array(self.covariance)))
        return " ".join(line)

    def is_gaussian(self):
        return False


# TODO: change the None distribution to WrapGaussianDist
class UnarySE2ApproximateGaussianPriorFactor(ExplicitPriorFactor, UnaryFactor):
    def __init__(self,
                 var: Variable,
                 prior_pose: SE2Pose,
                 covariance: np.ndarray,
                 correlated_R_t: bool = True):
        super().__init__([var], distribution=None)
        dim = var.dim
        assert dim == 3 and dim == prior_pose.dim and dim == covariance.shape[0]
        self._translation_dim = 2

        self._dim = dim
        self._observation = prior_pose.array
        self._noise_distribution = dist.GaussianDistribution(mu=np.zeros(dim),
                                                             sigma=covariance)
        self._prior_pose = prior_pose
        self._inv_prior_pose = prior_pose.inverse()
        self._covariance = covariance
        self._precision = np.linalg.inv(covariance)
        self._cov_sqrt = scipy.linalg.sqrtm(covariance)
        self._info_sqrt = scipy.linalg.sqrtm(self._precision)
        self._correlated_R_t = correlated_R_t
        self._est_rot_dispersion = 1.0 / covariance[self._translation_dim,
                                                    self._translation_dim]
        self._lnorm = -0.5 * (np.log(_TWO_PI) * var.dim +
                              np.log(np.linalg.det(self._covariance)))  # ln(normalization)

    @property
    def observation(self)-> np.ndarray:
        return self._observation

    @property
    def mu(self) -> np.ndarray:
        return self.observation

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance

    @property
    def precision(self) -> np.ndarray:
        return self._precision

    def sample(self, num_samples: int, **kwargs) -> np.ndarray:
        noise_samples = self._noise_distribution.rvs(num_samples)
        if self._correlated_R_t:
            var_samples = np.array(
                [(self._prior_pose * SE2Pose.by_exp_map(sample)).array
                 for sample in noise_samples])
        else:
            theta_array = np.random.vonmises(mu=0.0,
                                             kappa=self._est_rot_dispersion,
                                             size=num_samples)
            var_samples = np.empty_like(noise_samples)
            for i in range(num_samples):
                t = self._prior_pose.translation + \
                    Point2.by_array(noise_samples[i, 0:self._translation_dim])
                R = self._prior_pose.rotation * Rot2(theta=theta_array[i])
                var_samples[i, 0] = t.x
                var_samples[i, 1] = t.y
                var_samples[i, 2] = R.theta
        return var_samples

    def unif_to_sample(self, u) -> np.array:
        # u is a 1*D numpy array on se(2)
        # return a sample on SE(2)
        normal_var = scistats.norm.ppf(u)  # convert to standard normal
        noise_sample = np.dot(self._cov_sqrt, normal_var)
        if self._correlated_R_t:
            SE2_sample = (self._prior_pose * SE2Pose.by_exp_map(noise_sample)).array
        else:
            SE2_sample = np.empty_like(noise_sample)
            t = self._prior_pose.translation + \
                Point2.by_array(noise_sample[0:self._translation_dim])
            R = self._prior_pose.rotation * Rot2(theta=noise_sample[self._translation_dim])
            SE2_sample[0] = t.x
            SE2_sample[1] = t.y
            SE2_sample[2] = R.theta
        return SE2_sample

    def dvardu(self, var):
        Ti = SE2Pose.by_array(var)
        # perturbation pose
        Tn = self._inv_prior_pose * Ti
        dvidvn = np.eye(3)
        dvidvn[:2,:2] = self._prior_pose.rotation.matrix
        dvidlie = dvidvn @ Tn.grad_xi_expmap()

        # logmap
        pdf_arr = scistats.norm.pdf(self._info_sqrt @ Tn.log_map())
        # bing returned
        dvidu = dvidlie @ self._cov_sqrt @ np.diag(1.0 / pdf_arr)
        return dvidu

    # @property
    # def circular_dim_list(self) -> List[bool]:
    #     return self._circular_dim_list

    @classmethod
    def construct_from_text(cls, line: str, variables
                            ) -> "UnarySE2ApproximateGaussianPriorFactor":
        line = line.strip().split()
        name_to_var = {var.name: var for var in variables}
        if line[0] == cls.__name__:
            var = name_to_var[line[1]]
            x, y, theta = float(line[2]), float(line[3]), float(line[4])
            mat = np.array([[float(line[6]), float(line[7]), float(line[8])],
                            [float(line[9]), float(line[10]), float(line[11])],
                            [float(line[12]), float(line[13]), float(line[14])]])
            if line[5] == 'covariance':
                cov = mat
            elif line[5] == 'information':
                cov = np.linalg.inv(mat)
            else:
                raise ValueError("Either covariance or information should be specified")
            factor = UnarySE2ApproximateGaussianPriorFactor(var=var, prior_pose=SE2Pose(x, y, theta), covariance=cov)
        else:
            raise ValueError("The factor name is incorrect")
        return factor

    def __str__(self) -> str:
        line = ["Factor", self.__class__.__name__, str(self.vars[0].name),
                str(self.mu[0]), str(self.mu[1]), str(self.mu[2]), "covariance",
                str(self.covariance[0, 0]), str(self.covariance[0, 1]), str(self.covariance[0, 2]),
                str(self.covariance[1, 0]), str(self.covariance[1, 1]), str(self.covariance[1, 2]),
                str(self.covariance[2, 0]), str(self.covariance[2, 1]), str(self.covariance[2, 2])]
        return " ".join(line)

    def evaluate_loglike(self, x):
        return self.log_pdf(np.array([x]))[0]
        # T_i = SE2Pose.by_array(x)
        # delta = (self._inv_prior_pose * T_i).log_map()
        # return -0.5 * np.dot(delta, np.dot(self._precision, delta)) + self._lnorm

    def pdf(self, x: np.ndarray) -> np.ndarray:
        # careful: this is a PDF of x, y, theta so remember to multiply a Jacobian!
        dTs = [self._inv_prior_pose * SE2Pose.by_array(arr) for arr in x]
        logmap_arr = np.array([dT.log_map() for dT in dTs])
        det_jac_arr = abs(np.array([dT.det_grad_x_logmap() for dT in dTs]))
        return self._noise_distribution.pdf(logmap_arr) * det_jac_arr

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        dTs = [self._inv_prior_pose * SE2Pose.by_array(arr) for arr in x]
        logmap_arr = np.array([dT.log_map() for dT in dTs])
        det_jac_arr = abs(np.array([dT.det_grad_x_logmap() for dT in dTs]))
        return self._noise_distribution.log_pdf(logmap_arr) + np.log(det_jac_arr)

    def grad_x_log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        x: (-1,3) array
        """
        grad_T_log_pdf = np.zeros_like(x)
        for i in range(x.shape[0]):
            arr = x[i]
            dpose = self._inv_prior_pose * SE2Pose.by_array(arr)
            dpose_logmap = dpose.log_map()
            dlogmap_ddT = dpose.grad_x_logmap()
            ddT_dT = deepcopy(self._inv_prior_pose.matrix)
            ddT_dT[0, 2] = 0.0
            ddT_dT[1, 2] = 0.0
            grad_T_log_pdf[i:i + 1, :] = (self._noise_distribution.grad_x_log_pdf(dpose_logmap).reshape((1,-1)) @ dlogmap_ddT +
                                          np.array([dpose.grad_x_det_grad_x_logmap()])/dpose.det_grad_x_logmap())\
                                         @ ddT_dT
        return grad_T_log_pdf

    @property
    def is_gaussian(self):
        return True

class UnarySE3Factor(ExplicitPriorFactor, UnaryFactor):
    def __init__(self,
                 var: Variable,
                 prior_pose: SE3Pose,
                 covariance: np.ndarray):
        """
        :param var:
        :param prior_pose:
        :param covariance: elements ordering is Rx, Ry, Rz, tx, ty, tz
        """
        super().__init__([var], distribution=None)
        dim = var.dim
        assert dim == 6 and dim == prior_pose.dim and dim == covariance.shape[0]
        self._translation_dim = 3
        self._dim = dim
        self._observation = prior_pose.mat
        self._noise_distribution = dist.GaussianDistribution(mu=np.zeros(dim),
                                                             sigma=covariance)
        self._prior_pose = prior_pose
        self._inv_prior_pose = prior_pose.inverse()
        self._covariance = covariance
        self._precision = np.linalg.inv(covariance)
        self._cov_sqrt = scipy.linalg.sqrtm(covariance)
        self._info_sqrt = scipy.linalg.sqrtm(self._precision)
        self._lnorm = -0.5 * (np.log(_TWO_PI) * var.dim +
                              np.log(np.linalg.det(self._covariance)))  # ln(normalization)

    @property
    def observation(self)-> np.ndarray:
        return self._observation

    @property
    def mu(self) -> np.ndarray:
        return self.observation

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance

    @property
    def precision(self) -> np.ndarray:
        return self._precision

    def sample(self, num_samples: int, **kwargs) -> np.ndarray:
        noise_samples = self._noise_distribution.rvs(num_samples)
        var_samples = np.array(
            [(self._prior_pose * SE3Pose.by_exp_map(sample)).array
             for sample in noise_samples])
        return var_samples

    def unif_to_sample(self, u) -> np.array:
        # u is a 1*D numpy array on se(3)
        # return a sample on SE(3)
        normal_var = scistats.norm.ppf(u)  # convert to standard normal
        noise_sample = np.dot(self._cov_sqrt, normal_var)
        SE2_sample = (self._prior_pose * SE3Pose.by_exp_map(noise_sample)).array
        return SE2_sample

    @classmethod
    def construct_from_text(cls, line: str, variables
                            ) -> "UnarySE3Factor":
        line = line.strip().split()
        name_to_var = {var.name: var for var in variables}
        if line[0] == cls.__name__:
            var = name_to_var[line[1]]
            vec = [float(line[k]) for k in range(2, 9)]
            prior_pose = SE3Pose.by_transQuat(*vec)
            mat = np.array([float(line[k]) for k in range(10, 46)]).reshape((6,6))
            if line[9] == 'covariance':
                cov = mat
            elif line[9] == 'information':
                cov = np.linalg.inv(mat)
            else:
                raise ValueError("Either covariance or information should be specified")
            factor = UnarySE3Factor(var=var, prior_pose=prior_pose, covariance=cov)
        else:
            raise ValueError("The factor name is incorrect")
        return factor

    def __str__(self) -> str:
        line = ["Factor", self.__class__.__name__, str(self.vars[0].name)]
        line += [str(k) for k in  self._prior_pose.transQuat]
        line += ["covariance"]
        line += [str(k) for k in self.covariance.flatten() ]
        return " ".join(line)

    def evaluate_loglike(self, x):
        return self.log_pdf(np.array([x]))[0]
        # T_i = SE2Pose.by_array(x)
        # delta = (self._inv_prior_pose * T_i).log_map()
        # return -0.5 * np.dot(delta, np.dot(self._precision, delta)) + self._lnorm

    def pdf(self, x: np.ndarray) -> np.ndarray:
        # careful: this is a PDF of x, y, theta so remember to multiply a Jacobian!
        dTs = [self._inv_prior_pose * SE3Pose.by_trans_rotvec(arr) for arr in x]
        logmap_arr = np.array([dT.log_map() for dT in dTs])
        det_jac_arr = abs(np.array([dT.det_grad_x_logmap() for dT in dTs]))
        return self._noise_distribution.pdf(logmap_arr) * det_jac_arr

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        dTs = [self._inv_prior_pose * SE3Pose.by_trans_rotvec(arr) for arr in x]
        logmap_arr = np.array([dT.log_map() for dT in dTs])
        det_jac_arr = abs(np.array([dT.det_grad_x_logmap() for dT in dTs]))
        return self._noise_distribution.log_pdf(logmap_arr) + np.log(det_jac_arr)

class ImplicitPriorFactor(PriorFactor, metaclass=ABCMeta):
    pass


class InverseTransportFactor(ImplicitPriorFactor, metaclass=ABCMeta):
    def __init__(self, transport_map: maps.TransportMap,
                 vars: List[Variable]) -> None:
        super().__init__(transport_map.dim)
        self._vars = vars
        self._density = dist.PullBackTransportMapDistribution(
            base_distribution
            =dist.StandardNormalDistribution(transport_map.dim),
            transport_map=transport_map)

    @property
    def is_gaussian(self) -> bool:
        # TODO: Add the Gaussian judgement
        return False

    def pdf(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self._density.pdf(x)

    def log_pdf(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self._density.log_pdf(x)

    def grad_x_log_pdf(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self._density.grad_x_log_pdf(x)

    def hess_x_log_pdf(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        return self._density.hess_x_log_pdf(x)

    @property
    def vars(self) -> List[Variable]:
        return self._vars


# grad for pymc3 sampling


# define a theano Op for our likelihood function


# define a theano Op for our likelihood function


class R2LikelihoodFactor(LikelihoodFactor):
    pass


class RelativeLikelihoodFactor(LikelihoodFactor):
    pass


class BinaryLinearGaussianRelativeLikelihoodFactor(RelativeLikelihoodFactor):
    pass

class OdomFactor(BinaryFactor):
    pass

# TODO: replace unary_dim by variable's dim
class R2RelativeGaussianLikelihoodFactor(ExplicitLikelihoodFactor, BinaryFactor,
                                         BinaryLinearGaussianRelativeLikelihoodFactor, R2LikelihoodFactor):
    measurement_dim = 2
    measurement_type = R2Variable

    def __init__(self, var1: Variable, var2: Variable,
                 observation: np.ndarray,
                 covariance: np.ndarray = None,
                 precision: np.ndarray = None) -> None:
        """
        Construct a binary displacement Gaussian factor
            The measurement is the displacement from var1 to var2 plus a
            Gaussian additive noise
        :param var1
        :param var2
        :param observation: realization of the measurement
        :param covariance: sigma matrix of the noise
        :param precision: precision matrix of the noise
        """
        dim = var1.dim
        if dim != var2.dim:
            raise ValueError("The two variables must have the same "
                             "dimensionality")
        if len(observation) != dim:
            raise ValueError("The observation must have the same "
                             "dimensionality as the two variables")
        noise_distribution = dist.GaussianDistribution(mu=np.zeros(dim),
                                                       sigma=covariance,
                                                       precision=precision)
        minus_mat = np.zeros((dim, dim * 2))
        for d in range(dim):
            minus_mat[d, d] = -1.0
            minus_mat[d, dim + d] = 1.0
        log_likelihood = like.AdditiveLinearGaussianLogLikelihood(
            y=observation, c=np.zeros(dim), mu=np.zeros(dim),
            sigma=covariance, precision=precision, T=minus_mat)
        super().__init__(
            vars=[var1, var2], log_likelihood=log_likelihood)
        self._noise_distribution = noise_distribution
        self._unary_dim = dim
        self._observation = observation
        self._covariance = covariance
        # this factor is supposed to be on Euclidean space
        # self._circular_dim_list = [False] * dim
        if covariance is not None:
            self._covariance = covariance
            self._precision = np.linalg.inv(covariance)
            self._cov_sqrt = scipy.linalg.sqrtm(covariance)
        elif precision is not None:
            self._covariance = np.linalg.inv(precision)
            self._precision = precision
            self._cov_sqrt = scipy.linalg.sqrtm(self._covariance)
        else:
            raise ValueError("None of cov and info. were defined.")
        self._lnorm = -0.5 * (np.log(_TWO_PI) * self._unary_dim +
                              np.log(np.linalg.det(self._covariance)))  # ln(normalization)
        self._observation_var = type(self). \
            measurement_type(name="O" + var1.name + var2.name, variable_type=VariableType.Measurement)

    @classmethod
    def construct_from_text(cls, line: str, variables: Iterable[Variable]
                            ) -> "R2RelativeGaussianLikelihoodFactor":
        line = line.strip().split()
        name_to_var = {var.name: var for var in variables}
        if line[0] == cls.__name__:
            var1 = name_to_var[line[1]]
            var2 = name_to_var[line[2]]
            obs = np.array([float(line[3]), float(line[4])])
            key = line[5]
            mat = np.array([[float(line[6]), float(line[7])],
                            [float(line[8]), float(line[9])]])
            factor = R2RelativeGaussianLikelihoodFactor(
                var1=var1, var2=var2, observation=obs, **{key: mat})
        else:
            raise ValueError("The factor name is incorrect")
        return factor

    @property
    def observation_var(self):
        return self._observation_var

    @property
    def circular_dim_list(self):
        return self._observation_var.circular_dim_list

    @property
    def observation(self):
        return self._observation

    def sample(self, var1: np.ndarray = None, var2: np.ndarray = None
               ) -> np.ndarray:
        """
        Generate samples with given samples
            When var1 samples are var2 samples are given, generate observation
                samples
            When var2 samples are given, generate var1 samples
            When var1 samples are given, generate var2 samples
        :param var1
        :param var2
        :return: generated samples
        """
        if var1 is None:  # var2 samples are given, wants samples of var1
            if var2 is None:
                raise ValueError("Samples of at least one variable must be "
                                 "specified")
            if (len(var2.shape) != 2 or var2.shape[0] == 0 or
                    var2.shape[1] != self._unary_dim):
                raise ValueError("The dimensionality of variable 2 is wrong")
            num_samples = var2.shape[0]
            noise_samples = self._noise_distribution.rvs(num_samples)
            return var2 - noise_samples - self._observation
        elif var2 is None:  # var1 samples are given, wants samples of var2
            if (len(var1.shape) != 2 or var1.shape[0] == 0 or
                    var1.shape[1] != self._unary_dim):
                raise ValueError("The dimensionality of variable 1 is wrong")
            num_samples = var1.shape[0]
            noise_samples = self._noise_distribution.rvs(num_samples)
            return var1 + noise_samples + self._observation
        else:  # var1 and var2 samples are given, wants samples of observation
            if not (len(var1.shape) == len(var2.shape) == 2 and
                    var1.shape[0] == var2.shape[0] and
                    var1.shape[1] == var2.shape[1] == self._unary_dim):
                raise ValueError("Dimensionality of variable 1 or variable 2 is"
                                 " wrong")
            num_samples = var1.shape[0]
            return var2 - var1 + self._noise_distribution.rvs(num_samples)

    @property
    def vars(self) -> List[R2Variable]:
        return self._vars

    def unif_to_sample(self, u=np.ndarray, var1: np.ndarray = None, var2: np.ndarray = None
                       ) -> np.ndarray:
        """
        Generate samples with given samples
            u is a 1*D numpy array sampled from a uniform hypercube
            When var2 samples are given, generate var1 samples
            When var1 samples are given, generate var2 samples
        :param var1
        :param var2
        :return: generated samples
        """
        normal_var = scistats.norm.ppf(u)
        noise_sample = np.dot(self._cov_sqrt, normal_var)
        if var1 is None:  # var2 samples are given, wants samples of var1
            if var2 is None:
                raise ValueError("Samples of at least one variable must be "
                                 "specified")
            if (len(var2) != self._unary_dim):
                raise ValueError("The dimensionality of variable 2 is wrong")
            return var2 - noise_sample - self._observation
        elif var2 is None:  # var1 samples are given, wants samples of var2
            if (len(var1) != self._unary_dim):
                raise ValueError("The dimensionality of variable 1 is wrong")
            return var1 + noise_sample + self._observation
        else:
            raise ValueError("None of the vars are given.")

    def evaluate_loglike(self, x):
        var1 = x[0:self._unary_dim]
        var2 = x[self._unary_dim:]
        delta = (var2 - var1 - self._observation)
        return -0.5 * np.dot(delta, np.dot(self._precision, delta)) + self._lnorm

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance

    def __str__(self) -> str:
        line = ["Factor", self.__class__.__name__,
                str(self.vars[0].name), str(self.vars[1].name),
                str(self.observation[0]), str(self.observation[1]),
                "covariance",
                str(self.covariance[0, 0]), str(self.covariance[0, 1]),
                str(self.covariance[1, 0]), str(self.covariance[1, 1])]
        return " ".join(line)

    @property
    def is_gaussian(self):
        return True


# TODO: change the None loglikelihood to WrapGaussianDist
class SE2RelativeGaussianLikelihoodFactor(ExplicitLikelihoodFactor, OdomFactor):
    """
    Likelihood factor on SE(2)
    """
    measurement_dim = 3
    measurement_type = SE2Variable

    def __init__(self,
                 var1: Variable,
                 var2: Variable,
                 observation: Union[SE2Pose, np.ndarray],
                 covariance: np.ndarray = None,
                 correlated_R_t: bool = True
                 ) -> None:
        """
        :param var1
        :param var2
        :param observation: relative Pose2 mean from var1 to var2
        :param covariance: sigma matrix
            coordinate ordering must be x, y, theta
        :param correlated_R_t
        """
        if isinstance(observation, (np.ndarray, list, tuple)):
            observation = SE2Pose(*observation)
        dim = var1.dim
        if not dim == var2.dim == len(observation.array) == 3:
            raise ValueError("Dimensionality of poses, relative pose and "
                             "observation must be 3")
        self._translation_dim = 2

        log_likelihood = None
        super().__init__(vars=[var1, var2], log_likelihood=log_likelihood)

        self._noise_distribution = dist.GaussianDistribution(mu=np.zeros(dim),
                                                             sigma=covariance)
        self._unary_dim = dim
        self._observation = observation
        self._inv_pose = observation.inverse()
        self._correlated_Rt = correlated_R_t
        self._est_rot_dispersion = 1.0 / covariance[self._translation_dim,
                                                    self._translation_dim]
        self._pose_log_map = observation.log_map()
        self._covariance = covariance
        self._cov_sqrt = scipy.linalg.sqrtm(covariance)
        self._information = np.linalg.inv(covariance)
        self._info_sqrt = scipy.linalg.sqrtm(self._information)
        self._lnorm = -0.5 * (np.log(_TWO_PI) * self._unary_dim +
                              np.log(np.linalg.det(self._covariance)))  # ln(normalization)
        self._observation_var = type(self). \
            measurement_type(name="O" + var1.name + var2.name, variable_type=VariableType.Measurement)

    @property
    def observation_var(self):
        return self._observation_var

    @property
    def circular_dim_list(self):
        return self._observation_var.circular_dim_list

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance

    @property
    def observation(self) -> np.ndarray:
        return self._observation.array

    @property
    def noise_cov(self) -> np.ndarray:
        return self._covariance

    def __str__(self) -> str:
        line = ["Factor", self.__class__.__name__]
        line += [str(var.name) for var in self.vars]
        line += [str(self.observation[0]), str(self.observation[1]),
                 str(self.observation[2])]
        line += ["covariance", str(self.covariance[0, 0]), str(self.covariance[0, 1]), str(self.covariance[0, 2]),
                 str(self.covariance[1, 0]), str(self.covariance[1, 1]), str(self.covariance[1, 2]),
                 str(self.covariance[2, 0]), str(self.covariance[2, 1]), str(self.covariance[2, 2])]
        return " ".join(line)

    @classmethod
    def construct_from_text(cls, line: str,
                            variables: Iterable[Variable]) -> "SE2RelativeGaussianLikelihoodFactor":
        line = line.strip().split()
        name_to_var = {var.name: var for var in variables}
        if line[0] == cls.__name__:
            var1 = name_to_var[line[1]]
            var2 = name_to_var[line[2]]
            obs = SE2Pose(float(line[3]), float(line[4]), float(line[5]))
            key = line[6]
            mat = np.array([[float(line[7]), float(line[8]), float(line[9])],
                            [float(line[10]), float(line[11]), float(line[12])],
                            [float(line[13]), float(line[14]), float(line[15])]])
            factor = SE2RelativeGaussianLikelihoodFactor(
                var1=var1, var2=var2, observation=obs, **{key: mat})
        else:
            raise ValueError("The factor name is incorrect")
        return factor

    def sample(self,
               var1: Union[np.ndarray, SE2Pose, None] = None,
               var2: Union[np.ndarray, SE2Pose, None] = None
               ) -> np.ndarray:
        """
        Generate samples with given samples
            When var1 samples are var2 samples are given, generate observation
                samples
            When var2 samples are given, generate var1 samples
            When var1 samples are given, generate var2 samples
        :param var1: samples of var1
        :param var2: samples of var2
        :return: generated samples
        """
        if var1 is None:  # var2 samples are given, wants samples of var1
            if var2 is None:
                raise ValueError("Samples of at least one variable must be "
                                 "specified")
            if (len(var2.shape) != 2 or var2.shape[0] == 0 or
                    var2.shape[1] != self._unary_dim):
                raise ValueError("The dimensionality of variable 2 is wrong")
            num_samples = var2.shape[0]

            # Generate noise samples in SE(2) Lie algebra
            noise_samples = self._noise_distribution.rvs(num_samples)
            var1_samples = np.zeros_like(var2)
            if self._correlated_Rt:  # wrapped Gaussian distribution
                for i in range(num_samples):
                    T_j = SE2Pose.by_array(var2[i])
                    # T_ij_log_map = self._pose_log_map + noise_samples[i]
                    # T_ij_noised = SE2Pose.by_exp_map(T_ij_log_map)
                    T_ij_noised = self._observation * SE2Pose.by_exp_map(noise_samples[i, :])
                    T_i = T_j / T_ij_noised
                    var1_samples[i] = T_i.array
            else:  # Gaussian for translation, von Mises for rotation
                theta_array = np.random.vonmises(mu=0.0,
                                                 kappa=self._est_rot_dispersion,
                                                 size=num_samples)
                for i in range(num_samples):
                    T_j = SE2Pose.by_array(var2[i, :])
                    t_noise = Point2.by_array(noise_samples[i,
                                              0:self._translation_dim])
                    R_noise = Rot2(theta=theta_array[i])
                    R_i = T_j.rotation \
                          / R_noise \
                          / self._observation.rotation
                    t_i = T_j.translation - \
                          R_i * (self._observation.translation +
                                 t_noise)
                    var1_samples[i, 0] = t_i.x
                    var1_samples[i, 1] = t_i.y
                    var1_samples[i, 2] = R_i.theta
            return var1_samples
        elif var2 is None:  # var1 samples are given, wants samples of var2
            if (len(var1.shape) != 2 or var1.shape[0] == 0 or
                    var1.shape[1] != self._unary_dim):
                raise ValueError("The dimensionality of variable 1 is wrong")
            num_samples = var1.shape[0]
            noise_samples = self._noise_distribution.rvs(num_samples)
            var2_samples = np.zeros_like(var1)
            if self._correlated_Rt:  # wrapped Gaussian distribution
                for i in range(num_samples):
                    T_i = SE2Pose.by_array(var1[i])
                    # T_ij_log_map = self._pose_log_map + noise_samples[i]
                    # T_ij_noised = SE2Pose.by_exp_map(T_ij_log_map)
                    T_ij_noised = self._observation * SE2Pose.by_exp_map(noise_samples[i, :])
                    T_j = T_i * T_ij_noised
                    var2_samples[i] = T_j.array
            else:  # Gaussian for translation, von Mises for rotation
                theta_array = np.random.vonmises(mu=0.0,
                                                 kappa=self._est_rot_dispersion,
                                                 size=num_samples)
                for i in range(num_samples):
                    T_i = SE2Pose.by_array(var1[i])
                    t_noise = Point2.by_array(noise_samples[i,
                                              0:self._translation_dim])
                    R_noise = Rot2(theta=theta_array[i])
                    R_j = T_i.rotation \
                          * self._observation.rotation \
                          * R_noise
                    t_j = T_i.translation + \
                          T_i.rotation * (self._observation.translation +
                                          t_noise)
                    var2_samples[i, 0] = t_j.x
                    var2_samples[i, 1] = t_j.y
                    var2_samples[i, 2] = R_j.theta
            return var2_samples
        else:  # var1 and var2 samples are given, wants samples of observation
            if not (len(var1.shape) == len(var2.shape) == 2 and
                    var1.shape[0] == var2.shape[0] and
                    var1.shape[1] == var2.shape[1] == self._unary_dim):
                raise ValueError("Dimensionality of variable 1 or variable 2 is"
                                 " wrong")
            num_samples = var1.shape[0]
            noise_samples = self._noise_distribution.rvs(num_samples)
            obs_samples = np.zeros_like(var1)
            if self._correlated_Rt:  # wrapped Gaussian distribution
                for i in range(num_samples):
                    T_i = SE2Pose.by_array(var1[i])
                    T_j = SE2Pose.by_array(var2[i])
                    # T_ij_log_map = (T_i.inverse() * T_j).log_map() \
                    #                + noise_samples[i]
                    # T_ij_noised = SE2Pose.by_exp_map(T_ij_log_map)
                    T_ij_noised = (T_i.inverse() * T_j) * SE2Pose.by_exp_map(noise_samples[i, :])
                    obs_samples[i] = T_ij_noised.array
            else:  # Gaussian for translation, von Mises for rotation
                theta_array = np.random.vonmises(mu=0.0,
                                                 kappa=self._est_rot_dispersion,
                                                 size=num_samples)
                for i in range(num_samples):
                    T_i = SE2Pose.by_array(var1[i])
                    T_j = SE2Pose.by_array(var2[i])
                    T_ij = T_i.inverse() * T_j
                    t_noise = Point2.by_array(noise_samples[i,
                                              0:self._translation_dim])
                    R_noise = Rot2(theta=theta_array[i])
                    R_ij = T_ij.rotation * R_noise
                    t_ij = T_ij.translation + t_noise
                    obs_samples[i, 0] = t_ij.x
                    obs_samples[i, 1] = t_ij.y
                    obs_samples[i, 2] = R_ij.theta
            return obs_samples

    def unif_to_sample(self,
                       u,
                       var1: Union[np.ndarray, SE2Pose, None] = None,
                       var2: Union[np.ndarray, SE2Pose, None] = None
                       ) -> np.ndarray:
        """
        Generate samples with given samples
            u: noise sampled from a uniform cube, u is 1*D numpy array
            When var2 samples are given, generate var1 samples
            When var1 samples are given, generate var2 samples
        :param var1: a sample of var1
        :param var2: a sample of var2
        :return: generated samples
        """
        normal_var = scistats.norm.ppf(u)
        noise_sample = np.dot(self._cov_sqrt, normal_var)
        if var1 is None:  # var2 samples are given, wants samples of var1
            if var2 is None:
                raise ValueError("Samples of at least one variable must be "
                                 "specified")
            if (len(var2) != self._unary_dim):
                raise ValueError("The dimensionality of variable 2 is wrong")
            if self._correlated_Rt:  # wrapped Gaussian distribution
                T_j = SE2Pose.by_array(var2)
                # T_ij_log_map = self._pose_log_map + noise_sample
                # T_ij_noised = SE2Pose.by_exp_map(T_ij_log_map)
                T_ij_noised = self._observation * SE2Pose.by_exp_map(noise_sample)
                T_i = T_j / T_ij_noised
                var1_sample = T_i.array
            else:  # Gaussian for translation, von Mises for rotation
                theta_noise = noise_sample[self._translation_dim]
                T_j = SE2Pose.by_array(var2)
                t_noise = Point2.by_array(noise_sample[0:self._translation_dim])
                R_noise = Rot2(theta=theta_noise)
                R_i = T_j.rotation \
                      / R_noise \
                      / self._observation.rotation
                t_i = T_j.translation - \
                      R_i * (self._observation.translation +
                             t_noise)
                T_i = SE2Pose(x=t_i.x, y=t_i.y, theta=R_i.theta)
                var1_sample = T_i.array
            return var1_sample
        elif var2 is None:  # var1 samples are given, wants samples of var2
            if (len(var1) != self._unary_dim):
                raise ValueError("The dimensionality of variable 1 is wrong")
            if self._correlated_Rt:  # wrapped Gaussian distribution
                T_i = SE2Pose.by_array(var1)
                # T_ij_log_map = self._pose_log_map + noise_sample
                # T_ij_noised = SE2Pose.by_exp_map(T_ij_log_map)
                T_ij_noised = self._observation * SE2Pose.by_exp_map(noise_sample)
                T_j = T_i * T_ij_noised
                var2_sample = T_j.array
            else:  # Gaussian for translation, von Mises for rotation
                theta_noise = noise_sample[self._translation_dim]
                T_i = SE2Pose.by_array(var1)
                t_noise = Point2.by_array(noise_sample[0:self._translation_dim])
                R_noise = Rot2(theta=theta_noise)
                R_j = T_i.rotation \
                      * self._observation.rotation \
                      * R_noise
                t_j = T_i.translation + \
                      T_i.rotation * (self._observation.translation +
                                      t_noise)
                T_j = SE2Pose(x=t_j.x, y=t_j.y, theta=R_j.theta)
                var2_sample = T_j.array
            return var2_sample
        else:  # var1 and var2 samples are given, wants samples of observation
            raise ValueError("None of the vars are given.")

    def dvar2du(self, var1, var2):
        vj, vi = var2, var1
        Tj, Ti = SE2Pose.by_array(vj), SE2Pose.by_array(vi)
        # perturbation pose
        Tn = self._inv_pose * Ti.inverse() * Tj
        dvjdvn = np.eye(3)
        thj = Ti.theta + self._observation.theta
        dvjdvn[:2, :2] = Rot2(thj).matrix
        dvjdlie = dvjdvn @ Tn.grad_xi_expmap()

        # logmap
        pdf_arr = scistats.norm.pdf(self._info_sqrt @ Tn.log_map())
        # bing returned
        dvjdu = dvjdlie @ self._cov_sqrt @ np.diag(1.0 / pdf_arr)

        dvjdvi = np.eye(3)
        d_rot_thj = Rot2(thj).dmatdth
        d_rot_thi = Ti.rotation.dmatdth
        dvjdvi[:2,2] = d_rot_thj @ Tn.translation.array + d_rot_thi @ self._observation.translation.array  # sum product over last axis
        return dvjdvi, dvjdu

    def dvar1du(self, var1, var2):
        vj, vi = var2, var1
        Tj, Ti = SE2Pose.by_array(vj), SE2Pose.by_array(vi)
        # perturbation pose
        Tn = self._inv_pose * Ti.inverse() * Tj
        dvidvn = np.eye(3)
        thj_thn = Tj.theta - Tn.theta
        dvidvn[:2, :2] = Rot2(thj_thn)
        dvidvn = -dvidvn
        dvidvn[:2, 2] = Rot2(Tj.theta - Tn.theta - self._observation.theta).dmatdth \
                        @ self._observation.translation.array \
                        + Rot2(Tj.theta - Tn.theta).dmatdth @ Tn.translation.array
        dvidlie = dvidvn @ Tn.grad_xi_expmap()

        # logmap
        pdf_arr = scistats.norm.pdf(self._info_sqrt @ Tn.log_map())
        # bing returned
        dvidu = dvidlie @ self._cov_sqrt @ np.diag(1.0 / pdf_arr)

        dvidvj = np.eye(3)
        dvidvj[:2,2] = -dvidvn[:2, 2]  # sum product over last axis
        return dvidvj, dvidu

    def evaluate_loglike(self, x):
        return self.log_pdf(np.array([x]))[0]

    def pdf(self, x: np.ndarray) -> np.ndarray:
        dTs = [self._inv_pose * (SE2Pose.by_array(arr[:self._unary_dim]).inverse() *
                                SE2Pose.by_array(arr[self._unary_dim:])) for arr in x]
        logmap_arr = np.array([dT.log_map() for dT in dTs])
        det_jac_arr = abs(np.array([dT.det_grad_x_logmap() for dT in dTs]))
        return self._noise_distribution.pdf(logmap_arr) * det_jac_arr

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        dTs = [self._inv_pose * (SE2Pose.by_array(arr[:self._unary_dim]).inverse() *
                                SE2Pose.by_array(arr[self._unary_dim:])) for arr in x]
        logmap_arr = np.array([dT.log_map() for dT in dTs])
        det_jac_arr = abs(np.array([dT.det_grad_x_logmap() for dT in dTs]))
        return self._noise_distribution.log_pdf(logmap_arr) + np.log(det_jac_arr)

    def grad_x_log_pdf(self, x: np.ndarray) -> np.ndarray:
        grad_T_log_pdf = np.zeros_like(x)
        for i in range(x.shape[0]):
            pose_i = SE2Pose.by_array(x[i, :self._unary_dim])
            inv_pose_i = pose_i.inverse()
            pose_j = SE2Pose.by_array(x[i, self._unary_dim:])
            dpose = self._inv_pose * (inv_pose_i * pose_j)
            dpose_logmap = dpose.log_map()
            dlogmap_ddT = dpose.grad_x_logmap()
            ddT_dTij = deepcopy(self._inv_pose.matrix)
            ddT_dTij[0, 2] = 0.0
            ddT_dTij[1, 2] = 0.0
            dTij_dT = np.zeros((3, 6))
            dTij_dT[0:3, 0:3] = -(inv_pose_i.matrix)
            dTij_dT[0:3, 3:6] = inv_pose_i.matrix
            dTij_dT[0, 5] = 0.0
            dTij_dT[1, 5] = 0.0
            th_i = pose_i.theta
            dTij_dT[0:2, 2:3] = np.array([[-np.sin(th_i), np.cos(th_i)], [-np.cos(th_i), -np.sin(th_i)]]) @ \
                                (np.array(x[i, 3:5] - x[i, 0:2]).reshape(-1, 1))
            grad_T_log_pdf[i:i + 1, :] = (self._noise_distribution.grad_x_log_pdf(dpose_logmap) @ dlogmap_ddT +
                                          np.array([dpose.grad_x_det_grad_x_logmap()])/dpose.det_grad_x_logmap()) @ \
                                         ddT_dTij @ \
                                         dTij_dT
        return grad_T_log_pdf

    @property
    def is_gaussian(self):
        return True

class SE3RelativeGaussianLikelihoodFactor(ExplicitLikelihoodFactor, OdomFactor):
    """
    Likelihood factor on SE(3)
    """
    measurement_dim = 6
    measurement_type = SE3Variable

    def __init__(self,
                 var1: Variable,
                 var2: Variable,
                 observation: Union[SE3Pose, np.ndarray],
                 covariance: np.ndarray = None
                 ) -> None:
        """
        :param var1
        :param var2
        :param observation: relative Pose2 mean from var1 to var2
        :param covariance: elements ordering is Rx, Ry, Rz, tx, ty, tz
        :param correlated_R_t
        """
        if isinstance(observation, (np.ndarray, list, tuple)):
            observation = SE3Pose.by_transQuat(*observation)
        dim = var1.dim
        if not dim == var2.dim == len(observation.array) == 6:
            raise ValueError("Dimensionality of poses, relative pose and "
                             "observation must be 6")
        self._translation_dim = 3

        log_likelihood = None
        super().__init__(vars=[var1, var2], log_likelihood=log_likelihood)

        self._noise_distribution = dist.GaussianDistribution(mu=np.zeros(dim),
                                                             sigma=covariance)
        self._unary_dim = dim
        self._observation = observation
        self._inv_pose = observation.inverse()
        self._pose_log_map = observation.log_map()
        self._covariance = covariance
        self._cov_sqrt = scipy.linalg.sqrtm(covariance)
        self._information = np.linalg.inv(covariance)
        self._info_sqrt = scipy.linalg.sqrtm(self._information)
        self._lnorm = -0.5 * (np.log(_TWO_PI) * self._unary_dim +
                              np.log(np.linalg.det(self._covariance)))  # ln(normalization)
        self._observation_var = type(self). \
            measurement_type(name="O" + var1.name + var2.name, variable_type=VariableType.Measurement)

    @property
    def observation_var(self):
        return self._observation_var

    @property
    def circular_dim_list(self):
        return self._observation_var.circular_dim_list

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance

    @property
    def observation(self) -> np.ndarray:
        return self._observation.mat

    @property
    def noise_cov(self) -> np.ndarray:
        return self._covariance

    def __str__(self) -> str:
        line = ["Factor", self.__class__.__name__]
        line += [str(var.name) for var in self.vars]
        line += [str(k) for k in self._observation.transQuat]
        line += ["covariance"]
        line += [str(k) for k in self.covariance.flatten()]
        return " ".join(line)

    @classmethod
    def construct_from_text(cls, line: str,
                            variables: Iterable[Variable]) -> "SE3RelativeGaussianLikelihoodFactor":
        line = line.strip().split()
        name_to_var = {var.name: var for var in variables}
        if line[0] == cls.__name__:
            var1 = name_to_var[line[1]]
            var2 = name_to_var[line[2]]
            vec = [float(line[k]) for k in range(3, 10)]
            prior_pose = SE3Pose.by_transQuat(*vec)
            mat = np.array([float(line[k]) for k in range(11, 47)]).reshape((6,6))
            if line[10] == 'covariance':
                cov = mat
            elif line[10] == 'information':
                cov = np.linalg.inv(mat)
            else:
                raise ValueError("Either covariance or information should be specified")
            factor = SE3RelativeGaussianLikelihoodFactor(var1=var1, var2=var2, observation=prior_pose, covariance=cov)
        else:
            raise ValueError("The factor name is incorrect")
        return factor

    def sample(self,
               var1: Union[np.ndarray, SE3Pose, None] = None,
               var2: Union[np.ndarray, SE3Pose, None] = None
               ) -> np.ndarray:
        """
        Generate samples with given samples
            When var1 samples are var2 samples are given, generate observation
                samples
            When var2 samples are given, generate var1 samples
            When var1 samples are given, generate var2 samples
        :param var1: samples of var1
        :param var2: samples of var2
        :return: generated samples
        """
        if var1 is None:  # var2 samples are given, wants samples of var1
            if var2 is None:
                raise ValueError("Samples of at least one variable must be "
                                 "specified")
            if (len(var2.shape) != 2 or var2.shape[0] == 0 or
                    var2.shape[1] != self._unary_dim):
                raise ValueError("The dimensionality of variable 2 is wrong")
            num_samples = var2.shape[0]

            # Generate noise samples in SE(3) Lie algebra
            noise_samples = self._noise_distribution.rvs(num_samples)
            var1_samples = np.zeros_like(var2)
            for i in range(num_samples):
                T_j = SE3Pose.by_trans_rotvec(var2[i])
                T_ij_noised = self._observation * SE3Pose.by_exp_map(noise_samples[i, :])
                T_i = T_j / T_ij_noised
                var1_samples[i] = T_i.array
            return var1_samples
        elif var2 is None:  # var1 samples are given, wants samples of var2
            if (len(var1.shape) != 2 or var1.shape[0] == 0 or
                    var1.shape[1] != self._unary_dim):
                raise ValueError("The dimensionality of variable 1 is wrong")
            num_samples = var1.shape[0]
            noise_samples = self._noise_distribution.rvs(num_samples)
            var2_samples = np.zeros_like(var1)
            for i in range(num_samples):
                T_i = SE3Pose.by_trans_rotvec(var1[i])
                T_ij_noised = self._observation * SE3Pose.by_exp_map(noise_samples[i, :])
                T_j = T_i * T_ij_noised
                var2_samples[i] = T_j.array
            return var2_samples
        else:  # var1 and var2 samples are given, wants samples of observation
            if not (len(var1.shape) == len(var2.shape) == 2 and
                    var1.shape[0] == var2.shape[0] and
                    var1.shape[1] == var2.shape[1] == self._unary_dim):
                raise ValueError("Dimensionality of variable 1 or variable 2 is"
                                 " wrong")
            num_samples = var1.shape[0]
            noise_samples = self._noise_distribution.rvs(num_samples)
            obs_samples = np.zeros_like(var1)
            for i in range(num_samples):
                T_i = SE3Pose.by_trans_rotvec(var1[i])
                T_j = SE3Pose.by_trans_rotvec(var2[i])
                T_ij_noised = (T_i.inverse() * T_j) * SE3Pose.by_exp_map(noise_samples[i, :])
                obs_samples[i] = T_ij_noised.array
            return obs_samples

    def unif_to_sample(self,
                       u,
                       var1: Union[np.ndarray, SE3Pose, None] = None,
                       var2: Union[np.ndarray, SE3Pose, None] = None
                       ) -> np.ndarray:
        """
        Generate samples with given samples
            u: noise sampled from a uniform cube, u is 1*D numpy array
            When var2 samples are given, generate var1 samples
            When var1 samples are given, generate var2 samples
        :param var1: a sample of var1
        :param var2: a sample of var2
        :return: generated samples
        """
        normal_var = scistats.norm.ppf(u)
        noise_sample = np.dot(self._cov_sqrt, normal_var)
        if var1 is None:  # var2 samples are given, wants samples of var1
            if var2 is None:
                raise ValueError("Samples of at least one variable must be "
                                 "specified")
            if (len(var2) != self._unary_dim):
                raise ValueError("The dimensionality of variable 2 is wrong")
            T_j = SE3Pose.by_trans_rotvec(var2)
            # T_ij_log_map = self._pose_log_map + noise_sample
            # T_ij_noised = SE2Pose.by_exp_map(T_ij_log_map)
            T_ij_noised = self._observation * SE3Pose.by_exp_map(noise_sample)
            T_i = T_j / T_ij_noised
            var1_sample = T_i.array
            return var1_sample
        elif var2 is None:  # var1 samples are given, wants samples of var2
            if (len(var1) != self._unary_dim):
                raise ValueError("The dimensionality of variable 1 is wrong")
            T_i = SE3Pose.by_trans_rotvec(var1)
            # T_ij_log_map = self._pose_log_map + noise_sample
            # T_ij_noised = SE2Pose.by_exp_map(T_ij_log_map)
            T_ij_noised = self._observation * SE3Pose.by_exp_map(noise_sample)
            T_j = T_i * T_ij_noised
            var2_sample = T_j.array
            return var2_sample
        else:  # var1 and var2 samples are given, wants samples of observation
            raise ValueError("None of the vars are given.")

    def evaluate_loglike(self, x):
        return self.log_pdf(np.array([x]))[0]

    def pdf(self, x: np.ndarray) -> np.ndarray:
        dTs = [self._inv_pose * (SE3Pose.by_trans_rotvec(arr[:self._unary_dim]).inverse() *
                                SE3Pose.by_trans_rotvec(arr[self._unary_dim:])) for arr in x]
        logmap_arr = np.array([dT.log_map() for dT in dTs])
        det_jac_arr = abs(np.array([dT.det_grad_x_logmap() for dT in dTs]))
        return self._noise_distribution.pdf(logmap_arr) * det_jac_arr

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        dTs = [self._inv_pose * (SE3Pose.by_trans_rotvec(arr[:self._unary_dim]).inverse() *
                                SE3Pose.by_trans_rotvec(arr[self._unary_dim:])) for arr in x]
        logmap_arr = np.array([dT.log_map() for dT in dTs])
        det_jac_arr = abs(np.array([dT.det_grad_x_logmap() for dT in dTs]))
        return self._noise_distribution.log_pdf(logmap_arr) + np.log(det_jac_arr)

class CameraProjectionFactor(ExplicitLikelihoodFactor, BinaryFactor):
    """
    a general bearing factor for R2 and SE(2) variables
    """
    measurement_dim = 2
    measurement_type = Bearing3DVariable

    def __init__(self,
                 var1: Variable,
                 var2: Variable,
                 observation: np.ndarray,
                 covariance: np.ndarray,
                 cam_params: np.ndarray,
                 min_depth = .1,
                 max_depth = 1.0
                 ) -> None:
        """
        :param var1: a SE2 pose variable
        :param var2: a R2 point variable
        :param observation: x and y pixels in image
        :param covariance: covariance of x and y image coordinates
        :param cam_params: an array of fx fy cx cy
        :param min_depth: min depth from the camera
        :param max_depth: max depth from the camera
        """
        assert min_depth < max_depth
        assert isinstance(var1, SE3Variable)
        assert isinstance(var2, R3Variable)
        super().__init__(vars=[var1, var2], log_likelihood=None)
        self._observation = observation
        self._covariance = covariance
        self._noise_distribution = dist.GaussianDistribution(
            mu=np.zeros(2), sigma=covariance)

        # this is for evaluating log-likelihood
        self._observation_var = type(self). \
            measurement_type(name="O" + var1.name + var2.name, variable_type=VariableType.Measurement)

        self._cam_intrinsic_mat = np.array([[cam_params[0], 0, cam_params[2]],
                                            [0, cam_params[1], cam_params[3]],
                                            [0, 0, 1]])

        # visible ranges for the bearing factor
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._cam_params = cam_params

    @property
    def observation_var(self):
        return self._observation_var

    @property
    def cam_intrinsic_mat(self):
        return self._cam_intrinsic_mat

    @property
    def fx(self):
        return self._cam_params[0]

    @property
    def fy(self):
        return self._cam_params[1]

    @property
    def cx(self):
        return self._cam_params[2]

    @property
    def cy(self):
        return self._cam_params[3]

    @property
    def circular_dim_list(self):
        return self._observation_var.circular_dim_list

    @classmethod
    def construct_from_text(cls, line: str, variables: Iterable[Variable]
                            ) -> "CameraProjectionFactor":
        line = line.strip().split()
        name_to_var = {var.name: var for var in variables}
        if line[0] == cls.__name__:
            var1 = name_to_var[line[1]]
            var2 = name_to_var[line[2]]
            obs = [float(line[i]) for i in range(3, 5)]
            assert line[5] == "covariance"
            cov = np.array([float(line[i]) for i in range(6, 10)]).reshape((2,2))
            assert line[10] == "camparams"
            camparams = [float(line[i]) for i in range(11, 15)]
            if len(line) == 15:
                factor = cls(var1=var1, var2=var2, observation=obs, covariance=cov, cam_params=camparams)
            elif len(line) == 16:
                factor = cls(var1=var1, var2=var2, observation=obs, covariance=cov, cam_params=camparams, min_depth=float(line[15]))
            elif len(line) == 17:
                factor = cls(var1=var1, var2=var2, observation=obs, covariance=cov, cam_params=camparams, min_depth=float(line[15]),
                         max_depth=float(line[16]))
            else:
                raise ValueError("The number of arguments is incorrect")
        else:
            raise ValueError("The factor name is incorrect")
        return factor

    def __str__(self) -> str:
        line = ["Factor", self.__class__.__name__,
                str(self.vars[0].name), str(self.vars[1].name)]
        line += [str(k) for k in self.observation]
        line += ["covariance"]
        line += [str(k) for k in self._covariance.flatten()]
        line += ["camparams"]
        line += [str(self._cam_intrinsic_mat[0,0]), str(self._cam_intrinsic_mat[1,1]),
                 str(self._cam_intrinsic_mat[0,2]),str(self._cam_intrinsic_mat[1,2])]
        line += [str(self._min_depth), str(self._max_depth)]
        return " ".join(line)

    @property
    def observation(self) -> np.ndarray:
        return self._observation

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance

    def sample_var2_from_var1(self, var1_samples: np.ndarray) -> np.ndarray:
        if (len(var1_samples.shape) != 2 or var1_samples.shape[0] == 0 or
                var1_samples.shape[1] != self.var1.dim):
            raise ValueError("The dimensionality of variable 1 is wrong")
        """
        var1 is a SE3 variable
        Assume the camera frame with z pointing front, x pointing right, z pointing down
        Assume the elevation angle refers to the x-z plane with positive angles pointing up
        Assume the azimuth angle refers to the y-z plane with positive angles pointing right
        """
        num_samples = var1_samples.shape[0]
        pixel_samples = self._noise_distribution.rvs(num_samples) + self._observation
        depth_samples = np.random.uniform(self._min_depth, self._max_depth, num_samples)
        # homogeneous form
        var2_samples = np.array([SE3Pose.by_trans_rotvec(var1_s).depth2point(pixel_samples[i], depth_samples[i], self._cam_intrinsic_mat) for i, var1_s in enumerate(var1_samples)])
        return var2_samples

    def sample_var1_from_var2(self, var2_samples: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Haven't implemented projecting SE3 variables from R3 variables.")

    def sample_observations(self,
                            var1_samples: np.ndarray,
                            var2_samples: np.ndarray
                            ) -> np.ndarray:
        raise NotImplementedError("Haven't implemented projecting elevation and azimuth angles between SE3 variable and R3 R3 variables.")

    def sample(self,
               var1: Union[np.ndarray, None] = None,
               var2: Union[np.ndarray, None] = None
               ) -> np.ndarray:
        """
        Generate samples with given samples
            When var1 samples are var2 samples are given, generate observation
                samples
            When var2 samples are given, generate var1 samples
            When var1 samples are given, generate var2 samples
        :param var1: samples of var1
        :param var2: samples of var2
        :return: generated samples
        """
        if var1 is None:  # var2 samples are given, wants samples of var1
            if var2 is None:
                raise ValueError("Samples of at least one variable must be "
                                 "specified")
            return self.sample_var1_from_var2(var2)
        elif var2 is None:  # var1 samples are given, wants samples of var2
            return self.sample_var2_from_var1(var1)
        else:  # var1 and var2 samples are given, wants samples of observation
            return self.sample_observations(var1, var2)

    def unif_to_sample(self, u: np.ndarray,
                       var1: Union[np.ndarray, None] = None,
                       var2: Union[np.ndarray, None] = None
                       ) -> np.ndarray:
        """
        Nested sampling necessity
        u.shape is (2,) which is determined by JointFactorForNestedSampler
        the known sample has to be on SE2 currently
        """
        raise NotImplementedError

    def evaluate_loglike(self, x):
        """
        Nested sampling necessity
        x is (dim, ) where dim = var1.dim + var2.dim
        """
        return self.log_pdf(np.array([x]))[0]

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.log_pdf(x))

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        var1_sample = x[:, 0:self.var1.dim]
        var2_sample = x[:, self.var1.dim:]
        var2_pixels = np.array([SE3Pose.by_trans_rotvec(var1_s).point2pixel(var2_sample[i], self._cam_intrinsic_mat)
                                for i, var1_s in enumerate(var1_sample)])
        delta_pixels = var2_pixels - self._observation
        # log of p(r) is neglected as it is subject a uniform distribution
        return self._noise_distribution.log_pdf(delta_pixels)

    def grad_x_log_pdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def is_gaussian(self):
        return False

    def get_lmk_samples(self, rbt_value: gtsam.Pose3, num_samples: int):
        pixel_samples = self._noise_distribution.rvs(num_samples) + self._observation
        depth_samples = np.random.uniform(self._min_depth, self._max_depth, num_samples)
        rxry = (pixel_samples - self._cam_intrinsic_mat[0:2, 2]) / [self._cam_intrinsic_mat[0, 0], self._cam_intrinsic_mat[1, 1]]
        ptInCam = np.hstack((rxry, np.ones((num_samples, 1)))) * depth_samples.reshape((-1, 1))
        # homogeneous form
        ptInCam = np.hstack((ptInCam, np.ones((num_samples, 1))))
        return (rbt_value.matrix()@ptInCam.T).T[:, :3]

    def sample_lmk_from_rbt(self, rbt_samples: np.ndarray):
        """
        :param rbt_samples: n*4*4 array where n is the number of samples
        :return:
        """
        num_samples = rbt_samples.shape[0]
        pixel_samples = self._noise_distribution.rvs(num_samples) + self._observation
        depth_samples = np.random.uniform(self._min_depth, self._max_depth, num_samples)
        rxry = (pixel_samples - self._cam_intrinsic_mat[0:2, 2]) / [self._cam_intrinsic_mat[0, 0], self._cam_intrinsic_mat[1, 1]]
        ptInCam = np.hstack((rxry, np.ones((num_samples, 1)))) * depth_samples.reshape((-1, 1))
        # homogeneous form
        ptInCam = np.hstack((ptInCam, np.ones((num_samples, 1))))
        return np.einsum('ikj,ij->ik', rbt_samples, ptInCam)[:, :3]

    def samples2logpdf(self, rbt_samples, lmk_samples):
        """
        :param rbt_samples: n*4*4 array where n is the number of samples. Note that each transform matrix is the
        inverse of robot pose, i.e., world frame in robot frame
        :param lmk_samples: n*3 array
        :return:
        """
        # homogeneous form
        h_samples = np.hstack((lmk_samples, np.ones((lmk_samples.shape[0], 1))))
        ptInCam = np.einsum('ikj,ij->ki', rbt_samples, h_samples)[:3, :]
        ptInCam = ptInCam/ptInCam[-1]
        var2_pixels = (self._cam_intrinsic_mat @ ptInCam).T[:, :2]
        delta_pixels = var2_pixels - self._observation
        return self._noise_distribution.log_pdf(delta_pixels)

    def samples2pdf(self, rbt_samples, lmk_samples):
        """
        :param rbt_samples: n*4*4 array where n is the number of samples. Note that each transform matrix is the
        inverse of robot pose, i.e., world frame in robot frame
        :param lmk_samples: n*3 array
        :return:
        """
        return np.exp(self.samples2logpdf(rbt_samples, lmk_samples))

class SE2R2BearingLikelihoodFactor(ExplicitLikelihoodFactor, BinaryFactor):
    """
    a general bearing factor for R2 and SE(2) variables
    """
    measurement_dim = 1
    measurement_type = Bearing2DVariable

    def __init__(self,
                 var1: Variable,
                 var2: Variable,
                 observation: Union[np.ndarray, float],
                 sigma: float,
                 min_range = .1,
                 max_range = 1.0
                 ) -> None:
        """
        :param var1: a SE2 pose variable
        :param var2: a R2 point variable
        :param bearing: relative radians from var1 to var2
        :param sigma: standard deviation of Gaussian distribution
        :param min_range: min distance between var1 and var2
        :param max_range: max distance between var1 and var2
        """
        assert min_range < max_range
        assert isinstance(var1, SE2Variable)
        assert isinstance(var2, SE2Variable)
        super().__init__(vars=[var1, var2], log_likelihood=None)
        self._observation = observation if isinstance(observation, np.ndarray) \
            else np.array([observation])
        self._noise_distribution = dist.GaussianDistribution(
            mu=np.zeros(1), sigma=np.array([[sigma ** 2]]))
        self._sigma = sigma
        # this is for evaluating log-likelihood
        self._lnorm = -0.5 * np.log(_TWO_PI) - np.log(sigma)
        self._variance = sigma ** 2
        self._cov_sqrt = np.sqrt(self._variance)
        self._observation_var = type(self). \
            measurement_type(name="O" + var1.name + var2.name, variable_type=VariableType.Measurement)

        # visible ranges for the bearing factor
        self._min_range = min_range
        self._max_range = max_range

    @property
    def observation_var(self):
        return self._observation_var

    @property
    def circular_dim_list(self):
        return self._observation_var.circular_dim_list

    @classmethod
    def construct_from_text(cls, line: str, variables: Iterable[Variable]
                            ) -> "SE2R2BearingLikelihoodFactor":
        line = line.strip().split()
        name_to_var = {var.name: var for var in variables}
        if line[0] == cls.__name__:
            var1 = name_to_var[line[1]]
            var2 = name_to_var[line[2]]
            obs = float(line[3])
            sigma = float(line[4])
            if len(line) == 5:
                factor = cls(var1=var1, var2=var2, observation=obs, sigma=sigma)
            elif len(line) == 6:
                factor = cls(var1=var1, var2=var2, observation=obs, sigma=sigma, min_range=float(line[5]))
            elif len(line) == 7:
                factor = cls(var1=var1, var2=var2, observation=obs, sigma=sigma, min_range=float(line[5]),
                         max_range=float(line[6]))
            else:
                raise ValueError("The number of arguments is incorrect")
        else:
            raise ValueError("The factor name is incorrect")
        return factor

    def __str__(self) -> str:
        line = ["Factor", self.__class__.__name__,
                str(self.vars[0].name), str(self.vars[1].name),
                str(self.observation[0]), str(self.sigma), str(self._min_range), str(self._max_range)]
        return " ".join(line)

    @property
    def observation(self) -> np.ndarray:
        return self._observation

    def sample_var2_from_var1(self, var1_samples: np.ndarray) -> np.ndarray:
        if (len(var1_samples.shape) != 2 or var1_samples.shape[0] == 0 or
                var1_samples.shape[1] != self.var1.dim):
            raise ValueError("The dimensionality of variable 1 is wrong")
        if self.var1.dim == 3:
            """
            var1 is a SE2 variable
            """
            num_samples = var1_samples.shape[0]
            angle_samples = var1_samples[:, self.var1.R_dim_indices] + self._noise_distribution.rvs(num_samples) + self._observation
            dist_samples = np.random.uniform(self._min_range, self._max_range, (num_samples, 1))
            var2_samples = deepcopy(var1_samples)[:, self.var1.t_dim_indices]
            var2_samples += np.hstack((dist_samples * np.cos(angle_samples),
                              dist_samples * np.sin(angle_samples)))
        else:
            """
            var1 is a R2 variable
            """
            num_samples = var1_samples.shape[0]
            trans_angles = np.random.uniform(.0, _TWO_PI, (num_samples, 1))
            rot_samples = trans_angles + self._noise_distribution.rvs(num_samples) - self._observation
            dist_samples = np.random.uniform(self._min_range, self._max_range, (num_samples, 1))
            var2_samples = np.zeros((num_samples, self.var2.dim))
            var2_samples[:, self.var2.t_dim_indices] = var1_samples + np.hstack((dist_samples * np.cos(trans_angles),
                              dist_samples * np.sin(trans_angles)))
            var2_samples[:, self.var2.R_dim_indices] = rot_samples
        return var2_samples

    def sample_var1_from_var2(self, var2_samples: np.ndarray) -> np.ndarray:
        if (len(var2_samples.shape) != 2 or var2_samples.shape[0] == 0 or
                var2_samples.shape[1] != self.var2.dim):
            raise ValueError("The dimensionality of variable 2 is wrong")
        if self.var2.dim == 3:
            """
            var2 is a SE2 variable
            """
            num_samples = var2_samples.shape[0]
            angle_samples = var2_samples[:, self.var2.R_dim_indices] + self._noise_distribution.rvs(num_samples) + self._observation
            dist_samples = np.random.uniform(self._min_range, self._max_range, (num_samples, 1))
            var1_samples = deepcopy(var2_samples)[:, self.var2.t_dim_indices]
            var1_samples += np.hstack((dist_samples * np.cos(angle_samples),
                              dist_samples * np.sin(angle_samples)))
        else:
            """
            var2 is a R2 variable
            """
            num_samples = var2_samples.shape[0]
            trans_angles = np.random.uniform(.0, _TWO_PI, (num_samples, 1))
            rot_samples = trans_angles + self._noise_distribution.rvs(num_samples) - self._observation
            dist_samples = np.random.uniform(self._min_range, self._max_range, (num_samples, 1))
            var1_samples = np.zeros((num_samples, self.var1.dim))
            var1_samples[:, self.var1.t_dim_indices] = var2_samples + np.hstack((dist_samples * np.cos(trans_angles),
                              dist_samples * np.sin(trans_angles)))
            var1_samples[:, self.var1.R_dim_indices] = theta_to_pipi(rot_samples)
        return var1_samples

    def sample_observations(self,
                            var1_samples: np.ndarray,
                            var2_samples: np.ndarray
                            ) -> np.ndarray:
        if not (len(var1_samples.shape) == len(var2_samples.shape) == 2 and
                var1_samples.shape[0] == var2_samples.shape[0] and
                (var1_samples.shape[1] == self.var1.dim) and (var2_samples.shape[1] ==
                                                              self.var2.dim)):
            raise ValueError("Dimensionality of variable 1 or variable 2 is"
                             " wrong")
        num_samples = var1_samples.shape[0]
        noise_samples = self._noise_distribution.rvs(num_samples)
        if self.var1.dim == 3:
            res = theta_to_pipi(np.arctan2(var2_samples[1] - var1_samples[1], var2_samples[0] - var1_samples[0]) - var1_samples[:,-1] + noise_samples)
        else:
            res = theta_to_pipi(np.arctan2(var1_samples[1] - var2_samples[1], var1_samples[0] - var2_samples[0]) - var2_samples[:,-1] + noise_samples)
        return res

    @property
    def sigma(self) -> float:
        return self._sigma

    def sample(self,
               var1: Union[np.ndarray, None] = None,
               var2: Union[np.ndarray, None] = None
               ) -> np.ndarray:
        """
        Generate samples with given samples
            When var1 samples are var2 samples are given, generate observation
                samples
            When var2 samples are given, generate var1 samples
            When var1 samples are given, generate var2 samples
        :param var1: samples of var1
        :param var2: samples of var2
        :return: generated samples
        """
        if var1 is None:  # var2 samples are given, wants samples of var1
            if var2 is None:
                raise ValueError("Samples of at least one variable must be "
                                 "specified")
            return self.sample_var1_from_var2(var2)
        elif var2 is None:  # var1 samples are given, wants samples of var2
            return self.sample_var2_from_var1(var1)
        else:  # var1 and var2 samples are given, wants samples of observation
            return self.sample_observations(var1, var2)

    def unif_to_sample(self, u: np.ndarray,
                       var1: Union[np.ndarray, None] = None,
                       var2: Union[np.ndarray, None] = None
                       ) -> np.ndarray:
        """
        Nested sampling necessity
        u.shape is (2,) which is determined by JointFactorForNestedSampler
        the known sample has to be on SE2 currently
        """
        assert len(u) == 2
        u_for_dist = u[0]
        u_for_angle = u[1]
        normal_var = scistats.norm.ppf(u_for_angle)  # convert to standard normal
        angle_sample = (self._cov_sqrt * normal_var + self._observation)[0]
        dist_sample = self._min_range + u_for_dist * (self._max_range - self._min_range)

        if var1 is None:  # var2 samples are given, wants samples of var1
            if var2 is None:
                raise ValueError("Samples of at least one variable must be "
                                 "specified")
            assert len(var2) == 3
            var1 = np.zeros(self.var1.dim)
            var1[self.var1.t_dim_indices] = var2[self.var2.t_dim_indices] + np.array([dist_sample * np.cos(var2[self.var2.R_dim_indices][0] + angle_sample),
                                                             dist_sample * np.sin(var2[self.var2.R_dim_indices][0] + angle_sample)])
            return var1
        elif var2 is None:  # var1 samples are given, wants samples of var2
            assert len(var1) == 3
            var2 = np.zeros(self.var2.dim)
            var2[self.var2.t_dim_indices] = var1[self.var1.t_dim_indices] + np.array([dist_sample * np.cos(var1[self.var1.R_dim_indices][0] + angle_sample),
                                                             dist_sample * np.sin(var1[self.var1.R_dim_indices][0] + angle_sample)])
            return var2
        else:  # var1 and var2 samples are given, wants samples of observation
            raise ValueError("Both of var1 and var2 are given.")

    def dvardu(self, top_var: Variable,
               top_arr: np.ndarray,
               bot_var: Variable,
               bot_arr: np.ndarray):
        """
        dtop_var/dbot_var, dtop_var/du
        """
        raise NotImplementedError

    def dvar1du(self, var1, var2):
        raise NotImplementedError

    def dvar2du(self, var1, var2):
        raise NotImplementedError

    def evaluate_loglike(self, x):
        """
        Nested sampling necessity
        x is (dim, ) where dim = var1.dim + var2.dim
        """
        var1_sample = x[0:self.var1.dim]
        var2_sample = x[self.var1.dim:]
        if self.var1.dim == 3:
            delta = np.arctan2(var2_sample[self.var2.t_dim_indices][1] - var1_sample[self.var1.t_dim_indices][1],
                               var2_sample[self.var2.t_dim_indices][0] - var1_sample[self.var1.t_dim_indices][0]) -\
                var1_sample[self.var1.R_dim_indices] -\
                self._observation[0]
        else:
            delta = np.arctan2(var1_sample[self.var1.t_dim_indices][1] - var2_sample[self.var2.t_dim_indices][1],
                               var1_sample[self.var1.t_dim_indices][0] - var2_sample[self.var2.t_dim_indices][0]) -\
                var2_sample[self.var2.R_dim_indices] -\
                self._observation[0]
        delta = theta_to_pipi(delta)
        return -0.5 * (delta ** 2 / self._variance) + self._lnorm

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.log_pdf(x))

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        var1_sample = x[:, 0:self.var1.dim]
        var2_sample = x[:, self.var1.dim:]
        if self.var1.dim == 3:
            delta = np.arctan2(var2_sample[:, self.var2.t_dim_indices[1]] - var1_sample[:, self.var1.t_dim_indices[1]],
                               var2_sample[:, self.var2.t_dim_indices[0]] - var1_sample[:, self.var1.t_dim_indices][0]) -\
                var1_sample[:, self.var1.R_dim_indices] -\
                self._observation[0]
        else:
            delta = np.arctan2(var1_sample[:, self.var1.t_dim_indices[1]] - var2_sample[:, self.var2.t_dim_indices[1]],
                               var1_sample[:, self.var1.t_dim_indices[0]] - var2_sample[:, self.var2.t_dim_indices[0]]) -\
                var2_sample[:, self.var2.R_dim_indices] -\
                self._observation[0]
        delta = theta_to_pipi(delta)
        # log of p(r) is neglected as it is subject a uniform distribution
        return self._noise_distribution.log_pdf(delta)

    def grad_x_log_pdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def is_gaussian(self):
        return False

class SE2BearingLikelihoodFactor(ExplicitLikelihoodFactor, OdomFactor):
    """
    a general bearing factor for R2 and SE(2) variables
    """
    measurement_dim = 1
    measurement_type = Bearing2DVariable

    def __init__(self,
                 var1: Variable,
                 var2: Variable,
                 observation: Union[np.ndarray, float],
                 sigma: float,
                 min_range = .1,
                 max_range = 1.0
                 ) -> None:
        """
        :param var1
        :param var2
        :param bearing: relative radians from var1 to var2
        :param sigma: standard deviation of Gaussian distribution
        :param min_range: min distance between var1 and var2
        :param max_range: max distance between var1 and var2
        """
        assert min_range < max_range
        assert isinstance(var1, SE2Variable)
        assert isinstance(var2, SE2Variable)
        super().__init__(vars=[var1, var2], log_likelihood=None)
        self._observation = observation if isinstance(observation, np.ndarray) \
            else np.array([observation])
        self._noise_distribution = dist.GaussianDistribution(
            mu=np.zeros(1), sigma=np.array([[sigma ** 2]]))
        self._sigma = sigma
        # this is for evaluating log-likelihood
        self._lnorm = -0.5 * np.log(_TWO_PI) - np.log(sigma)
        self._variance = sigma ** 2
        self._cov_sqrt = np.sqrt(self._variance)
        self._observation_var = type(self). \
            measurement_type(name="O" + var1.name + var2.name, variable_type=VariableType.Measurement)

        # visible ranges for the bearing factor
        self._min_range = min_range
        self._max_range = max_range

    @property
    def observation_var(self):
        return self._observation_var

    @property
    def circular_dim_list(self):
        return self._observation_var.circular_dim_list

    @classmethod
    def construct_from_text(cls, line: str, variables: Iterable[Variable]
                            ) -> "SE2BearingLikelihoodFactor":
        line = line.strip().split()
        name_to_var = {var.name: var for var in variables}
        if line[0] == cls.__name__:
            var1 = name_to_var[line[1]]
            var2 = name_to_var[line[2]]
            obs = float(line[3])
            sigma = float(line[4])
            if len(line) == 5:
                factor = cls(var1=var1, var2=var2, observation=obs, sigma=sigma)
            elif len(line) == 6:
                factor = cls(var1=var1, var2=var2, observation=obs, sigma=sigma, min_range=float(line[5]))
            elif len(line) == 7:
                factor = cls(var1=var1, var2=var2, observation=obs, sigma=sigma, min_range=float(line[5]),
                         max_range=float(line[6]))
            else:
                raise ValueError("The number of arguments is incorrect")
        else:
            raise ValueError("The factor name is incorrect")
        return factor

    def __str__(self) -> str:
        line = ["Factor", self.__class__.__name__,
                str(self.vars[0].name), str(self.vars[1].name),
                str(self.observation[0]), str(self.sigma), str(self._min_range), str(self._max_range)]
        return " ".join(line)

    @property
    def observation(self) -> np.ndarray:
        return self._observation

    def sample_var2_from_var1(self, var1_samples: np.ndarray) -> np.ndarray:
        if (len(var1_samples.shape) != 2 or var1_samples.shape[0] == 0 or
                var1_samples.shape[1] != self.var1.dim):
            raise ValueError("The dimensionality of variable 1 is wrong")
        num_samples = var1_samples.shape[0]
        angle_samples = self._noise_distribution.rvs(num_samples) + self._observation
        dist_samples = np.random.uniform(self._min_range, self._max_range, (num_samples, 1))
        var2_samples = deepcopy(var1_samples)
        var2_samples[:, self.var1.t_dim_indices] += np.hstack((dist_samples * np.cos(var2_samples[:, self.var1.R_dim_indices]),
                          dist_samples * np.sin(var2_samples[:, self.var1.R_dim_indices])))
        var2_samples[:, self.var1.R_dim_indices] += angle_samples
        var2_samples[:, self.var1.R_dim_indices] = theta_to_pipi(var2_samples[:, self.var1.R_dim_indices])
        return var2_samples

    def sample_var1_from_var2(self, var2_samples: np.ndarray) -> np.ndarray:
        if (len(var2_samples.shape) != 2 or var2_samples.shape[0] == 0 or
                var2_samples.shape[1] != self.var2.dim):
            raise ValueError("The dimensionality of variable 2 is wrong")
        num_samples = var2_samples.shape[0]
        angle_samples = var2_samples[:, self.var2.R_dim_indices] - self._noise_distribution.rvs(num_samples) - self._observation
        dist_samples = np.random.uniform(self._min_range, self._max_range, (num_samples, 1))
        var1_samples = deepcopy(var2_samples)
        var1_samples[:, self.var2.t_dim_indices] -= np.hstack((dist_samples * np.cos(angle_samples),
                          dist_samples * np.sin(angle_samples)))
        var1_samples[:, self.var2.R_dim_indices] = theta_to_pipi(angle_samples)
        return var1_samples

    def sample_observations(self,
                            var1_samples: np.ndarray,
                            var2_samples: np.ndarray
                            ) -> np.ndarray:
        if not (len(var1_samples.shape) == len(var2_samples.shape) == 2 and
                var1_samples.shape[0] == var2_samples.shape[0] and
                (var1_samples.shape[1] == self.var1.dim) and (var2_samples.shape[1] ==
                                                              self.var2.dim)):
            raise ValueError("Dimensionality of variable 1 or variable 2 is"
                             " wrong")
        num_samples = var1_samples.shape[0]
        noise_samples = self._noise_distribution.rvs(num_samples)
        res = theta_to_pipi(var2_samples[:, self.var2.R_dim_indices] - var1_samples[:, self.var1.R_dim_indices] + noise_samples)
        return res

    @property
    def sigma(self) -> float:
        return self._sigma

    def sample(self,
               var1: Union[np.ndarray, None] = None,
               var2: Union[np.ndarray, None] = None
               ) -> np.ndarray:
        """
        Generate samples with given samples
            When var1 samples are var2 samples are given, generate observation
                samples
            When var2 samples are given, generate var1 samples
            When var1 samples are given, generate var2 samples
        :param var1: samples of var1
        :param var2: samples of var2
        :return: generated samples
        """
        if var1 is None:  # var2 samples are given, wants samples of var1
            if var2 is None:
                raise ValueError("Samples of at least one variable must be "
                                 "specified")
            return self.sample_var1_from_var2(var2)
        elif var2 is None:  # var1 samples are given, wants samples of var2
            return self.sample_var2_from_var1(var1)
        else:  # var1 and var2 samples are given, wants samples of observation
            return self.sample_observations(var1, var2)

    def unif_to_sample(self, u: np.ndarray,
                       var1: Union[np.ndarray, None] = None,
                       var2: Union[np.ndarray, None] = None
                       ) -> np.ndarray:
        """
        Nested sampling necessity
        u.shape is (3,) which is determined by JointFactorForNestedSampler
        3 is more than what we need here. TODO: maybe adding the length of u to the field of class
        var1 and var2 are (3,)
        the known sample has to be on SE2 currently
        """
        assert len(u) == 3
        u_for_dist = u[0]
        u_for_angle = u[1]
        normal_var = scistats.norm.ppf(u_for_angle)  # convert to standard normal
        angle_sample = (self._cov_sqrt * normal_var + self._observation)[0]
        dist_sample = self._min_range + u_for_dist * (self._max_range - self._min_range)

        if var1 is None:  # var2 samples are given, wants samples of var1
            if var2 is None:
                raise ValueError("Samples of at least one variable must be "
                                 "specified")
            assert len(var2) == 3
            var1 = deepcopy(var2)
            var1[self.var1.R_dim_indices] = theta_to_pipi(var1[self.var1.R_dim_indices] - angle_sample)
            var1[self.var1.t_dim_indices] -= np.array([dist_sample * np.cos(var1[self.var1.R_dim_indices][0]),
                                                             dist_sample * np.sin(var1[self.var1.R_dim_indices][0])])
            return var1
        elif var2 is None:  # var1 samples are given, wants samples of var2
            assert len(var1) == 3
            var2 = deepcopy(var1)
            prev_th = var2[self.var2.R_dim_indices][0]
            var2[self.var2.t_dim_indices] += np.array([dist_sample * np.cos(prev_th),
                                                             dist_sample * np.sin(prev_th)])
            var2[self.var2.R_dim_indices] = theta_to_pipi(prev_th + angle_sample)
            return var2
        else:  # var1 and var2 samples are given, wants samples of observation
            raise ValueError("Both of var1 and var2 are given.")

    def dvardu(self, top_var: Variable,
               top_arr: np.ndarray,
               bot_var: Variable,
               bot_arr: np.ndarray):
        """
        dtop_var/dbot_var, dtop_var/du
        """
        raise NotImplementedError

    def dvar1du(self, var1, var2):
        raise NotImplementedError

    def dvar2du(self, var1, var2):
        raise NotImplementedError

    def evaluate_loglike(self, x):
        """
        Nested sampling necessity
        x is (dim, ) where dim = var1.dim + var2.dim
        """
        var1_sample = x[0:self.var1.dim]
        var2_sample = x[self.var1.dim:]
        delta = var2_sample[self.var2.R_dim_indices] -\
                var1_sample[self.var1.R_dim_indices] -\
                self._observation[0]
        return -0.5 * (delta ** 2 / self._variance) + self._lnorm

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.log_pdf(x))

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        var1_sample = x[:, 0:self.var1.dim]
        var2_sample = x[:, self.var1.dim:]
        delta = var2_sample[:, self.var2.R_dim_indices] - \
                var1_sample[:, self.var1.R_dim_indices] - \
                self._observation[0]
        # log of p(r) is neglected as it is subject a uniform distribution
        return self._noise_distribution.log_pdf(delta)

    def grad_x_log_pdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def is_gaussian(self):
        return False

class RelativeGaussianSlipGripSE2Factor(LikelihoodFactor, OdomFactor):
    """
        Likelihood factor on SE(2)
        When gripping occurs, it is just a usual relative factor
        When slipping occurs, the actual relative displacement is zero
    """
    measurement_dim = 3
    measurement_type = SE2Variable

    def __init__(self,
                 var1: Variable,
                 var2: Variable,
                 observation: SE2Pose,
                 covariance: np.ndarray = None,
                 prob_slip: float = 0.0,
                 correlated_Rt: bool = True
                 ) -> None:
        """
        :param var1
        :param var2
        :param observation: observed relative pose from var1 to var2
        :param covariance: sigma matrix
            coordinate ordering must be x, y, theta
        :param prob_slip: the probability of slipping
        :param correlated_Rt
        """
        dim = var1.dim
        self._translation_dim = 2
        super().__init__(vars=[var1, var2], log_likelihood=None)
        self._unary_dim = dim
        self._observation = observation
        self._grip_factor = SE2RelativeGaussianLikelihoodFactor(
            var1=var1,
            var2=var2,
            observation=observation,
            covariance=covariance,
            correlated_R_t=correlated_Rt)
        self._prob_slip = prob_slip
        self._correlated_Rt = correlated_Rt
        self._est_rot_dispersion = 1.0 / covariance[self._translation_dim,
                                                    self._translation_dim]
        self._pose_log_map = observation.log_map()
        self._noise_distribution = dist.GaussianDistribution(mu=np.zeros(dim),
                                                             sigma=covariance)
        self._observation_var = type(self). \
            measurement_type(name="O" + var1.name + var2.name, variable_type=VariableType.Measurement)

    @property
    def observation_var(self):
        return self._observation_var

    @property
    def circular_dim_list(self):
        return self._observation_var.circular_dim_list

    def observation(self) -> np.ndarray:
        return self._observation.array

    def sample(self,
               var1: Union[np.ndarray, SE2Pose, None] = None,
               var2: Union[np.ndarray, SE2Pose, None] = None
               ) -> np.ndarray:
        """
        Generate samples with given samples
            When var1 samples are var2 samples are given, generate observation
                samples
            When var2 samples are given, generate var1 samples
            When var1 samples are given, generate var2 samples
        :param var1: samples of var1
        :param var2: samples of var2
        :return: generated samples
        """
        if var1 is None:  # var2 samples are given, wants samples of var1
            if var2 is None:
                raise ValueError("Samples of at least one variable must be "
                                 "specified")
            if (len(var2.shape) != 2 or var2.shape[0] == 0 or
                    var2.shape[1] != self._unary_dim):
                raise ValueError("The dimensionality of variable 2 is wrong")
            num_samples = var2.shape[0]
            r = np.random.random(num_samples)

            # Generate noise samples in SE(2) Lie algebra
            noise_samples = self._noise_distribution.rvs(num_samples)
            var1_samples = np.zeros_like(var2)
            if self._correlated_Rt:  # wrapped Gaussian distribution
                for i in range(num_samples):
                    T_j = SE2Pose.by_array(var2[i])
                    # T_ij_log_map = noise_samples[i] if r < self._prob_slip \
                    #     else self._pose_log_map + noise_samples[i]
                    # T_ij_noised = SE2Pose.by_exp_map(T_ij_log_map)
                    T_ij_noised = SE2Pose.by_exp_map(noise_samples[i]) if r < self._prob_slip \
                        else self._observation * SE2Pose.by_exp_map(noise_samples[i])
                    T_i = T_j / T_ij_noised
                    var1_samples[i] = T_i.array
            else:  # Gaussian for translation, von Mises for rotation
                raise NotImplementedError("Von Mises distribution not"
                                          "implemented")
            return var1_samples
        elif var2 is None:  # var1 samples are given, wants samples of var2
            if (len(var1.shape) != 2 or var1.shape[0] == 0 or
                    var1.shape[1] != self._unary_dim):
                raise ValueError("The dimensionality of variable 1 is wrong")
            num_samples = var1.shape[0]
            r = np.random.random(num_samples)
            noise_samples = self._noise_distribution.rvs(num_samples)
            var2_samples = np.zeros_like(var1)
            if self._correlated_Rt:  # wrapped Gaussian distribution
                for i in range(num_samples):
                    T_i = SE2Pose.by_array(var1[i])
                    # T_ij_log_map = noise_samples[i] if r[i] < self._prob_slip \
                    #     else self._pose_log_map + noise_samples[i]
                    # T_ij_noised = SE2Pose.by_exp_map(T_ij_log_map)
                    T_ij_noised = SE2Pose.by_exp_map(noise_samples[i]) if r < self._prob_slip \
                        else self._observation * SE2Pose.by_exp_map(noise_samples[i])
                    T_j = T_i * T_ij_noised
                    var2_samples[i] = T_j.array
            else:  # Gaussian for translation, von Mises for rotation
                raise NotImplementedError("Von Mises distribution not "
                                          "implemented")
            return var2_samples
        else:  # var1 and var2 samples are given, wants samples of observation
            if not (len(var1.shape) == len(var2.shape) == 2 and
                    var1.shape[0] == var2.shape[0] and
                    var1.shape[1] == var2.shape[1] == self._unary_dim):
                raise ValueError("Dimensionality of variable 1 or variable 2 is"
                                 " wrong")
            num_samples = var1.shape[0]
            r = np.random.random(num_samples)
            noise_samples = self._noise_distribution.rvs(num_samples)
            obs_samples = np.zeros_like(var1)
            if self._correlated_Rt:  # wrapped Gaussian distribution
                for i in range(num_samples):
                    # TODO: is there an if-else here?
                    if r < self._prob_slip:
                        T_ij_log_map = self._pose_log_map + noise_samples[i]
                        T_ij_noised = SE2Pose.by_exp_map(T_ij_log_map)
                        obs_samples[i] = T_ij_noised.array
                    else:
                        T_i = SE2Pose.by_array(var1[i])
                        T_j = SE2Pose.by_array(var2[i])
                        T_ij = T_i.inverse() * T_j
                        T_ij_noised = T_ij * SE2Pose.by_exp_map(noise_samples[i])
                        obs_samples[i] = T_ij_noised.array
            else:  # Gaussian for translation, von Mises for rotation
                raise NotImplementedError("Von Mises distribution not "
                                          "implemented")
            return obs_samples


# class RelativeGaussianMixtureSE2Factor(LikelihoodFactor):
#     """
#     Likelihood factor on SE(2)
#     """
#     def __init__(self,
#                  var1: Variable,
#                  var2: Variable,
#                  observation: Pose2,
#                  relative_means: List[Pose2],
#                  covariances: List[np.ndarray] = None,
#                  correlated_Rts: List[bool] = True
#                  ) -> None:
#         """
#         :param var1
#         :param var2
#         :param relative_means: list of relative pose2 from var1 to var2
#         :param covariances: list of sigma matrix
#             coordinate ordering must be x, y, theta
#         :param correlated_Rts
#         """
#         dim = var1.dim
#         self._translation_dim = 2
#         super().__init__(vars=[var1, var2], log_likelihood=None)
#         self._unary_dim = dim
#         self._observation = observation
#         self._unimodal_factors = [RelativeGaussianSE2Factor(var1, var2,
#                                                             observation,
#                                                             relative_means[i],
#                                                             covariances[i],
#                                                             correlated_Rts[i])
#                                   for i in range(len(relative_means))]
#
#     @property
#     def observation(self):
#         return self._observation
#
#     def sample(self,
#                var1: Union[np.ndarray, Pose2, None] = None,
#                var2: Union[np.ndarray, Pose2, None] = None
#                ) -> np.ndarray:
#         """
#         Generate samples with given samples
#             When var1 samples are var2 samples are given, generate observation
#                 samples
#             When var2 samples are given, generate var1 samples
#             When var1 samples are given, generate var2 samples
#         :param var1: samples of var1
#         :param var2: samples of var2
#         :return: generated samples
#         """
#         if var1 is None:  # var2 samples are given, wants samples of var1
#             if var2 is None:
#                 raise ValueError("Samples of at least one variable must be "
#                                  "specified")
#             if (len(var2.shape) != 2 or var2.shape[0] == 0 or
#                     var2.shape[1] != self._unary_dim):
#                 raise ValueError("The dimensionality of variable 2 is wrong")
#             num_samples = var2.shape[0]
#
#             # Generate noise samples in SE(2) Lie algebra
#             noise_samples = self._noise_distribution.rvs(num_samples)
#             var1_samples = np.zeros_like(var2)
#             if self._correlated_R_t:  # wrapped Gaussian distribution
#                 for i in range(num_samples):
#                     T_j = Pose2.by_array(var2[i])
#                     T_ij_log_map = self._pose_log_map + noise_samples[i]
#                     T_ij_noised = Pose2.by_exp_map(T_ij_log_map)
#                     T_i = T_j / T_ij_noised
#                     var1_samples[i] = T_i.array
#             else:  # Gaussian for translation, von Mises for rotation
#                 theta_array = np.random.vonmises(mu=0.0,
#                                                  kappa=self._est_rot_dispersion,
#                                                  size=num_samples)
#                 for i in range(num_samples):
#                     T_j = Pose2.by_array(var2[i, :])
#                     t_noise = Point2.by_array(noise_samples[i,
#                                               0:self._translation_dim])
#                     R_noise = Rot2(theta=theta_array[i])
#                     R_i = T_j.rotation \
#                           / R_noise \
#                           / self._observation.rotation
#                     t_i = T_j.translation - \
#                           R_i * (self._observation.translation +
#                                  t_noise)
#                     var1_samples[i, 0] = t_i.x
#                     var1_samples[i, 1] = t_i.y
#                     var1_samples[i, 2] = R_i.theta
#             return var1_samples
#         elif var2 is None:  # var1 samples are given, wants samples of var2
#             if (len(var1.shape) != 2 or var1.shape[0] == 0 or
#                     var1.shape[1] != self._unary_dim):
#                 raise ValueError("The dimensionality of variable 1 is wrong")
#             num_samples = var1.shape[0]
#             noise_samples = self._noise_distribution.rvs(num_samples)
#             var2_samples = np.zeros_like(var1)
#             if self._correlated_R_t:  # wrapped Gaussian distribution
#                 for i in range(num_samples):
#                     T_i = Pose2.by_array(var1[i])
#                     T_ij_log_map = self._pose_log_map + noise_samples[i]
#                     T_ij_noised = Pose2.by_exp_map(T_ij_log_map)
#                     T_j = T_i * T_ij_noised
#                     var2_samples[i] = T_j.array
#             else:  # Gaussian for translation, von Mises for rotation
#                 theta_array = np.random.vonmises(mu=0.0,
#                                                  kappa=self._est_rot_dispersion,
#                                                  size=num_samples)
#                 for i in range(num_samples):
#                     T_i = Pose2.by_array(var1[i])
#                     t_noise = Point2.by_array(noise_samples[i,
#                                               0:self._translation_dim])
#                     R_noise = Rot2(theta=theta_array[i])
#                     R_j = T_i.rotation \
#                           * self._observation.rotation \
#                           * R_noise
#                     t_j = T_i.translation + \
#                           T_i.rotation * (self._observation.translation +
#                                  t_noise)
#                     var2_samples[i, 0] = t_j.x
#                     var2_samples[i, 1] = t_j.y
#                     var2_samples[i, 2] = R_j.theta
#             return var2_samples
#         else:  # var1 and var2 samples are given, wants samples of observation
#             if not (len(var1.shape) == len(var2.shape) == 2 and
#                     var1.shape[0] == var2.shape[0] and
#                     var1.shape[1] == var2.shape[1] == self._unary_dim):
#                 raise ValueError("Dimensionality of variable 1 or variable 2 is"
#                                  " wrong")
#             num_samples = var1.shape[0]
#             noise_samples = self._noise_distribution.rvs(num_samples)
#             obs_samples = np.zeros_like(var1)
#             if self._correlated_R_t:  # wrapped Gaussian distribution
#                 for i in range(num_samples):
#                     T_i = Pose2.by_array(var1[i])
#                     T_j = Pose2.by_array(var2[i])
#                     T_ij_log_map = (T_i.inverse() * T_j).log_map()\
#                                    + noise_samples[i]
#                     T_ij_noised = Pose2.by_exp_map(T_ij_log_map)
#                     obs_samples[i] = T_ij_noised.array
#             else: #Gaussian for translation, von Mises for rotation
#                 theta_array = np.random.vonmises(mu=0.0,
#                                                  kappa=self._est_rot_dispersion,
#                                                  size=num_samples)
#                 for i in range(num_samples):
#                     T_i = Pose2.by_array(var1[i])
#                     T_j = Pose2.by_array(var2[i])
#                     T_ij = T_i.inverse() * T_j
#                     t_noise = Point2.by_array(noise_samples[i,
#                                               0:self._translation_dim])
#                     R_noise = Rot2(theta=theta_array[i])
#                     R_ij = T_ij.rotation * R_noise
#                     t_ij = T_ij.translation + t_noise
#                     obs_samples[i, 0] = t_ij.x
#                     obs_samples[i, 1] = t_ij.y
#                     obs_samples[i, 2] = R_ij.theta
#             return obs_samples

class R2RangeGaussianLikelihoodFactor(ExplicitLikelihoodFactor,
                                      LikelihoodFactor, BinaryFactor):
    """
    Likelihood factor on R(2)
    """
    measurement_dim = 1
    measurement_type = R1Variable

    def __init__(self, var1: Variable, var2: Variable,
                 observation: Union[np.ndarray, float], sigma: float = 1.0,
                 ) -> None:
        """
        :param var1
        :param var2
        :param observation: observed distance from var1 to var2
        :param sigma: standard deviation of Gaussian distribution TODO: change it to sigma?
        """
        super().__init__(vars=[var1, var2], log_likelihood=None)
        self._unary_dim = 2
        self._observation = observation if isinstance(observation, np.ndarray) \
            else np.array([observation])
        self._noise_distribution = dist.GaussianDistribution(
            mu=np.zeros(1), sigma=np.array([[sigma ** 2]]))
        self._sigma = sigma
        # this is for evaluating log-likelihood
        self._lnorm = -0.5 * np.log(_TWO_PI) - np.log(sigma)
        self._variance = sigma ** 2
        self._cov_sqrt = np.sqrt(self._variance)
        self._observation_var = type(self). \
            measurement_type(name="O" + var1.name + var2.name, variable_type=VariableType.Measurement)

    @property
    def observation_var(self):
        return self._observation_var

    @property
    def circular_dim_list(self):
        return self._observation_var.circular_dim_list

    @classmethod
    def construct_from_text(cls, line: str, variables: Iterable[Variable]
                            ) -> "R2RangeGaussianLikelihoodFactor":
        line = line.strip().split()
        name_to_var = {var.name: var for var in variables}
        if line[0] == cls.__name__:
            var1 = name_to_var[line[1]]
            var2 = name_to_var[line[2]]
            obs = float(line[3])
            sigma = float(line[4])
            factor = cls(var1=var1, var2=var2, observation=obs, sigma=sigma)
        else:
            raise ValueError("The factor name is incorrect")
        return factor

    def __str__(self) -> str:
        line = ["Factor", self.__class__.__name__,
                str(self.vars[0].name), str(self.vars[1].name),
                str(self.observation[0]), str(self.sigma)]
        return " ".join(line)

    @property
    def observation(self) -> np.ndarray:
        return self._observation

    def sample_var2_from_var1(self, var1_samples: np.ndarray) -> np.ndarray:
        if (len(var1_samples.shape) != 2 or var1_samples.shape[0] == 0 or
                var1_samples.shape[1] != self._unary_dim):
            raise ValueError("The dimensionality of variable 1 is wrong")
        num_samples = var1_samples.shape[0]
        noise_samples = self._noise_distribution.rvs(num_samples)
        dist_samples = (np.zeros((num_samples, 1)) + self._observation +
                        noise_samples)
        angle_samples = np.random.uniform(-np.pi, np.pi, num_samples).reshape(
            (num_samples, 1))
        return var1_samples + np.hstack((dist_samples * np.cos(angle_samples),
                                         dist_samples * np.sin(angle_samples)))

    def sample_var1_from_var2(self, var2_samples: np.ndarray) -> np.ndarray:
        if (len(var2_samples.shape) != 2 or var2_samples.shape[0] == 0 or
                var2_samples.shape[1] != self._unary_dim):
            raise ValueError("The dimensionality of variable 2 is wrong")
        num_samples = var2_samples.shape[0]

        # Generate noise samples in range
        noise_samples = self._noise_distribution.rvs(num_samples)
        dist_samples = (np.zeros((num_samples, 1)) + self._observation +
                        noise_samples)
        angle_samples = np.random.uniform(-np.pi, np.pi, num_samples).reshape(
            num_samples, 1)
        return var2_samples + np.hstack((dist_samples * np.cos(angle_samples),
                                         dist_samples * np.sin(angle_samples)))

    def sample_observations(self,
                            var1_samples: np.ndarray,
                            var2_samples: np.ndarray
                            ) -> np.ndarray:
        if not (len(var1_samples.shape) == len(var2_samples.shape) == 2 and
                var1_samples.shape[0] == var2_samples.shape[0] and
                var1_samples.shape[1] == var2_samples.shape[1] ==
                self._unary_dim):
            raise ValueError("Dimensionality of variable 1 or variable 2 is"
                             " wrong")
        num_samples = var1_samples.shape[0]
        noise_samples = self._noise_distribution.rvs(num_samples)
        res = np.sqrt(np.sum((var2_samples - var1_samples) ** 2, axis=1
                             )).reshape((num_samples, 1)) + noise_samples
        return res

    @property
    def sigma(self) -> float:
        return self._sigma

    def sample(self,
               var1: Union[np.ndarray, None] = None,
               var2: Union[np.ndarray, None] = None
               ) -> np.ndarray:
        """
        Generate samples with given samples
            When var1 samples are var2 samples are given, generate observation
                samples
            When var2 samples are given, generate var1 samples
            When var1 samples are given, generate var2 samples
        :param var1: samples of var1
        :param var2: samples of var2
        :return: generated samples
        """
        if var1 is None:  # var2 samples are given, wants samples of var1
            if var2 is None:
                raise ValueError("Samples of at least one variable must be "
                                 "specified")
            return self.sample_var1_from_var2(var2)
        elif var2 is None:  # var1 samples are given, wants samples of var2
            return self.sample_var2_from_var1(var1)
        else:  # var1 and var2 samples are given, wants samples of observation
            return self.sample_observations(var1, var2)

    def unif_to_sample(self, u: np.ndarray,
                       var1: Union[np.ndarray, None] = None,
                       var2: Union[np.ndarray, None] = None
                       ) -> np.ndarray:
        # for R2 this u.shape is (2,)
        assert len(u) == 2
        u_for_dist = u[0]
        u_for_angle = u[1]
        normal_var = scistats.norm.ppf(u_for_dist)  # convert to standard normal
        dist_sample = (self._cov_sqrt * normal_var + self._observation)[0]
        angle_sample = (u_for_angle - 0.5) * _TWO_PI

        if var1 is None:  # var2 samples are given, wants samples of var1
            if var2 is None:
                raise ValueError("Samples of at least one variable must be "
                                 "specified")
            return var2 + np.array([dist_sample * np.cos(angle_sample),
                                    dist_sample * np.sin(angle_sample)])
        elif var2 is None:  # var1 samples are given, wants samples of var2
            return var1 + np.array([dist_sample * np.cos(angle_sample),
                                    dist_sample * np.sin(angle_sample)])
        else:  # var1 and var2 samples are given, wants samples of observation
            raise ValueError("Both of var1 and var2 are given.")

    def evaluate_loglike(self, x):
        var1 = x[0:self._unary_dim]
        var2 = x[self._unary_dim:]
        delta = np.linalg.norm(var1 - var2) - self._observation[0]
        return -0.5 * (delta ** 2 / self._variance) + self._lnorm

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.log_pdf(x))

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        var1_sample = x[:, 0:self.var1.dim]
        var2_sample = x[:, self.var1.dim:]
        delta = np.linalg.norm(var1_sample[:, self.var1.t_dim_indices] -
                               var2_sample[:, self.var2.t_dim_indices], axis=1, keepdims=True) - \
                self._observation[0]
        return self._noise_distribution.log_pdf(delta)

    def grad_x_log_pdf(self, x: np.ndarray) -> np.ndarray:
        var1_sample = x[:, 0:self.var1.dim]
        var2_sample = x[:, self.var1.dim:]
        distance = np.linalg.norm(var1_sample[:, self.var1.t_dim_indices] -
                                  var2_sample[:, self.var2.t_dim_indices], axis=1, keepdims=True)
        diff = var1_sample[:, self.var1.t_dim_indices] - var2_sample[:, self.var2.t_dim_indices]
        delta = distance - self._observation[0]
        res = np.zeros_like(x)
        res[:, 0:self.var1.dim][:, self.var1.t_dim_indices] = diff
        res[:, self.var1.dim:][:, self.var2.t_dim_indices] = -diff

        low_dist_idx = np.where(distance < 1e-8)[0]
        others = np.array([idx for idx in range(x.shape[0]) if idx not in low_dist_idx])
        if len(low_dist_idx) > 0:
            res[low_dist_idx] = (res[low_dist_idx] / 1e-8) * self._noise_distribution.grad_x_log_pdf(delta[low_dist_idx]).reshape((-1, 1))
        if len(others) > 0:
            res[others] = (res[others] / distance[others]) * self._noise_distribution.grad_x_log_pdf(delta[others]).reshape((-1, 1))
        return res

    @property
    def is_gaussian(self):
        return False

class UnaryR2RangeGaussianPriorFactor(ExplicitPriorFactor, UnaryFactor,
                                      metaclass=ABCMeta):
    measurement_variable_type = R1Variable

    def __init__(self, var: R2Variable, center: np.ndarray,
                 mu: float,
                 sigma: float) -> None:
        """
        Params:
        sigma: float
        """
        self._distribution = dists.GaussianRangeDistribution(
            center=center, mu=mu, sigma=sigma ** 2)
        super().__init__([var], distribution=self._distribution)
        self._covariance = sigma ** 2
        self._precision = 1.0 / self._covariance
        self._cov_sqrt = sigma
        self._lnorm = -0.5 * (np.log(_TWO_PI) +
                              np.log(self._covariance))  # ln(normalization)

    @property
    def vars(self) -> List[R2Variable]:
        return self._vars

    @property
    def mu(self) -> float:
        return self._distribution.mean

    @property
    def covariance(self) -> float:
        return self._distribution.sigma

    @property
    def center(self) -> np.ndarray:
        return self._distribution.center

    def __str__(self) -> str:
        line = ["Factor", self.__class__.__name__, str(self.vars[0].name),
                "center:", str(self.center[0]), str(self.center[1]),
                "mu:", str(self.mu),
                "sigma", str(self.covariance)]
        return " ".join(line)

    @classmethod
    def construct_from_text(cls, line: str, variables
                            ) -> "UnaryR2RangeGaussianPriorFactor":
        line = line.strip().split()
        name_to_var = {var.name: var for var in variables}
        if line[0] == cls.__name__:
            var = name_to_var[line[1]]
            center = np.array([float(line[2]), float(line[3])])
            mu = float(line[4])
            variance = float(line[5])
            factor = cls(
                var=var, mu=mu, covariance=variance)
        else:
            raise ValueError("The factor name is incorrect")
        return factor

    def unif_to_sample(self, u) -> np.array:
        # u is a (2,))numpy array
        # return a sample on R2
        # for R2 this u.shape is (2,)
        assert len(u) == 2
        u_for_dist = u[0]
        u_for_angle = u[1]
        normal_var = scistats.norm.ppf(u_for_dist)  # convert to standard normal
        dist_sample = (self._cov_sqrt * normal_var + self.mu)
        angle_sample = (u_for_angle - 0.5) * _TWO_PI

        return self.center + np.array([dist_sample * np.cos(angle_sample),
                                       dist_sample * np.sin(angle_sample)])

    @property
    def observation(self):
        return self.mu

    def evaluate_loglike(self, x):
        delta = (np.linalg.norm(x - self.center.flatten() - self.observation))
        return -0.5 * np.dot(delta, np.dot(self._precision, delta)) + self._lnorm

    @property
    def is_gaussian(self):
        return False

class UncertainR2RangeGaussianLikelihoodFactor(ExplicitLikelihoodFactor,
                                      LikelihoodFactor, BinaryFactor):
    """
    Likelihood factor on R(2)
    Inspired by the SNL example in https://arxiv.org/abs/1812.02609
    The likelihood model for this factor can be found in the paper
    """
    measurement_dim = 1
    measurement_type = R1Variable

    def __init__(self, var1: Variable, var2: Variable,
                 observation: Union[np.ndarray, float], sigma: float = 1.0,
                 observed_flag: bool = False, unobserved_sigma: float = .3
                 ) -> None:
        """
        :param var1
        :param var2
        :param observation: observed distance from var1 to var2
        :param sigma: standard deviation of Gaussian distribution for the range measurement
        :param observed_flag: true if the observation is valid
        :param unobserved_sigma: standard deviation of Gaussian distribution for the observability
        """
        super().__init__(vars=[var1, var2], log_likelihood=None)
        self._unary_dim = 2
        self._observation = observation if isinstance(observation, np.ndarray) \
            else np.array([observation])
        self._sigma = sigma
        # this is for evaluating log-likelihood
        self._observation_var = type(self). \
            measurement_type(name="O" + var1.name + var2.name, variable_type=VariableType.Measurement)
        self._observed_flag = observed_flag
        self._unobserved_sigma = unobserved_sigma

        self._new_var = self._sigma **2 * self._unobserved_sigma **2 / (self._sigma **2 + self._unobserved_sigma **2)
        self._new_mu = self._unobserved_sigma **2 * self._observation[0] / (self._sigma **2 + self._unobserved_sigma **2)
        self._noise_distribution = dist.GaussianDistribution(
            mu=np.zeros(1), sigma=np.array([[self._new_var]]))
        self._variance = self._new_var
        self._cov_sqrt = np.sqrt(self._new_var)

    @property
    def observation_var(self):
        return self._observation_var

    @property
    def circular_dim_list(self):
        return self._observation_var.circular_dim_list

    @classmethod
    def construct_from_text(cls, line: str, variables: Iterable[Variable]
                            ) -> "UncertainR2RangeGaussianLikelihoodFactor":
        line = line.strip().split()
        name_to_var = {var.name: var for var in variables}
        if line[0] == cls.__name__:
            var1 = name_to_var[line[1]]
            var2 = name_to_var[line[2]]
            obs = float(line[3])
            sigma = float(line[4])
            flag = bool(int(line[5]))
            obs_sigma = int(line[6])
            factor = cls(var1=var1, var2=var2, observation=obs,
                         sigma=sigma, observed_flag=flag, unobserved_sigma=obs_sigma)
        else:
            raise ValueError("The factor name is incorrect")
        return factor

    def __str__(self) -> str:
        line = ["Factor", self.__class__.__name__,
                str(self.vars[0].name), str(self.vars[1].name),
                str(self.observation[0]), str(self.sigma), str(int(self.observed_flag)), str(self.unobserved_sigma)]
        return " ".join(line)

    @property
    def observed_flag(self) -> bool:
        return self._observed_flag

    @property
    def unobserved_sigma(self) -> float:
        return self._unobserved_sigma

    @property
    def observation(self) -> np.ndarray:
        return self._observation

    def sample_var2_from_var1(self, var1_samples: np.ndarray) -> np.ndarray:
        if (len(var1_samples.shape) != 2 or var1_samples.shape[0] == 0 or
                var1_samples.shape[1] != self._unary_dim):
            raise ValueError("The dimensionality of variable 1 is wrong")
        num_samples = var1_samples.shape[0]
        noise_samples = self._noise_distribution.rvs(num_samples)
        dist_samples = (np.zeros((num_samples, 1)) + self._new_mu +
                        noise_samples)
        angle_samples = np.random.uniform(-np.pi, np.pi, num_samples).reshape(
            (num_samples, 1))
        return var1_samples + np.hstack((dist_samples * np.cos(angle_samples),
                                         dist_samples * np.sin(angle_samples)))

    def sample_var1_from_var2(self, var2_samples: np.ndarray) -> np.ndarray:
        if (len(var2_samples.shape) != 2 or var2_samples.shape[0] == 0 or
                var2_samples.shape[1] != self._unary_dim):
            raise ValueError("The dimensionality of variable 2 is wrong")
        num_samples = var2_samples.shape[0]

        # Generate noise samples in range
        noise_samples = self._noise_distribution.rvs(num_samples)
        dist_samples = (np.zeros((num_samples, 1)) + self._new_mu +
                        noise_samples)
        angle_samples = np.random.uniform(-np.pi, np.pi, num_samples).reshape(
            num_samples, 1)
        return var2_samples + np.hstack((dist_samples * np.cos(angle_samples),
                                         dist_samples * np.sin(angle_samples)))

    def sample_observations(self,
                            var1_samples: np.ndarray,
                            var2_samples: np.ndarray
                            ) -> np.ndarray:
        if not (len(var1_samples.shape) == len(var2_samples.shape) == 2 and
                var1_samples.shape[0] == var2_samples.shape[0] and
                var1_samples.shape[1] == var2_samples.shape[1] ==
                self._unary_dim):
            raise ValueError("Dimensionality of variable 1 or variable 2 is"
                             " wrong")
        num_samples = var1_samples.shape[0]
        noise_samples = self._noise_distribution.rvs(num_samples)
        res = np.sqrt(np.sum((var2_samples - var1_samples) ** 2, axis=1
                             )).reshape((num_samples, 1)) + noise_samples
        return res

    @property
    def sigma(self) -> float:
        return self._sigma

    def sample(self,
               var1: Union[np.ndarray, None] = None,
               var2: Union[np.ndarray, None] = None
               ) -> np.ndarray:
        """
        Generate samples with given samples
            When var1 samples are var2 samples are given, generate observation
                samples
            When var2 samples are given, generate var1 samples
            When var1 samples are given, generate var2 samples
        :param var1: samples of var1
        :param var2: samples of var2
        :return: generated samples
        """
        assert self._observed_flag == True

        if var1 is None:  # var2 samples are given, wants samples of var1
            if var2 is None:
                raise ValueError("Samples of at least one variable must be "
                                 "specified")
            return self.sample_var1_from_var2(var2)
        elif var2 is None:  # var1 samples are given, wants samples of var2
            return self.sample_var2_from_var1(var1)
        else:  # var1 and var2 samples are given, wants samples of observation
            return self.sample_observations(var1, var2)

    def unif_to_sample(self, u: np.ndarray,
                       var1: Union[np.ndarray, None] = None,
                       var2: Union[np.ndarray, None] = None
                       ) -> np.ndarray:
        # for R2 this u.shape is (2,)
        assert self._observed_flag == True

        assert len(u) == 2
        u_for_dist = u[0]
        u_for_angle = u[1]
        normal_var = scistats.norm.ppf(u_for_dist)  # convert to standard normal
        dist_sample = (self._cov_sqrt * normal_var + self._new_mu)
        angle_sample = (u_for_angle - 0.5) * _TWO_PI

        if var1 is None:  # var2 samples are given, wants samples of var1
            if var2 is None:
                raise ValueError("Samples of at least one variable must be "
                                 "specified")
            return var2 + np.array([dist_sample * np.cos(angle_sample),
                                    dist_sample * np.sin(angle_sample)])
        elif var2 is None:  # var1 samples are given, wants samples of var2
            return var1 + np.array([dist_sample * np.cos(angle_sample),
                                    dist_sample * np.sin(angle_sample)])
        else:  # var1 and var2 samples are given, wants samples of observation
            raise ValueError("Both of var1 and var2 are given.")

    def evaluate_loglike(self, x):
        var1 = x[0:self._unary_dim]
        var2 = x[self._unary_dim:]
        delta = np.linalg.norm(var1 - var2)
        if self._observed_flag == False:
            return np.log(1- np.exp(-0.5 * delta ** 2 / self._unobserved_sigma **2))
        else:
            return -0.5 * (delta - self._new_mu) ** 2 / self._new_var

    @property
    def is_gaussian(self):
        return False



class SE2R2RangeGaussianLikelihoodFactor(ExplicitLikelihoodFactor, BinaryFactor):
    """
    a general range factor for R2 and SE(2) variables
    """
    measurement_dim = 1
    measurement_type = R1Variable

    def __init__(self, var1: Variable, var2: Variable,
                 observation: Union[np.ndarray, float], sigma: float = 1.0,
                 ) -> None:
        """
        :param var1
        :param var2
        :param observation: observed distance from var1 to var2
        :param sigma: standard deviation of Gaussian distribution
        """
        super().__init__(vars=[var1, var2], log_likelihood=None)
        # self._unary_dim = 2
        self._observation = observation if isinstance(observation, np.ndarray) \
            else np.array([observation])
        self._noise_distribution = dist.GaussianDistribution(
            mu=np.zeros(1), sigma=np.array([[sigma ** 2]]))
        # this is for the observation variable
        # self._circular_dim_list = [False]
        self._sigma = sigma
        # this is for evaluating log-likelihood
        self._lnorm = -0.5 * np.log(_TWO_PI) - np.log(sigma)
        self._variance = sigma ** 2
        self._cov_sqrt = np.sqrt(self._variance)
        self._observation_var = type(self). \
            measurement_type(name="O" + var1.name + var2.name, variable_type=VariableType.Measurement)

    @property
    def observation_var(self):
        return self._observation_var

    @property
    def circular_dim_list(self):
        return self._observation_var.circular_dim_list

    @classmethod
    def construct_from_text(cls, line: str, variables: Iterable[Variable]
                            ) -> "SE2R2RangeGaussianLikelihoodFactor":
        line = line.strip().split()
        name_to_var = {var.name: var for var in variables}
        if line[0] == cls.__name__:
            var1 = name_to_var[line[1]]
            var2 = name_to_var[line[2]]
            obs = float(line[3])
            sigma = float(line[4])
            factor = cls(var1=var1, var2=var2, observation=obs, sigma=sigma)
        else:
            raise ValueError("The factor name is incorrect")
        return factor

    def __str__(self) -> str:
        line = ["Factor", self.__class__.__name__,
                str(self.vars[0].name), str(self.vars[1].name),
                str(self.observation[0]), str(self.sigma)]
        return " ".join(line)

    @property
    def observation(self) -> np.ndarray:
        return self._observation

    def sample_var2_from_var1(self, var1_samples: np.ndarray) -> np.ndarray:
        if (len(var1_samples.shape) != 2 or var1_samples.shape[0] == 0 or
                var1_samples.shape[1] != self.var1.dim):
            raise ValueError("The dimensionality of variable 1 is wrong")
        num_samples = var1_samples.shape[0]
        noise_samples = self._noise_distribution.rvs(num_samples)
        dist_samples = (np.zeros((num_samples, 1)) + self._observation +
                        noise_samples)
        angle_samples = np.random.uniform(-np.pi, np.pi, num_samples).reshape(
            (num_samples, 1))
        return var1_samples[:, self.var1.t_dim_indices] + \
               np.hstack((dist_samples * np.cos(angle_samples),
                          dist_samples * np.sin(angle_samples)))

    def sample_var1_from_var2(self, var2_samples: np.ndarray) -> np.ndarray:
        if (len(var2_samples.shape) != 2 or var2_samples.shape[0] == 0 or
                var2_samples.shape[1] != self.var2.dim):
            raise ValueError("The dimensionality of variable 2 is wrong")
        num_samples = var2_samples.shape[0]

        # Generate noise samples in range
        noise_samples = self._noise_distribution.rvs(num_samples)
        dist_samples = (np.zeros((num_samples, 1)) + self._observation +
                        noise_samples)
        angle_samples = np.random.uniform(-np.pi, np.pi, num_samples).reshape(
            num_samples, 1)
        return var2_samples[:, self.var2.t_dim_indices] + \
               np.hstack((dist_samples * np.cos(angle_samples),
                          dist_samples * np.sin(angle_samples)))

    def sample_observations(self,
                            var1_samples: np.ndarray,
                            var2_samples: np.ndarray
                            ) -> np.ndarray:
        if not (len(var1_samples.shape) == len(var2_samples.shape) == 2 and
                var1_samples.shape[0] == var2_samples.shape[0] and
                (var1_samples.shape[1] == self.var1.dim) and (var2_samples.shape[1] ==
                                                              self.var2.dim)):
            raise ValueError("Dimensionality of variable 1 or variable 2 is"
                             " wrong")
        num_samples = var1_samples.shape[0]
        noise_samples = self._noise_distribution.rvs(num_samples)
        res = np.sqrt(np.sum((var2_samples[:, self.var2.t_dim_indices] -
                              var1_samples[:, self.var1.t_dim_indices]) ** 2,
                             axis=1
                             )).reshape((num_samples, 1)) + noise_samples
        return res

    @property
    def sigma(self) -> float:
        return self._sigma

    def sample(self,
               var1: Union[np.ndarray, None] = None,
               var2: Union[np.ndarray, None] = None
               ) -> np.ndarray:
        """
        Generate samples with given samples
            When var1 samples are var2 samples are given, generate observation
                samples
            When var2 samples are given, generate var1 samples
            When var1 samples are given, generate var2 samples
        :param var1: samples of var1
        :param var2: samples of var2
        :return: generated samples
        """
        if var1 is None:  # var2 samples are given, wants samples of var1
            if var2 is None:
                raise ValueError("Samples of at least one variable must be "
                                 "specified")
            return self.sample_var1_from_var2(var2)
        elif var2 is None:  # var1 samples are given, wants samples of var2
            return self.sample_var2_from_var1(var1)
        else:  # var1 and var2 samples are given, wants samples of observation
            return self.sample_observations(var1, var2)

    def unif_to_sample(self, u: np.ndarray,
                       var1: Union[np.ndarray, None] = None,
                       var2: Union[np.ndarray, None] = None
                       ) -> np.ndarray:
        """
        Nested sampling necessity
        u.shape is (2,) for planar problem
        var1 or var2 can be either (2,) or (3,)
        the known sample has to be on SE2 currently
        """
        assert len(u) == 2
        u_for_dist = u[0]
        u_for_angle = u[1]
        normal_var = scistats.norm.ppf(u_for_dist)  # convert to standard normal
        dist_sample = (self._cov_sqrt * normal_var + self._observation)[0]
        angle_sample = (u_for_angle - 0.5) * _TWO_PI

        if var1 is None:  # var2 samples are given, wants samples of var1
            if var2 is None:
                raise ValueError("Samples of at least one variable must be "
                                 "specified")
            assert len(var2) == 3
            return var2[self.var2.t_dim_indices] + np.array([dist_sample * np.cos(angle_sample),
                                                             dist_sample * np.sin(angle_sample)])
        elif var2 is None:  # var1 samples are given, wants samples of var2
            assert len(var1) == 3
            return var1[self.var1.t_dim_indices] + np.array([dist_sample * np.cos(angle_sample),
                                                             dist_sample * np.sin(angle_sample)])
        else:  # var1 and var2 samples are given, wants samples of observation
            raise ValueError("Both of var1 and var2 are given.")

    def dvardu(self, top_var: Variable,
               top_arr: np.ndarray,
               bot_var: Variable,
               bot_arr: np.ndarray):
        """
        dtop_var/dbot_var, dtop_var/du
        """
        dtopdbot = np.zeros((len(top_arr), len(bot_arr)))
        dtopdbot[0, 0], dtopdbot[1, 1] = 1.0, 1.0
        dtopdu = np.zeros((len(top_arr), 2))
        vec = top_arr[top_var.t_dim_indices] - bot_arr[bot_var.t_dim_indices]
        distane = np.linalg.norm(vec)
        norm_dist = (distane - self.observation[0]) / self.sigma
        angle = np.arctan2(vec[1], vec[0])
        # r = sigma * ppf(u) + obs, dppf(u)/du = 1/p(x)
        dtopdrth = np.array([[np.cos(angle), -distane * np.sin(angle)],
                             [np.sin(angle), distane * np.cos(angle)]])
        drthdu = np.array([[self.sigma / scistats.norm.pdf(norm_dist), 0], [0, _TWO_PI]])
        dtopdu[top_var.t_dim_indices] = dtopdrth @ drthdu
        return dtopdbot, dtopdu

    def dvar1du(self, var1, var2):
        return self.dvardu(top_var=self.var1, top_arr=var1, bot_var=self.var2, bot_arr=var2)

    def dvar2du(self, var1, var2):
        return self.dvardu(bot_var=self.var1, bot_arr=var1, top_var=self.var2, top_arr=var2)

    def evaluate_loglike(self, x):
        """
        Nested sampling necessity
        x is (dim, ) where dim = var1.dim + var2.dim
        """
        var1_sample = x[0:self.var1.dim]
        var2_sample = x[self.var1.dim:]
        delta = np.linalg.norm(var1_sample[self.var1.t_dim_indices] -
                               var2_sample[self.var2.t_dim_indices]) - \
                self._observation[0]
        return -0.5 * (delta ** 2 / self._variance) + self._lnorm

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.log_pdf(x))

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        var1_sample = x[:, 0:self.var1.dim]
        var2_sample = x[:, self.var1.dim:]
        delta = np.linalg.norm(var1_sample[:, self.var1.t_dim_indices] -
                               var2_sample[:, self.var2.t_dim_indices], axis=1, keepdims=True) - \
                self._observation[0]
        return self._noise_distribution.log_pdf(delta)

    def grad_x_log_pdf(self, x: np.ndarray) -> np.ndarray:
        var1_sample = x[:, 0:self.var1.dim]
        var2_sample = x[:, self.var1.dim:]
        distance = np.linalg.norm(var1_sample[:, self.var1.t_dim_indices] -
                                  var2_sample[:, self.var2.t_dim_indices], axis=1, keepdims=True)
        diff = var1_sample[:, self.var1.t_dim_indices] - var2_sample[:, self.var2.t_dim_indices]
        delta = distance - self._observation[0]
        res = np.zeros_like(x)
        res[:, 0:self.var1.dim][:, self.var1.t_dim_indices] = diff
        res[:, self.var1.dim:][:, self.var2.t_dim_indices] = -diff
        low_dist_idx = np.where(distance < 1e-8)[0]
        others = np.array([idx for idx in range(x.shape[0]) if idx not in low_dist_idx])
        if len(low_dist_idx) > 0:
            res[low_dist_idx] = (res[low_dist_idx] / 1e-8) * self._noise_distribution.grad_x_log_pdf(delta[low_dist_idx]).reshape((-1, 1))
        if len(others) > 0:
            res[others] = (res[others] / distance[others]) * self._noise_distribution.grad_x_log_pdf(delta[others]).reshape((-1, 1))
        return res

    @property
    def is_gaussian(self):
        return False

    def get_lmk_samples(self, rbt_value: gtsam.Pose2, num_samples: int):
        noise_samples = self._noise_distribution.rvs(num_samples)
        dist_samples = (np.zeros((num_samples, 1)) + self._observation +
                        noise_samples)
        angle_samples = np.random.uniform(-np.pi, np.pi, num_samples).reshape(
            (num_samples, 1))
        return np.hstack((dist_samples * np.cos(angle_samples),
                          dist_samples * np.sin(angle_samples))) + [rbt_value.x(), rbt_value.y()]

    def sample_lmk_from_rbt(self, rbt_samples: np.ndarray) -> np.ndarray:
        """
        :param rbt_samples: n*3 array where n is the number of samples and 3 columns correspond to x, y ,theta
        :return: n*2 array
        """
        num_samples = rbt_samples.shape[0]
        noise_samples = self._noise_distribution.rvs(num_samples)
        dist_samples = (np.zeros((num_samples, 1)) + self._observation +
                        noise_samples)
        angle_samples = np.random.uniform(-np.pi, np.pi, num_samples).reshape(
            (num_samples, 1))
        return rbt_samples[:, self.var1.t_dim_indices] + \
               np.hstack((dist_samples * np.cos(angle_samples),
                          dist_samples * np.sin(angle_samples)))

    def samples2logpdf(self, rbt_samples, lmk_samples):
        """
        :param rbt_samples: n*3 array
        :param lmk_samples: n*2 array
        :return:
        """
        delta = np.linalg.norm(rbt_samples[:, self.var1.t_dim_indices] -
                               lmk_samples[:, self.var2.t_dim_indices], axis=1, keepdims=True) - \
                self._observation[0]
        return self._noise_distribution.log_pdf(delta)

    def samples2pdf(self, rbt_samples, lmk_samples):
        """
        :param rbt_samples: n*3 array
        :param lmk_samples: n*2 array
        :return:
        """
        return np.exp(self.samples2logpdf(rbt_samples, lmk_samples))

class SE2SE2RangeGaussianLikelihoodFactor(ExplicitLikelihoodFactor, BinaryFactor):
    """
    a general range factor for R2 and SE(2) variables
    """
    measurement_dim = 1
    measurement_type = R1Variable

    def __init__(self, var1: Variable, var2: Variable,
                 observation: Union[np.ndarray, float], sigma: float = 1.0,
                 ) -> None:
        """
        :param var1
        :param var2
        :param observation: observed distance from var1 to var2
        :param sigma: standard deviation of Gaussian distribution
        """
        super().__init__(vars=[var1, var2], log_likelihood=None)
        # self._unary_dim = 2
        self._observation = observation if isinstance(observation, np.ndarray) \
            else np.array([observation])
        self._noise_distribution = dist.GaussianDistribution(
            mu=np.zeros(1), sigma=np.array([[sigma ** 2]]))
        # this is for the observation variable
        # self._circular_dim_list = [False]
        self._sigma = sigma
        # this is for evaluating log-likelihood
        self._lnorm = -0.5 * np.log(_TWO_PI) - np.log(sigma)
        self._variance = sigma ** 2
        self._cov_sqrt = np.sqrt(self._variance)
        self._observation_var = type(self). \
            measurement_type(name="O" + var1.name + var2.name, variable_type=VariableType.Measurement)

        assert var1.t_dim_indices[0] == 0  and var1.t_dim_indices[1] == 1
        assert var2.t_dim_indices[0] == 0  and var2.t_dim_indices[1] == 1

    @property
    def observation_var(self):
        return self._observation_var

    @property
    def circular_dim_list(self):
        return self._observation_var.circular_dim_list

    @classmethod
    def construct_from_text(cls, line: str, variables: Iterable[Variable]
                            ) -> "SE2SE22RangeGaussianLikelihoodFactor":
        line = line.strip().split()
        name_to_var = {var.name: var for var in variables}
        if line[0] == cls.__name__:
            var1 = name_to_var[line[1]]
            var2 = name_to_var[line[2]]
            obs = float(line[3])
            sigma = float(line[4])
            factor = cls(var1=var1, var2=var2, observation=obs, sigma=sigma)
        else:
            raise ValueError("The factor name is incorrect")
        return factor

    def __str__(self) -> str:
        line = ["Factor", self.__class__.__name__,
                str(self.vars[0].name), str(self.vars[1].name),
                str(self.observation[0]), str(self.sigma)]
        return " ".join(line)

    @property
    def observation(self) -> np.ndarray:
        return self._observation

    def sample_var2_from_var1(self, var1_samples: np.ndarray) -> np.ndarray:
        if (len(var1_samples.shape) != 2 or var1_samples.shape[0] == 0 or
                var1_samples.shape[1] != self.var1.dim):
            raise ValueError("The dimensionality of variable 1 is wrong")
        num_samples = var1_samples.shape[0]
        noise_samples = self._noise_distribution.rvs(num_samples)
        dist_samples = (np.zeros((num_samples, 1)) + self._observation +
                        noise_samples)
        angle_samples = np.random.uniform(-np.pi, np.pi, num_samples).reshape(
            (num_samples, 1))
        var2_trans = var1_samples[:, self.var1.t_dim_indices] + \
               np.hstack((dist_samples * np.cos(angle_samples),
                          dist_samples * np.sin(angle_samples)))
        var2_heading = (np.random.random((var1_samples.shape[0],1)) - .5) * _TWO_PI
        var2_samples = np.hstack((var2_trans, var2_heading))
        return var2_samples

    def sample_var1_from_var2(self, var2_samples: np.ndarray) -> np.ndarray:
        if (len(var2_samples.shape) != 2 or var2_samples.shape[0] == 0 or
                var2_samples.shape[1] != self.var2.dim):
            raise ValueError("The dimensionality of variable 2 is wrong")
        num_samples = var2_samples.shape[0]

        # Generate noise samples in range
        noise_samples = self._noise_distribution.rvs(num_samples)
        dist_samples = (np.zeros((num_samples, 1)) + self._observation +
                        noise_samples)
        angle_samples = np.random.uniform(-np.pi, np.pi, num_samples).reshape(
            num_samples, 1)
        var1_trans = var2_samples[:, self.var2.t_dim_indices] + \
               np.hstack((dist_samples * np.cos(angle_samples),
                          dist_samples * np.sin(angle_samples)))
        var1_heading = np.random.random((var2_samples.shape[0],1)) * _TWO_PI - np.pi
        var1_samples = np.hstack((var1_trans, var1_heading))
        return var1_samples

    def sample_observations(self,
                            var1_samples: np.ndarray,
                            var2_samples: np.ndarray
                            ) -> np.ndarray:
        if not (len(var1_samples.shape) == len(var2_samples.shape) == 2 and
                var1_samples.shape[0] == var2_samples.shape[0] and
                (var1_samples.shape[1] == self.var1.dim) and (var2_samples.shape[1] ==
                                                              self.var2.dim)):
            raise ValueError("Dimensionality of variable 1 or variable 2 is"
                             " wrong")
        num_samples = var1_samples.shape[0]
        noise_samples = self._noise_distribution.rvs(num_samples)
        res = np.sqrt(np.sum((var2_samples[:, self.var2.t_dim_indices] -
                              var1_samples[:, self.var1.t_dim_indices]) ** 2,
                             axis=1
                             )).reshape((num_samples, 1)) + noise_samples
        return res

    @property
    def sigma(self) -> float:
        return self._sigma

    def sample(self,
               var1: Union[np.ndarray, None] = None,
               var2: Union[np.ndarray, None] = None
               ) -> np.ndarray:
        """
        Generate samples with given samples
            When var1 samples are var2 samples are given, generate observation
                samples
            When var2 samples are given, generate var1 samples
            When var1 samples are given, generate var2 samples
        :param var1: samples of var1
        :param var2: samples of var2
        :return: generated samples
        """
        if var1 is None:  # var2 samples are given, wants samples of var1
            if var2 is None:
                raise ValueError("Samples of at least one variable must be "
                                 "specified")
            return self.sample_var1_from_var2(var2)
        elif var2 is None:  # var1 samples are given, wants samples of var2
            return self.sample_var2_from_var1(var1)
        else:  # var1 and var2 samples are given, wants samples of observation
            return self.sample_observations(var1, var2)

    def unif_to_sample(self, u: np.ndarray,
                       var1: Union[np.ndarray, None] = None,
                       var2: Union[np.ndarray, None] = None
                       ) -> np.ndarray:
        """
        Nested sampling necessity
        u.shape is (2,) for planar problem
        var1 or var2 can be either (2,) or (3,)
        the known sample has to be on SE2 currently
        """
        assert len(u) == 3
        u_for_dist = u[0]
        u_for_angle = u[1]
        u_for_rbt_heading = u[2]
        normal_var = scistats.norm.ppf(u_for_dist)  # convert to standard normal
        dist_sample = (self._cov_sqrt * normal_var + self._observation)[0]
        angle_sample = (u_for_angle - 0.5) * _TWO_PI
        heading_sample = (u_for_rbt_heading - .5) * _TWO_PI

        if var1 is None:  # var2 samples are given, wants samples of var1
            if var2 is None:
                raise ValueError("Samples of at least one variable must be "
                                 "specified")
            assert len(var2) == 3
            return var2[:] + np.array([dist_sample * np.cos(angle_sample),
                                       dist_sample * np.sin(angle_sample),
                                       heading_sample])
        elif var2 is None:  # var1 samples are given, wants samples of var2
            assert len(var1) == 3
            return var1[:] + np.array([dist_sample * np.cos(angle_sample),
                                       dist_sample * np.sin(angle_sample),
                                       heading_sample])
        else:  # var1 and var2 samples are given, wants samples of observation
            raise ValueError("Both of var1 and var2 are given.")

    #TODO: update this from SE2R2 to SE2SE2
    # def dvardu(self, top_var: Variable,
    #            top_arr: np.ndarray,
    #            bot_var: Variable,
    #            bot_arr: np.ndarray):
    #     """
    #     dtop_var/dbot_var, dtop_var/du
    #     """
    #     dtopdbot = np.zeros((len(top_arr), len(bot_arr)))
    #     dtopdbot[0, 0], dtopdbot[1, 1] = 1.0, 1.0
    #     dtopdu = np.zeros((len(top_arr), 2))
    #     vec = top_arr[top_var.t_dim_indices] - bot_arr[bot_var.t_dim_indices]
    #     distane = np.linalg.norm(vec)
    #     norm_dist = (distane - self.observation[0]) / self.sigma
    #     angle = np.arctan2(vec[1], vec[0])
    #     # r = sigma * ppf(u) + obs, dppf(u)/du = 1/p(x)
    #     dtopdrth = np.array([[np.cos(angle), -distane * np.sin(angle)],
    #                          [np.sin(angle), distane * np.cos(angle)]])
    #     drthdu = np.array([[self.sigma / scistats.norm.pdf(norm_dist), 0], [0, _TWO_PI]])
    #     dtopdu[top_var.t_dim_indices] = dtopdrth @ drthdu
    #     return dtopdbot, dtopdu

    # def dvar1du(self, var1, var2):
    #     return self.dvardu(top_var=self.var1, top_arr=var1, bot_var=self.var2, bot_arr=var2)
    #
    # def dvar2du(self, var1, var2):
    #     return self.dvardu(bot_var=self.var1, bot_arr=var1, top_var=self.var2, top_arr=var2)

    def evaluate_loglike(self, x):
        """
        Nested sampling necessity
        x is (dim, ) where dim = var1.dim + var2.dim
        """
        var1_sample = x[0:self.var1.dim]
        var2_sample = x[self.var1.dim:]
        delta = np.linalg.norm(var1_sample[self.var1.t_dim_indices] -
                               var2_sample[self.var2.t_dim_indices]) - \
                self._observation[0]
        return -0.5 * (delta ** 2 / self._variance) + self._lnorm

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.log_pdf(x))

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        var1_sample = x[:, 0:self.var1.dim]
        var2_sample = x[:, self.var1.dim:]
        delta = np.linalg.norm(var1_sample[:, self.var1.t_dim_indices] -
                               var2_sample[:, self.var2.t_dim_indices], axis=1, keepdims=True) - \
                self._observation[0]
        return self._noise_distribution.log_pdf(delta)

    def grad_x_log_pdf(self, x: np.ndarray) -> np.ndarray:
        var1_sample = x[:, 0:self.var1.dim]
        var2_sample = x[:, self.var1.dim:]
        distance = np.linalg.norm(var1_sample[:, self.var1.t_dim_indices] -
                                  var2_sample[:, self.var2.t_dim_indices], axis=1, keepdims=True)
        diff = var1_sample[:, self.var1.t_dim_indices] - var2_sample[:, self.var2.t_dim_indices]
        delta = distance - self._observation[0]
        res = np.zeros_like(x)
        res[:, 0:self.var1.dim][:, self.var1.t_dim_indices] = diff
        res[:, self.var1.dim:][:, self.var2.t_dim_indices] = -diff
        low_dist_idx = np.where(distance < 1e-8)[0]
        others = np.array([idx for idx in range(x.shape[0]) if idx not in low_dist_idx])
        if len(low_dist_idx) > 0:
            res[low_dist_idx] = (res[low_dist_idx] / 1e-8) * self._noise_distribution.grad_x_log_pdf(delta[low_dist_idx]).reshape((-1, 1))
        if len(others) > 0:
            res[others] = (res[others] / distance[others]) * self._noise_distribution.grad_x_log_pdf(delta[others]).reshape((-1, 1))
        return res

    @property
    def is_gaussian(self):
        return False

class KWayFactor(Factor):
    @property
    def vars(self) -> List[Variable]:
        raise NotImplementedError

    @property
    def root_var(self) -> Variable:
        raise NotImplementedError

    @property
    def child_vars(self) -> List[Variable]:
        raise NotImplementedError


class FactorMixture:
    def __init__(self,
                 weights: np.ndarray,
                 factors: List[Factor]):
        self.weights = weights
        self.components = factors


class UnaryFactorMixture(UnaryFactor):
    """
    This class defines a unary factor which is a mixture of multiple unary factors.

    Parameters
    __________
    var: Variable
    weights: np.ndarray
    factors: List[Factor]
    """
    def __init__(self,
                 var: Variable,
                 weights: np.ndarray,
                 factors: List[Factor]):
        assert all(weights > 0) and len(weights) == len(factors)
        self.components = factors
        self.weights = weights
        self._vars = [var]
        self.cum_weights = np.cumsum(self.weights)

    @property
    def var(self):
        return self._vars[0]

    @property
    def vars(self):
        return self._vars


    def pdf(self, x: np.ndarray) -> np.ndarray:
        arr = np.zeros((x.shape[0], len(self.components)))
        for i, comp in enumerate(self.components):
            arr[:, i] = comp.pdf(x) * self.weights[i]
        return np.sum(arr, axis=1)

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        return np.log(self.pdf(x))

    def grad_x_log_pdf(self, x: np.ndarray) -> np.ndarray:
        wp_arr = np.zeros((x.shape[0], len(self.components)))
        res_arr = np.zeros_like(x)
        for i, comp in enumerate(self.components):
            wp_arr[:, i] = comp.pdf(x) * self.weights[i]
            # multiply w*p to each row of grad_x_log_p
            res_arr += wp_arr[:, i:i + 1] * comp.grad_x_log_pdf(x[:, self.comp2idx[comp]])
        # warning: dividing by zero can happen as p can be very small
        return res_arr / np.sum(wp_arr, axis=1, keepdims=True)

    # sample observations given samples of vars
    def sample(self, n: int) -> np.ndarray:
        comp_n = np.random.multinomial(n, self.weights)
        arr = np.zeros((n, self.dim))
        init_idx = 0
        for i, comp in enumerate(self.components):
            end_idx = init_idx + comp_n[i]
            arr[init_idx: end_idx, :] = \
                comp.sample(comp_n[i])
            init_idx = end_idx
        return arr

class BinaryFactorMixture(LikelihoodFactor):
    """
    This class defines a likelihood factor with ambiguous data association.

    Parameters
    ----------
    observer_var: a factor graph variable acquiring an observaton
    observed_vars: a list of factor graph variables potentially being measured to
    weights: a list of non-negative floats as long as observed_vars
    observation: the raw measurement
    covariance: a positive definite matrix
    factor_class: the factor type for each pair of associated variables
    """

    # TODO: add method to infer posterior of data associatin hypotheses
    def __init__(self,
                 observer_var: Variable,
                 observed_vars: List[Variable],
                 weights: np.ndarray,
                 binary_factor_class,
                 obs_arr: List,
                 sigma_arr: List
                 ):
        assert all(weights > 0) and len(weights) == len(obs_arr) == len(sigma_arr) == len(observed_vars)
        self.observer_var = observer_var

        seen = set()
        seen_add = seen.add
        self.observed_vars = [x for x in observed_vars if not (x in seen or seen_add(x))]
        self._vars = [observer_var] + self.observed_vars
        self.weights = weights / sum(weights)
        self.observations = obs_arr
        self.sigmas = sigma_arr
        self.components = [binary_factor_class(observer_var,
                                               var,
                                               obs_arr[i],
                                               sigma_arr[i]) for i, var in enumerate(observed_vars)]
        self.var2idx = {}
        init_idx = 0
        for var in self._vars:
            end_idx = var.dim + init_idx
            self.var2idx[var] = np.arange(init_idx, end_idx)
            init_idx = end_idx
        self.comp2idx = {}
        for comp in self.components:
            self.comp2idx[comp] = np.concatenate((self.var2idx[comp.var1], self.var2idx[comp.var2]))
        # will be frequently used to choose a component for a hypercube sample
        self.cum_weights = np.cumsum(self.weights)

    @property
    def vars(self):
        return self._vars

    @property
    def observation_var(self):
        return self.components[0].observation_var

    @property
    def measurement_dim(self):
        return self.observation_var.dim

    @property
    def is_gaussian(self):
        return False

    # TODO: remove it and use log_pdf for nested sampling
    def evaluate_loglike(self, x):
        """
        Nested sampling necessity
        x is (dim, ) where dim = sum([var.dim for var in self.vars])
        """

        log_cmp = np.array([comp.evaluate_loglike(x[self.comp2idx[comp]]) + np.log(self.weights[i])
                        for i, comp in enumerate(self.components)])
        largest_idx, sec_l_idx = log_cmp.argsort()[-2:][::-1]
        if log_cmp[largest_idx] - log_cmp[sec_l_idx] > 5.0:
            # some approx.
            return log_cmp[largest_idx]
        else:
            # arr = np.array([np.exp(comp.evaluate_loglike(x[self.comp2idx[comp]])) * self.weights[i]
            #                 for i, comp in enumerate(self.components)])
            return np.log(sum(np.exp(log_cmp)))

    def pdf(self, x: np.ndarray) -> np.ndarray:
        arr = np.zeros((x.shape[0], len(self.components)))
        for i, comp in enumerate(self.components):
            arr[:, i] = comp.pdf(x[:, self.comp2idx[comp]]) * self.weights[i]
        return np.sum(arr, axis=1)

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        return np.log(self.pdf(x))

    def grad_x_log_pdf(self, x: np.ndarray) -> np.ndarray:
        wp_arr = np.zeros((x.shape[0], len(self.components)))
        res_arr = np.zeros_like(x)
        for i, comp in enumerate(self.components):
            wp_arr[:, i] = comp.pdf(x[:, self.comp2idx[comp]]) * self.weights[i]
            # multiply w*p to each row of grad_x_log_p
            res_arr[:, self.comp2idx[comp]] += wp_arr[:, i:i + 1] * comp.grad_x_log_pdf(x[:, self.comp2idx[comp]])
        # warning: dividing by zero can happen as p can be very small
        return res_arr / np.sum(wp_arr, axis=1, keepdims=True)

    # sample observations given samples of vars
    def sample_observations(self, var_samples: Dict[Variable, np.ndarray]) -> np.ndarray:
        n = var_samples[self.observer_var].shape[0]
        comp_n = np.random.multinomial(n, self.weights)
        arr = np.zeros((n, self.measurement_dim))
        init_idx = 0
        for i, comp in enumerate(self.components):
            end_idx = init_idx + comp_n[i]
            arr[init_idx: end_idx, :] = \
                comp.sample(var1=var_samples[comp.var1][init_idx: end_idx, :],
                            var2=var_samples[comp.var2][init_idx: end_idx, :])
            init_idx = end_idx
        return arr

    def posterior_weights(self, var2x: Dict[Variable, np.ndarray]):
        """
        This methods use posterior samples to re-evaluate the weights of hypotheses

        params:
        ---------
        x: variable to samples

        return ((len(weights),)
        """
        x = np.concatenate([var2x[var] for var in self.vars], axis=1)
        hypo_likelihoods = np.array([comp.pdf(x[:, self.comp2idx[comp]]) * self.weights[i]
                                     for i, comp in enumerate(self.components)])
        hypo_sum = hypo_likelihoods.sum(axis=0)
        mask = np.ones(hypo_likelihoods.shape[1], np.bool)
        zero_columns = np.where(hypo_sum == .0)[0]
        mask[zero_columns] = False
        hypo_weights = np.zeros((len(self.components), x.shape[0]))
        hypo_weights[:, mask] = hypo_likelihoods[:, mask] / hypo_sum[mask]
        hypo_weights[:, zero_columns] = .5
        hypo_weight = hypo_weights.sum(axis=1) / hypo_weights.sum()
        return hypo_weight


class BinaryMixtureWithSameData(BinaryFactorMixture):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def observation(self) -> np.ndarray:
        return self.components[0].observation


class AmbiguousDataAssociationFactor(BinaryMixtureWithSameData, KWayFactor):
    """
    This class defines a likelihood factor with ambiguous data association.

    Parameters
    ----------
    observer_var: a factor graph variable acquiring an observaton
    observed_vars: a list of factor graph variables potentially being measured to
    weights: a list of non-negative floats as long as observed_vars
    observation: the raw measurement
    covariance: a positive definite matrix
    factor_class: the factor type for each pair of associated variables
    """

    def __init__(self,
                 observer_var: Variable,
                 observed_vars: List[Variable],
                 weights: np.ndarray,
                 binary_factor_class,
                 observation,
                 sigma
                 ):
        k = len(observed_vars)
        assert k == len(weights)
        super().__init__(observer_var, observed_vars, weights,
                         binary_factor_class, [observation] * k, [sigma] * k)

    @property
    def root_var(self) -> Variable:
        return self.observer_var

    @property
    def child_vars(self) -> List[Variable]:
        return self.observed_vars

    @classmethod
    def construct_from_text(cls, line: str, variables: Iterable[Variable]
                            ) -> "AmbiguousDataAssociationFactor":
        line = line.strip().split()
        name_to_var = {var.name: var for var in variables}
        if line[0] == cls.__name__:
            observer_idx = line.index('Observer') + 1
            observed_idx = line.index('Observed') + 1
            weight_idx = line.index('Weights') + 1
            factor_idx = line.index('Binary') + 1
            observation_idx = line.index('Observation') + 1
            sigma_idx = line.index('Sigma') + 1  # std for scalar observation and cov for vector observation

            observer_var = name_to_var[line[observer_idx]]
            observed_vars = [name_to_var[line[idx]] for idx in range(observed_idx, weight_idx - 1)]
            weights = np.array(line[weight_idx:factor_idx - 1]).astype(float)
            assert len(weights) == len(observed_vars)
            binary_factor = globals()[line[factor_idx]]

            obs_len = sigma_idx - observation_idx - 1

            if obs_len == 1:
                observation = float(line[observation_idx])
                sigma = float(line[sigma_idx])
            else:
                observation = np.array(line[observation_idx:sigma_idx - 1]).astype(float)
                sigma = np.array(line[sigma_idx:sigma_idx + obs_len * obs_len]).astype(float).reshape(
                    (obs_len, obs_len))
            factor = cls(observer_var, observed_vars, weights, binary_factor, observation, sigma)
        else:
            raise ValueError("The factor name is incorrect")
        return factor

    def sample_observer(self, var2sample: Dict[Variable, np.ndarray]):
        n = var2sample[self.observed_vars[0]].shape[0]
        comp_n = np.random.multinomial(n, self.weights)
        arr = np.zeros((n, self.observer_var.dim))
        init_idx = 0
        for i, comp in enumerate(self.components):
            end_idx = init_idx + comp_n[i]
            if comp.var1 == self.observer_var:
                arr[init_idx: end_idx, :] = \
                    comp.sample(var2=var2sample[comp.var2][init_idx: end_idx, :])
            elif comp.var2 == self.observer_var:
                arr[init_idx: end_idx, :] = \
                    comp.sample(var1=var2sample[comp.var1][init_idx: end_idx, :])
            else:
                raise ValueError("None of the vars of component matches the observer var.")
            init_idx = end_idx
        return arr

    def __str__(self) -> str:
        line = ["Factor", self.__class__.__name__, "Observer", str(self.observer_var.name), "Observed"]
        line += [str(var.name) for var in self.observed_vars]
        line += ["Weights"]
        line += [str(w) for w in self.weights]
        line += ["Binary", self.components[0].__class__.__name__]
        line += ["Observation"]
        obs = self.observation
        if isinstance(obs, (np.ndarray, List)):
            line += [str(v) for v in obs]
        elif isinstance(obs, float):
            line += [str(obs)]
        line += ["Sigma"]
        if hasattr(self.components[0], 'sigma'):
            sigma = self.components[0].sigma
            line += [str(sigma)]
        elif hasattr(self.components[0], 'covariance'):
            cov = self.components[0].covariane
            line += [str(v) for v in cov.flatten()]
        return " ".join(line)


class BinaryFactorWithNullHypo(BinaryMixtureWithSameData, BinaryFactor):
    def __init__(self,
                 var1: Variable,
                 var2: Variable,
                 weights: np.ndarray,
                 binary_factor_class,
                 observation,
                 sigma,
                 null_sigma_scale=10.0
                 ):
        assert len(weights) == 2
        self.null_sigma_scale = null_sigma_scale
        super().__init__(var1, [var2, var2], weights,
                         binary_factor_class, [observation] * 2, [sigma, sigma * null_sigma_scale])

    def sample(self,
               var1: Union[np.ndarray, None] = None,
               var2: Union[np.ndarray, None] = None
               ) -> np.ndarray:
        """
        Generate samples with given samples
            When var1 samples are var2 samples are given, generate observation
                samples
            When var2 samples are given, generate var1 samples
            When var1 samples are given, generate var2 samples
        :param var1: samples of var1
        :param var2: samples of var2
        :return: generated samples
        """
        if var1 is None:  # var2 samples are given, wants samples of var1
            if var2 is None:
                raise ValueError("Samples of at least one variable must be "
                                 "specified")
            return self.sample_var1_from_var2(var2)
        elif var2 is None:  # var1 samples are given, wants samples of var2
            return self.sample_var2_from_var1(var1)
        else:  # var1 and var2 samples are given, wants samples of observation
            return self.sample_observations(var1, var2)

    def sample_var1_from_var2(self, x):
        n = x.shape[0]
        comp_n = np.random.multinomial(n, self.weights)
        arr = np.zeros((n, self.var1.dim))
        init_idx = 0
        for i, comp in enumerate(self.components):
            end_idx = init_idx + comp_n[i]
            arr[init_idx: end_idx, :] = \
                comp.sample(var2=x[init_idx: end_idx, :])
            init_idx = end_idx
        return arr

    def sample_var2_from_var1(self, x):
        n = x.shape[0]
        comp_n = np.random.multinomial(n, self.weights)
        arr = np.zeros((n, self.var2.dim))
        init_idx = 0
        for i, comp in enumerate(self.components):
            end_idx = init_idx + comp_n[i]
            arr[init_idx: end_idx, :] = \
                comp.sample(var1=x[init_idx: end_idx, :])
            init_idx = end_idx
        return arr

    def sample_observations(self, var1: np.ndarray, var2: np.ndarray) -> np.ndarray:
        n = var1.shape[0]
        comp_n = np.random.multinomial(n, self.weights)
        arr = np.zeros((n, self.measurement_dim))
        init_idx = 0
        for i, comp in enumerate(self.components):
            end_idx = init_idx + comp_n[i]
            arr[init_idx: end_idx, :] = \
                comp.sample(var1=var1[init_idx: end_idx, :],
                            var2=var2[init_idx: end_idx, :])
            init_idx = end_idx
        return arr

    def unif_to_sample(self, u=np.ndarray, var1: np.ndarray = None, var2: np.ndarray = None
                       ) -> np.ndarray:
        """
        Generate samples of observed vars with samples of observer var and
        uniform distribution samples being transformed to noisy observations.
            u is a (dim of unique observed vars, ) numpy array sampled from a uniform hypercube
            When var1 samples are given, generate var2 samples
        :param var1
        :param var2
        :return: generated samples
        """
        # use the first dim of u sample to determine the category or component of the mixture
        comp_idx = np.where(u[0] < self.cum_weights)[0][0]
        scaled_u = np.array(u)
        if comp_idx == 0:
            int_offset = 0.0
        else:
            int_offset = self.cum_weights[comp_idx - 1]
        scaled_u[0] = (scaled_u[0] - int_offset) / self.weights[comp_idx]

        if var1 is None:  # var2 samples are given, wants samples of var1
            if var2 is None:
                raise ValueError("Samples of at least one variable must be "
                                 "specified")
            return self.components[comp_idx].unif_to_sample(scaled_u, var2=var2)
        elif var2 is None:  # var1 samples are given, wants samples of var2
            return self.components[comp_idx].unif_to_sample(scaled_u, var1=var1)
        else:  # var1 and var2 samples are given, wants samples of observation
            raise ValueError("Both of var1 and var2 are given.")

    @classmethod
    def construct_from_text(cls, line: str, variables: Iterable[Variable]
                            ) -> "BinaryFactorWithNullHypo":
        line = line.strip().split()
        name_to_var = {var.name: var for var in variables}
        if line[0] == cls.__name__:
            observer_idx = line.index('Observer') + 1
            observed_idx = line.index('Observed') + 1
            weight_idx = line.index('Weights') + 1
            factor_idx = line.index('Binary') + 1
            observation_idx = line.index('Observation') + 1
            sigma_idx = line.index('Sigma') + 1  # std for scalar observation and cov for vector observation
            null_sigma_idx = line.index(
                'NullSigmaScale') + 1  # std for scalar observation and cov for vector observation

            observer_var = name_to_var[line[observer_idx]]
            observed_var = name_to_var[line[observed_idx]]
            weights = np.array(line[weight_idx:factor_idx - 1]).astype(float)
            binary_factor = globals()[line[factor_idx]]

            obs_len = sigma_idx - observation_idx - 1

            if obs_len == 1:
                observation = float(line[observation_idx])
                sigma = float(line[sigma_idx])
            else:
                observation = np.array(line[observation_idx:sigma_idx - 1]).astype(float)
                sigma = np.array(line[sigma_idx:sigma_idx + obs_len * obs_len]).astype(float).reshape(
                    (obs_len, obs_len))
            factor = cls(observer_var, observed_var, weights, binary_factor, observation, sigma,
                         float(line[null_sigma_idx]))
        else:
            raise ValueError("The factor name is incorrect")
        return factor

    def __str__(self) -> str:
        line = ["Factor", self.__class__.__name__, "Observer", str(self.observer_var.name), "Observed"]
        line += [str(var.name) for var in self.observed_vars]
        line += ["Weights"]
        line += [str(w) for w in self.weights]
        line += ["Binary", self.components[0].__class__.__name__]
        line += ["Observation"]
        obs = self.observation
        if isinstance(obs, (np.ndarray, List)):
            line += [str(v) for v in obs]
        elif isinstance(obs, float):
            line += [str(obs)]
        line += ["Sigma"]
        if hasattr(self.components[0], 'sigma'):
            sigma = self.components[0].sigma
            line += [str(sigma)]
        elif hasattr(self.components[0], 'covariance'):
            cov = self.components[0].covariane
            line += [str(v) for v in cov.flatten()]

        line += ["NullSigmaScale", str(self.null_sigma_scale)]
        return " ".join(line)
