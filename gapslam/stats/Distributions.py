from abc import ABCMeta
import numpy as np
from typing import List
import TransportMaps.Distributions as dist
import TransportMaps.Likelihoods as like
from utils.LinAlg import is_spd


class Distribution(metaclass=ABCMeta):
    @property
    def dim(self) -> int:
        raise NotImplementedError

    def rvs(self, num_samples: int) -> np.ndarray:
        """
        Generate samples from the distribution
        :return: samples
        :rtype: 2-dimensional numpy array
            each row is a sample; each column is a dimension
        """
        raise NotImplementedError

    def pdf(self, location: np.ndarray) -> np.ndarray:
        """
        Compute PDF at given locations
        :param location: location at which PDFs are evaluated
            each row is a location; each column is a dimension
        :return: PDFs at given locations
        :rtype: a 1-dimensional numpy array
        """
        raise NotImplementedError

    def log_pdf(self, location: np.ndarray) -> np.ndarray:
        """
        Compute log PDF at given locations
        :param location: location at which log PDFs are evaluated
            each row is a location; each column is a dimension
        :return: log PDFs at given locations
        :rtype: a 1-dimensional numpy array
        """
        raise NotImplementedError

    def grad_x_log_pdf(self, location: np.ndarray) -> np.ndarray:
        """
        Compute gradients of log PDF at given locations
        :param location: location at which gradients are evaluated
            each row is a location; each column is a dimension
        :return: PDFs at given locations
        :rtype: a 2-dimensional numpy array
            each row is a location; each column is a dimension
        """
        raise NotImplementedError


class GaussianDistribution(Distribution, metaclass=ABCMeta):
    def __init__(self, mu: np.ndarray, sigma: np.ndarray = None,
                 precision: np.ndarray = None):
        if len(mu.shape) != 1:
            raise ValueError("Dimensionality of mu is incorrect")
        self._mu = mu
        if sigma is not None:
            if sigma.shape != (self.dim, self.dim):
                raise ValueError("Dimensionality of sigma is incorrect")
            if not is_spd(sigma):
                raise ValueError("sigma must be symmetric positive definite")
            self._sigma = sigma
            self._precision = np.linalg.inv(self._sigma)
        else:
            if precision.shape != (self.dim, self.dim):
                raise ValueError("Dimensionality of precision matrix is "
                                 "incorrect")
            if not is_spd(precision):
                raise ValueError("sigma must be symmetric positive definite")
            self._precision = precision
            self._sigma = np.linalg.inv(self._precision)
        self._det_sigma = np.linalg.det(self._sigma)
        self._normalizing_constant = 1.0 / np.sqrt(
            (2.0 * np.pi) ** self.dim * self._det_sigma)
        self._sigma_chol = np.linalg.cholesky(self._sigma)
        self._precision_chol = np.linalg.cholesky(self._precision)

    @property
    def dim(self) -> int:
        return self._mu.shape[0]

    @property
    def mean(self) -> np.ndarray:
        return self._mu

    @property
    def covariance(self) -> np.ndarray:
        return self._sigma

    @property
    def precision(self) -> np.ndarray:
        return self._precision

    def rvs(self, num_samples: int) -> np.ndarray:
        samples = np.random.multivariate_normal(mean=self.mean,
                                                cov=self.covariance,
                                                size=num_samples)
        return samples

    # def log_pdf(self, locations: np.ndarray) -> np.ndarray:
    #     if len(locations.shape) != 2 or locations.shape[1] != self.dim:
    #         raise ValueError("Dimensionality of locations is incorrect")
    #     num_samples = locations.shape[0]
    #     b = locations.reshape(()) - self._mu.reshape((1, self.dim))
    #     return (np.log(self._normalizing_constant) -
    #             0.5 * b @ self._precision @ b.T)


class GaussianRangeDistribution(Distribution, metaclass=ABCMeta):
    def __init__(self, center: np.ndarray, mu: float, sigma: float) -> None:
        if len(center.shape) != 1:
            raise ValueError("The center has incorrect dimensionality")
        self._center = center
        self._mu = mu
        self._sigma = sigma
        self._dim = self._center.shape[0]
        self._noise_distribution = GaussianDistribution(mu=np.array([0]),
                                                        sigma=np.array([
                                                            [sigma]]))

    def rvs(self, num_samples: int) -> np.ndarray:
        noises = self._noise_distribution.rvs(num_samples)
        distances = self._mu + noises
        angle = np.random.uniform(-np.pi, np.pi, (num_samples, 1))
        return np.broadcast_to(self._center, (1, 2)) + np.hstack((
            distances * np.cos(angle), distances * np.sin(angle)))

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def center(self) -> np.ndarray:
        return self._center

    @property
    def mean(self) -> float:
        return self._mu

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def covariance(self) -> float:
        return self._sigma ** 2


class GaussianMixtureDistribution(dist.Distribution, metaclass=ABCMeta):
    def __init__(self, weights: List[float], means: List[np.ndarray],
                 sigmas: List[np.ndarray] = None,
                 precisions: List[np.ndarray] = None) -> None:
        # TODO: change assert statement to raise statement, add docstring
        self._num_components = len(weights)
        assert self._num_components == len(means) == len(sigmas) > 0
        assert all([w > 0 for w in weights])
        assert np.isclose(np.sum(weights), 1.0)
        self._dim = means[0].shape[0]
        super(GaussianMixtureDistribution, self).__init__(self._dim)
        for mean in means:
            assert len(mean.shape) == 1 and mean.shape[0] == self._dim
        for cov in sigmas:
            assert (len(cov.shape) == 2 and is_spd(cov))
        self._weights = weights
        self._means = means
        self._covs = sigmas
        self._components = []
        if sigmas:
            for comp in range(self._num_components):
                self._components.append(dist.GaussianDistribution(
                    mu=means[comp], sigma=sigmas[comp]))
        elif precisions:
            for comp in range(self._num_components):
                self._components.append(dist.GaussianDistribution(
                    mu=means[comp], precision=precisions[comp]))
        else:
            raise ValueError("Either precision or sigma should be"
                             " specified")

    def rvs(self, m: int, *args, **kwargs) -> np.ndarray:
        """
        TODO: Code optimization
        :param m: The number of samples to be generated
        :type: int
        :param args:
        :param kwargs:
        :return: Random samples
        :rtype: numpy.ndarray with shape (m, dim)
        :raise ValueError: if the input m is negative
        """
        if m < 0:
            raise ValueError("The number of samples must be non-negative")
        samples = np.zeros((m, self._dim))
        for i in range(m):
            comp = np.random.choice(
                range(self._num_components), p=self._weights)
            samples[[i], :] = self._components[comp].rvs(1)
        return samples

    def quadrature(self, qtype: int = 0, qparams: int = 100, *args, **kwargs):
        """
        :param qtype: 0 for Monte Carlo
        :param qparams: number of samples
        :param args
        :param kwargs
        :return: num_samples X dim np.ndarray
        """
        return self.rvs(qparams), np.ones(qparams) / float(qparams)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def num_components(self) -> int:
        return self._num_components

    @property
    def weights(self) -> List[float]:
        return self._weights

    @property
    def means(self) -> List[np.ndarray]:
        return self._means

    @property
    def sigmas(self) -> List[np.ndarray]:
        return self._covs

    def component(self, index: int) -> dist.GaussianDistribution:
        """
        Retrieve the index-th Gaussian distribution component
        :param index: the index of the component
        :type: int
        :return: The Gaussian component
        :rtype: TransportMaps.Distributions.GaussianDistribution
        :raise IndexError: if the the given index is invalid
        """
        if 0 <= index < self._num_components:
            return self._components[index]
        else:
            raise IndexError("The index is out of bound")

    def pdf(self, x: np.ndarray, params=None,
            idxs_slice=slice(None, None, None), **kwargs) -> np.ndarray:
        """
        Evaluate probability densities at given locations
        :param x: Locations to evaluate the pdf's
        :type: numpy.ndarray with shape (num_locs, dim)
        :param params:
        :param idxs_slice:
        :param kwargs:
        :return: The densities
        :rtype: numpy.ndarray with shape (num_locs)
        :raise ValueError: if the dimensionality of locations to evaluate
            the probability densities is incorrect
        """
        if len(x.shape) != 2 or x.shape[1] != self._dim:
            raise ValueError("The dimensionality of the locations is incorrect")
        num_locs = x.shape[0]
        densities = np.zeros(num_locs)
        for comp in range(self._num_components):
            densities += self._weights[comp] * dist.GaussianDistribution(
                mu=self._means[comp], sigma=self._covs[comp]).pdf(x)
        return densities

    def log_pdf(self, x: np.ndarray, params=None,
                idxs_slice=slice(None, None, None), **kwargs) -> np.ndarray:
        """
        Evaluate the logarithm of probability densities at given locations
        :param x: Locations to evaluate the pdf's
        :type: numpy.ndarray with shape (num_locs, dim)
        :param params:
        :param idxs_slice:
        :param kwargs:
        :return: The logarithm of densities
        :rtype: numpy.ndarray with shape (num_locs)
        """
        if self._components != 2:
            return np.log(self.pdf(x))
        pdf = self.pdf(x)
        indices = pdf == 0
        log_pdf = np.log(pdf)
        log_pdf[indices] = np.max(np.vstack((self._components[0].log_pdf(x),
                                             self._components[1].log_pdf(x))),
                                  axis=0)
        return log_pdf

    def grad_x_log_pdf(self, x: np.ndarray, params=None,
                       idxs_slice=slice(None, None, None),
                       **kwargs) -> np.ndarray:
        """
        # TODO: include multi-dimensional support
        # TODO: improve efficiency and numerical stability
        Evaluate the gradient of log(pdf) at given locations
        :param x: Locations to evaluate the pdf's
        :type: numpy.ndarray with shape (num_locs, dim)
        :param params:
        :param idxs_slice:
        :param kwargs:
        :return: The logarithm of densities
        :rtype: numpy.ndarray with shape (num_locs, dim)
        :raise ValueError: if the dimensionality of input is incorrect
        """
        if len(x.shape) != 2 or x.shape[1] != self._dim:
            raise ValueError("The dimensionality of the locations is incorrect")
        num_locs = x.shape[0]
        grad = np.zeros((num_locs, self._dim))
        det_sigmas = np.array([component.det_sigma
                               for component in self._components])
        for i in range(num_locs):
            loc = x[[i], :]
            displacements = [loc - component.mu
                             for component in self._components]
            exponents = np.array([-0.5 * float(displacements[c]
                                               @ self._components[c].inv_sigma
                                               @ displacements[c].T) for c in
                                  range(self._num_components)])
            max_exponent = np.max(exponents)
            exponents -= max_exponent
            pdfs = (np.exp(exponents) * np.array(self._weights)
                    / np.sqrt((2.0 * np.pi) ** self._dim * det_sigmas))
            denominator = np.sum(pdfs)
            factors = np.array([(-displacements[c]
                                 @ self._components[c].inv_sigma.T).
                               reshape(-1) for c in
                                range(self._num_components)])
            numerator = np.zeros((1, self._dim))
            for c in range(self._num_components):
                numerator += pdfs[c] * factors[c]
            grad[[i], :] = numerator / denominator
        return grad

    def hess_x_log_pdf(self, x, params=None, idxs_slice=slice(None, None, None),
                       *args, **kwargs):
        raise NotImplementedError


class GaussianRangeLogLikelihood(like.LogLikelihood):

    def __init__(self, distance: float, dim: int, variance: float):
        """
        Create a GaussianRangeLogLikelihood object
        :param distance: the distance measured between two nodes
        :type: float
        :param dim: the dimension of the distribution
        :type: int
        :param variance: sigma of the distance
        :type: float
        :raise ValueError:
            if distance is not positive
            if dim is not positive
            if sigma is not positive
        """
        # INIT MAP AND DISTRIBUTION
        if distance < 0:
            raise ValueError("Negative distance")
        if dim < 0:
            raise ValueError("Negative dimensionality")
        if variance < 0:
            raise ValueError("Negative sigma")
        self._dim = 2 * dim  # dim of two nodes
        self._y = distance
        self._dim_per_node = dim  # dim of one node

        super(GaussianRangeLogLikelihood, self).__init__(self._y, self._dim)
        self._distance_gaussian_distribution = \
            dist.GaussianDistribution(mu=np.array([distance]),
                                      sigma=np.array([[variance]]))

    def evaluate(self, x, *args, **kwargs):
        if len(x.shape) != 2 or x.shape[1] != self._dim:
            raise ValueError("The dimensionality of the locations is incorrect")
        distances = np.sqrt(
            ((x[:, self._dim_per_node:] - x[:, :self._dim_per_node]) ** 2)
                .sum(axis=1).reshape((-1, 1)))
        #        areas = (self._unit_sphere_area * distances ** (self._dim - 1)).reshape(-1)

        gaussian_log_pdf = self._distance_gaussian_distribution.log_pdf(
            distances)
        #        tmp = gaussian_log_pdf - np.log(areas)
        tmp = gaussian_log_pdf
        return tmp

    def grad_x(self, x, *args, **kwargs):
        if len(x.shape) != 2 or x.shape[1] != self._dim:
            raise ValueError("The dimensionality of the locations is incorrect")
        diff = x[:, self._dim_per_node:] - x[:, :self._dim_per_node]
        sq_distances = (diff ** 2).sum(axis=1).reshape((-1, 1))
        distances = np.sqrt(sq_distances)
        dev = np.column_stack((-diff, diff))
        #        tmp = (self._distance_gaussian_distribution.grad_x_log_pdf(distances) \
        #                * dev / distances - (self._dim - 1) * dev / sq_distances)
        tmp = (self._distance_gaussian_distribution.grad_x_log_pdf(
            distances) * dev / distances)
        return tmp

    def tuple_grad_x(self, x, cache=None, **kwargs):
        return (self.evaluate(x, cache=cache, **kwargs),
                self.grad_x(x, cache=cache, **kwargs))

    def hess_x(self, x, *args, **kwargs):
        raise NotImplementedError("To be implemented")

    def action_hess_x(self, x, dx, *args, **kwargs):
        raise NotImplementedError("To be implemented")


class GaussianDisplacementDistribution(dist.Distribution, metaclass=ABCMeta):
    def __init__(self, center: np.ndarray, variance: float,
                 distance: float) -> None:
        """
        Create a GaussianDisplacementDistribution object
        :param center: the center from which displacement is computed
        :type: np.ndarray (1-D)
        :param variance: the sigma of the 1-D Gaussian distribution
        :type: float
        :param distance: the expected distance
        :raise ValueError:
            if mu is not a 1-D numpy array
            if sigma is not positive
        """
        if len(center.shape) != 1:
            raise ValueError("mu must be a 1-D numpy array")
        elif variance <= 0.0:
            raise ValueError("sigma must be a positive float")
        self._dim = center.shape[0]
        super(GaussianDisplacementDistribution, self).__init__(self._dim)
        self._center = center
        self._distance_gaussian_distribution = \
            dist.GaussianDistribution(mu=np.array([distance]),
                                      sigma=np.array([[variance]]))
        V, S = 1.0, 2.0
        for n in range(self._dim - 1):
            V, S = S / float(n + 1), 2.0 * np.pi * V
        self._unit_sphere_area = S

    def rvs(self, m: int, *args, **kwargs) -> np.ndarray:
        """
        TODO: Code optimization
        :param m: The number of samples to be generated
        :type: int
        :param args:
        :param kwargs:
        :return: Random samples
        :rtype: numpy.ndarray with shape (m, dim)
        :raise ValueError: if the input m is negative
        """
        distances = self._distance_gaussian_distribution.rvs(m)
        sphere_samples = np.hstack([dist.StandardNormalDistribution(1).rvs(m)
                                    for _ in range(self._dim)])
        sphere_samples /= np.sqrt(sphere_samples.sum(axis=1).reshape(-1, 1))
        return sphere_samples * distances + self._center.reshape((1, -1))

    def quadrature(self, qtype: int = 0, qparams: int = 100, *args, **kwargs):
        """
        :param qtype: 0 for Monte Carlo
        :param qparams: number of samples
        :param args
        :param kwargs
        :return: num_samples X dim np.ndarray
        """
        return self.rvs(qparams), np.ones(qparams) / float(qparams)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def center(self) -> np.ndarray:
        return self._center

    @property
    def variance(self) -> np.ndarray:
        return self._distance_gaussian_distribution.sigma[0, 0]

    @property
    def distance(self) -> float:
        return self._distance_gaussian_distribution.mu[0]

    def pdf(self, x: np.ndarray, params=None,
            idxs_slice=slice(None, None, None), **kwargs) -> np.ndarray:
        """
        Evaluate probability densities at given locations
        :param x: Locations to evaluate the pdf's
        :type: numpy.ndarray with shape (num_locs, dim)
        :param params:
        :param idxs_slice:
        :param kwargs:
        :return: The densities
        :rtype: numpy.ndarray with shape (num_locs)
        :raise ValueError: if the dimensionality of locations to evaluate
            the probability densities is incorrect
        """
        if len(x.shape) != 2 or x.shape[1] != self._dim:
            raise ValueError("The dimensionality of the locations is incorrect")
        distances = np.sqrt(((x - self._center.reshape((1, -1))) ** 2)
                            .sum(axis=1).reshape((-1, 1)))
        areas = (self._unit_sphere_area * distances ** (self._dim - 1)
                 ).reshape(-1)
        return (self._distance_gaussian_distribution.pdf(distances) / areas)

    def log_pdf(self, x: np.ndarray, params=None,
                idxs_slice=slice(None, None, None), **kwargs) -> np.ndarray:
        """
        Evaluate the logarithm of probability densities at given locations
        :param x: Locations to evaluate the pdf's
        :type: numpy.ndarray with shape (num_locs, dim)
        :param params:
        :param idxs_slice:
        :param kwargs:
        :return: The logarithm of densities
        :rtype: numpy.ndarray with shape (num_locs)
        :raise ValueError: if the dimensionality of locations to evaluate
            the probability densities is incorrect
        """
        if len(x.shape) != 2 or x.shape[1] != self._dim:
            raise ValueError("The dimensionality of the locations is incorrect")
        distances = np.sqrt(((x - self._center.reshape((1, -1))) ** 2)
                            .sum(axis=1).reshape((-1, 1)))
        areas = (self._unit_sphere_area * distances ** (self._dim - 1)
                 ).reshape(-1)
        #        tmp = (self._distance_gaussian_distribution.log_pdf(distances) - np.log(areas))
        tmp = self._distance_gaussian_distribution.log_pdf(distances)
        return tmp

    def grad_x_log_pdf(self, x: np.ndarray, params=None,
                       idxs_slice=slice(None, None, None),
                       **kwargs) -> np.ndarray:
        """
        Evaluate the gradient of log(pdf) at given locations
        :param x: Locations to evaluate the pdf's
        :type: numpy.ndarray with shape (num_locs, dim)
        :param params:
        :param idxs_slice:
        :param kwargs:
        :return: The logarithm of densities
        :rtype: numpy.ndarray with shape (num_locs, dim)
        :raise ValueError: if the dimensionality of input is incorrect
        """
        if len(x.shape) != 2 or x.shape[1] != self._dim:
            raise ValueError("The dimensionality of the locations is incorrect")
        dev = x - self._center.reshape((1, -1))
        sq_distances = (dev ** 2).sum(axis=1).reshape((-1, 1))
        distances = np.sqrt(sq_distances)
        #        tmp = (self._distance_gaussian_distribution.grad_x_log_pdf(distances) * dev / distances - (self._dim - 1) * dev / sq_distances)
        tmp = self._distance_gaussian_distribution.grad_x_log_pdf(
            distances) * dev / distances
        return tmp

    # def hess_x_log_pdf(self, x, params=None, idxs_slice=slice(None,None,None),
    #                    *args, **kwargs):
    #     h = 1e-2
    #     return (self.log_pdf(x + h) + self.log_pdf(x - h) - 2.0 * self.log_pdf(x)) / h ** 2
