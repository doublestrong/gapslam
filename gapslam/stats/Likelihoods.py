from abc import ABCMeta
import numpy as np
import TransportMaps.Distributions as dist
import TransportMaps.Likelihoods as like
from typing import List


class LogLikelihood(metaclass=ABCMeta):
    @property
    def observation(self) -> np.ndarray:
        raise NotImplementedError

    def evaluate(self, x: np.ndarray) -> np.ndarray:
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

    def grad_x(self, x: np.ndarray) -> np.ndarray:
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
    def dim(self) -> int:
        """
        Dimensionality of latent variables, not the observation/measurement
        """
        raise NotImplementedError


def is_symmetric(a: np.ndarray, rtol=1e-5, atol=1e-8) -> bool:
    """
    Check whether matrix a is symmetric
    :param a: the matrix to be tested
    :type: numpy.ndarray
    :param rtol: relative tolerance
    :type: float
    :param atol: absolute tolerance
    :type: float
    :return: whether matrix a is symmetric
    :rtype: bool
    :raise ValueError: when the numpy array is not a matrix
    """
    if len(a.shape) == 2:
        return np.allclose(a, a.T, rtol=rtol, atol=atol)
    else:
        raise ValueError("The input numpy array must be a matrix")


class GaussianMixtureLogLikelihood(LogLikelihood):
    def __init__(self, weights: List[float], means: List[np.ndarray],
                 sigmas: List[np.ndarray], minus_mat: np.ndarray) -> None:
        # TODO: change assert statement to raise statement, add docstring
        #caution: y = c + epsilon + T*x, epsilon~(mu, sigma)
        self._num_components = len(weights)
        self._dim = minus_mat.shape[1]
        self._y = means[0]
        self._dim_y = len(means[0])
        assert self._num_components == len(means) == len(sigmas) > 0
        assert all([w > 0 for w in weights])
        assert np.isclose(np.sum(weights), 1.0)
        assert minus_mat.shape[0] == self._dim_y
        assert minus_mat.shape[1] %2 == 0
        self._dim_per_node = int(self._dim / 2)

        for mean in means:
            assert len(mean.shape) == 1 and mean.shape[0] == self._dim_y
        for cov in sigmas:
            assert (len(cov.shape) == 2 and is_symmetric(cov) and
                    np.all(np.linalg.eigvals(cov) > 0) and cov.shape[0] == self._dim_y)

        super(GaussianMixtureLogLikelihood, self).__init__(self._y, self._dim)
        self._weights = weights
        self._means = means
        self._covs = sigmas
        self._T = minus_mat
        self._components = [like.AdditiveLinearGaussianLogLikelihood(
            mu=np.zeros(self._dim_y), sigma=sigmas[comp], y=means[comp],
            c=np.zeros(self._dim_y), T=self._T) for comp in range(self._num_components)]
        self._det_sigmas = np.array([ np.linalg.det(cov) for cov in sigmas])
        self._inv_sigmas = [ np.linalg.inv(cov) for cov in sigmas]

    @property
    def num_components(self) -> int:
        return self._num_components

    @property
    def weights(self) -> List[float]:
        return self._weights

    @property
    def means(self) -> List[np.ndarray]:
        return self._y

    @property
    def sigmas(self) -> List[np.ndarray]:
        return self._covs

    def component(self, index: int) -> like.AdditiveLinearGaussianLogLikelihood:
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

    def evaluate(self, x: np.ndarray, idxs_slice=slice(None, None, None),
                 cache=None, **kwargs) -> np.ndarray:
        """
        """
        if len(x.shape) != 2 or x.shape[1] != self._dim:
            raise ValueError("The dimensionality of the locations is incorrect")
        num_locs = x.shape[0]
        densities = np.zeros(num_locs)
        for comp in range(self._num_components):
            densities += self._weights[comp] * dist.GaussianDistribution(
                mu=(self._means[comp]), sigma=self._covs[comp]).pdf( (self._T @ x.T).T )
        return np.log(densities)

    def grad_x(self, x: np.ndarray,
               idxs_slice=slice(None, None, None), cache=None,
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
        for i in range(num_locs):
            loc = x[[i], :]
            displacements = [loc @ self._T.T - mean
                             for mean in self._means]
            exponents = np.array([-0.5 * float(displacements[c]
                                           @ self._inv_sigmas[c]
                                           @ displacements[c].T)
                                           for c 
                                           in range(self._num_components)])
            max_exponent = np.max(exponents)
            exponents -= max_exponent
            pdfs = (np.exp(exponents) * np.array(self._weights)
                    / np.sqrt((2.0 * np.pi) ** self._dim_y * self._det_sigmas))
            denominator = np.sum(pdfs)
            factors = np.array([(-displacements[c] 
                                @ self._inv_sigmas[c]
                                @ self._T).
                                reshape(-1) for c in
                                range(self._num_components)])
            numerator = np.zeros((1, self._dim))
            for c in range(self._num_components):
                numerator += pdfs[c] * factors[c]
            grad[[i], :] = numerator / denominator
        return grad


    # def hess_x_log_pdf(self, x, params=None, idxs_slice=slice(None,None,None),
    #                    *args, **kwargs):
    #     h = 1e-2
    #     return (self.log_pdf(x + h) + self.log_pdf(x - h) - 2.0 * self.log_pdf(x)) / h ** 2

# class GaussianMixtureDistribution(dist.GaussianDistribution):
#     def __init__(self, weights: List[float], means: List[np.ndarray],
#                  sigmas: List[np.ndarray]) -> None:
#         # TODO: change assert statement to raise statement, add docstring
#         super(dist.GaussianDistribution, self).__init__(1)
#         self._mu = means[0]
#         self._sigma = sigmas[0]
#         self.inv_sigma = np.linalg.inv(self._sigma)
#         self.log_det_sigma = np.log(np.linalg.det(self._sigma))

class GaussianRangeLogLikelihood(LogLikelihood):

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


class GaussianDisplacementDistribution(LogLikelihood):
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