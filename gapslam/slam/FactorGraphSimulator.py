from geometry.TwoDimension import SE2Pose
from factors.Factors import SE2RelativeGaussianLikelihoodFactor, \
    R2RelativeGaussianLikelihoodFactor, R2RangeGaussianLikelihoodFactor, \
    LikelihoodFactor, SE2R2RangeGaussianLikelihoodFactor
import numpy as np
import TransportMaps.Distributions as dist
from slam.Variables import VariableType, Variable, R2Variable, \
    SE2Variable
import matplotlib.pyplot as plt
from typing import Tuple, Iterable, Dict, List, ClassVar
from factors.Factors import Factor, UnaryR2GaussianPriorFactor, UnarySE2ApproximateGaussianPriorFactor
from enum import Enum


class SensorType(Enum):
    relative = 0
    range_only_gaussian = 1


def read_variable_and_truth_from_line(line: str) -> Tuple[Variable, np.ndarray]:
    """
    Read and create a Variable object from a list of strings
        The string is formatted as: [variable_type, class_name, var_name]
        + true_pose
    Return the ground truth location/pose of the variable
    """
    var = Variable.construct_from_text(line)
    line = line.strip().split()
    val = np.array([float(line[i]) for i in range(4, 4 + var.dim)])
    return var, val


def write_variable_and_truth_to_line(
        var: Variable, truth: np.ndarray = None) -> str:
    """
    Write a variable and its ground truth location/pose to a string
    """
    line = str(var)
    if truth is not None:
        line += " " + " ".join([str(v) for v in truth])
    return line


def factor_graph_to_string(variables: Iterable[Variable],
                           factors: Iterable[Factor],
                           var_truth: Dict[Variable, np.ndarray] = None) -> str:
    res = [None] * (len(variables) + len(factors))
    for i, var in enumerate(variables):
        truth = var_truth[var] if var in var_truth else None
        res[i] = write_variable_and_truth_to_line(var, truth)
    for i, factor in enumerate(factors):
        res[len(variables) + i] = str(factor)
    return "\n".join(res)


def read_factor_graph_from_file(file_name: str) -> Tuple[List[Variable],
                                                         Dict[Variable,
                                                              np.ndarray],
                                                         List[Factor]]:
    with open(file_name) as file:
        variables = []
        var_poses = {}
        line = file.readline()
        factors = []
        while line:
            texts = line.strip().split()
            if texts[0] == "Variable":
                var, val = read_variable_and_truth_from_line(line)
                variables.append(var)
                var_poses[var] = val
            elif texts[0] == "Factor":
                factors.append(Factor.construct_from_text(line, variables))
            line = file.readline()
    return variables, var_poses, factors


def \
        generate_measurements_for_factor_graph(
        input_file_name: str,
        odometry_class: ClassVar,
        landmark_measurement_class: ClassVar,
        landmark_measurement_range: float,
        output_file_name: str = None,
        max_measurements_allowed: int = 1,
        **kwargs) -> Tuple[List[Variable], Dict[Variable, np.ndarray],
                           List[Factor]]:
    def _create_r2_relative_gaussian_factor(
            var1: R2Variable, var2: R2Variable, covariance: np.ndarray,
            obs: np.ndarray = None
    ) -> R2RelativeGaussianLikelihoodFactor:
        if obs is None:
            obs = np.zeros(R2RelativeGaussianLikelihoodFactor.measurement_dim)
        elif len(obs.shape) != 1 or obs.shape[0] \
                != R2RelativeGaussianLikelihoodFactor.measurement_dim:
            raise ValueError("Dimensionality of obs is incorrect")
        return R2RelativeGaussianLikelihoodFactor(var1=var1, var2=var2,
                                                  observation=obs,
                                                  covariance=covariance)

    def _create_se2_relative_gaussian_factor(
            var1: R2Variable, var2: R2Variable, covariance: np.ndarray,
            obs: np.ndarray = None
    ) -> SE2RelativeGaussianLikelihoodFactor:
        if obs is None:
            obs = np.zeros(SE2RelativeGaussianLikelihoodFactor.measurement_dim)
        elif len(obs.shape) != 1 or obs.shape[0] \
                != SE2RelativeGaussianLikelihoodFactor.measurement_dim:
            raise ValueError("Dimensionality of obs is incorrect")
        return SE2RelativeGaussianLikelihoodFactor(var1=var1, var2=var2,
                                                  observation=SE2Pose.by_array(obs),
                                                  covariance=covariance)

    def _create_se2_r2_range_gaussian_factor(
        var1: SE2Variable, var2: R2Variable, sigma: float, obs: np.ndarray = None
    ) -> SE2R2RangeGaussianLikelihoodFactor:
        if obs is None:
            obs = np.zeros(SE2R2RangeGaussianLikelihoodFactor.measurement_dim)
        elif len(obs.shape) != 1 or obs.shape[0] \
                != SE2R2RangeGaussianLikelihoodFactor.measurement_dim:
            raise ValueError("Dimensionality of obs is incorrect")
        return SE2R2RangeGaussianLikelihoodFactor(var1=var1, var2=var2, observation=obs, sigma=sigma)

    def _create_r2_range_gaussian_factor(
            var1: R2Variable, var2: R2Variable, sigma: float,
            obs: np.ndarray = None
    ) -> R2RangeGaussianLikelihoodFactor:
        if obs is None:
            obs = np.zeros(R2RangeGaussianLikelihoodFactor.measurement_dim)
        elif len(obs.shape) != 1 or obs.shape[0] \
                != R2RangeGaussianLikelihoodFactor.measurement_dim:
            raise ValueError("Dimensionality of obs is incorrect")
        return R2RangeGaussianLikelihoodFactor(var1=var1, var2=var2,
                                               observation=obs,
                                               sigma=sigma)

    def _create_se2_r2_range_gaussian_factor_with_ambiguous_association(
        var1: SE2Variable, var2: R2Variable, sigma: float, obs: np.ndarray = None
    ) -> SE2R2RangeGaussianLikelihoodFactor:
        if obs is None:
            obs = np.zeros(SE2R2RangeGaussianLikelihoodFactor.measurement_dim)
        elif len(obs.shape) != 1 or obs.shape[0] \
                != SE2R2RangeGaussianLikelihoodFactor.measurement_dim:
            raise ValueError("Dimensionality of obs is incorrect")
        return SE2R2RangeGaussianLikelihoodFactor(var1=var1, var2=var2, observation=obs, sigma=sigma)

    def _create_odometry_factor(var1: Variable, var2: Variable,
                                obs: np.ndarray = None) -> LikelihoodFactor:
        if odometry_class == R2RelativeGaussianLikelihoodFactor:
            covariance = kwargs["odometry_covariance"] \
                if "odometry_covariance" in kwargs else \
                np.identity(2) * kwargs["odometry_sigma"] ** 2
            return _create_r2_relative_gaussian_factor(var1=var1, var2=var2,
                                                       covariance=covariance,
                                                       obs=obs)
        elif odometry_class == SE2RelativeGaussianLikelihoodFactor:
            if "odometry_covariance" in kwargs:
                covariance = kwargs["odometry_covariance"]
            else:
                covariance = np.identity(3) * kwargs["odometry_sigma"] ** 2
                covariance[2, 2] = kwargs["orientation_sigma"] ** 2
            return _create_se2_relative_gaussian_factor(var1=var1, var2=var2,
                                                       covariance=covariance,
                                                       obs=obs)
        else:
            raise ValueError("The odometry factor is not supported yet")

    def _create_landmark_measurement_factor(
            pose_var: Variable, landmark_var: Variable, obs: np.ndarray = None
    ) -> LikelihoodFactor:
        if landmark_measurement_class == R2RelativeGaussianLikelihoodFactor:
            covariance = kwargs["landmark_covariance"] \
                if "landmark_covariance" in kwargs else \
                np.identity(2) * kwargs["landmark_sigma"] ** 2
            return _create_r2_relative_gaussian_factor(var1=pose_var,
                                                       var2=landmark_var,
                                                       covariance=covariance,
                                                       obs=obs)
        elif landmark_measurement_class == R2RangeGaussianLikelihoodFactor:
            return _create_r2_range_gaussian_factor(var1=pose_var,
                                                    var2=landmark_var,
                                                    sigma=kwargs[
                                                        "landmark_sigma"],
                                                    obs=obs)
        elif landmark_measurement_class == SE2R2RangeGaussianLikelihoodFactor:
            return _create_se2_r2_range_gaussian_factor(var1=pose_var,
                                                    var2=landmark_var,
                                                    sigma=kwargs[
                                                        "landmark_sigma"],
                                                    obs=obs)

    # Read all nodes, true positions, and pre-defined factors
    variables, truth, factors = read_factor_graph_from_file(input_file_name)

    # Generate odometry measurements
    poses = [var for var in variables if var.type == VariableType.Pose]
    landmarks = [var for var in variables if var.type == VariableType.Landmark]
    for i in range(len(poses) - 1):
        var_from, var_to = poses[i: i + 2]
        tmp_factor = _create_odometry_factor(
            var1=var_from, var2=var_to)
        obs = tmp_factor.sample(var1=truth[var_from].reshape(
            (1, var_from.dim)), var2=truth[var_to].reshape((1, var_to.dim))
        ).reshape(tmp_factor.measurement_dim)
        factor = _create_odometry_factor(var1=var_from, var2=var_to, obs=obs)
        factors.append(factor)

    # Generate landmark measurements
    for var in poses:
        translational_dim = var.translational_dim
        pose_location = truth[var][:translational_dim]
        landmark_distances = {l: np.sqrt(np.sum((pose_location -
                                                 truth[l][:translational_dim])
                                                ** 2)
                                         ) for l in landmarks}
        detected_landmarks = [l for l in landmarks if landmark_distances[l] <=
                              landmark_measurement_range]
        if not detected_landmarks:
            continue
        else:
            num_factors = min(max_measurements_allowed, len(detected_landmarks))
            landmarks_to_add = sorted(detected_landmarks,
                                      key=lambda x: landmark_distances[x])[
                               :num_factors]
            for landmark in landmarks_to_add:
                tmp_factor = _create_landmark_measurement_factor(
                    pose_var=var, landmark_var=landmark)
                obs = tmp_factor.sample(var1=truth[var].reshape(
                    (1, var.dim)),
                    var2=truth[landmark].reshape((1, landmark.dim))
                ).reshape(tmp_factor.measurement_dim)
                factor = _create_landmark_measurement_factor(
                    pose_var=var, landmark_var=landmark, obs=obs)
                factors.append(factor)

    # Write into output file
    text_file = open(output_file_name, "w")
    text_file.write(factor_graph_to_string(variables, factors, truth))
    text_file.close()

    return variables, truth, factors


def separate_incremental_nodes_and_factors(variables: Iterable[Variable],
                                           truth: Dict[Variable, np.ndarray],
                                           factors: Iterable[Factor]
                                           ) -> Tuple[List[List[Variable]]]:
    pass


class G2oToroPoseGraphReader(object):
    file_type_list = ["g2o", "graph"]
    node_header_list = ["VERTEX_SE2", "VERTEX2"]
    factor_header_list = ["EDGE_SE2", "EDGE2"]
    info_mat_format_list = [[(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)],
                            [(0, 0), (0, 1), (1, 1), (2, 2), (0, 2), (1, 2)]]

    def __init__(self, file_path: str, correlated_R_t: bool = True,
                 ignore_orientation: bool = False,
                 synthetic_observation: bool = False, covariance: float = None):
        """
        formats of pose graph files refer to
            https://www.dropbox.com/s/uwwt3ni7uzdv1j7/g2oVStoro.pdf?dl=0
            convert a pose graph file to our NFiSAM object
        :param file_path: a string of file directory
        :param correlated_R_t: true if using wrapped Gaussian on manifold;
            false if using von Mises
        :param ignore_orientation: when false, SE2 poses are used; when true,
            R2 poses are used
        :param synthetic_observation: when false, we use the data from g2o file;
            when true, we generate synthetic observations
        :param covariance: the sigma of likelihood factors
        """
        self._correlated_R_t = correlated_R_t
        self._file_path = file_path
        self._file_type, self._node_head, self._factor_head, \
        self._info_mat_format = self.getFileType()
        self._node_list = []
        self._factor_list = []
        self._true_location_mapping = {}
        dim = 2 if ignore_orientation else 3
        with open(file_path) as fp:
            line = fp.readline()
            cnt = 1
            while line:
                line_list = line.strip().split()
                if line_list[0] == self._node_head:
                    # reading lines for init solution to find new nodes
                    var = R2Variable(line_list[1]) if ignore_orientation else SE2Variable(line_list[1])
                    self._node_list.append(var)
                    self._true_location_mapping[var] = \
                        np.array([float(line_list[2]), float(line_list[3])]) if ignore_orientation else \
                        np.array([float(line_list[2]), float(line_list[3]), float(line_list[4])])
                elif line_list[0] == self._factor_head:
                    if not synthetic_observation:
                        info_mat = np.zeros((dim, dim))
                        for i in range(6, 12):
                            info_mat[self._info_mat_format[i - 6]] = float(
                                line_list[i])
                            info_mat[
                                self._info_mat_format[i - 6][::-1]] = float(
                                line_list[i])
                        cov_mat = np.linalg.inv(info_mat)
                        var1 = R2Variable(line_list[1]) if ignore_orientation else SE2Variable(line_list[1])
                        var2 = R2Variable(line_list[2]) if ignore_orientation else SE2Variable(line_list[2])
                        tmp = R2RelativeGaussianLikelihoodFactor(
                            var1=var1,
                            var2=var2,
                            observation=np.ndarray([line_list[3], line_list[4]]
                                                   ),
                            covariance=cov_mat[:2, :2]) \
                            if ignore_orientation else \
                            SE2RelativeGaussianLikelihoodFactor(
                                var1=var1,
                                var2=var2,
                                observation=SE2Pose(x=float(line_list[3]),
                                                    y=float(line_list[4]),
                                                    theta=float(line_list[5])),
                                covariance=cov_mat,
                                correlated_R_t=correlated_R_t)
                        self._factor_list.append(tmp)
                    else:
                        if ignore_orientation:
                            var1 = R2Variable(line_list[1])
                            var2 = R2Variable(line_list[2])
                            obs = self._true_location_mapping[var2] - \
                                  self._true_location_mapping[var1]
                            if covariance is None:
                                cov = np.identity(dim)
                            else:
                                cov = covariance
                                obs += dist.GaussianDistribution(
                                    mu=np.zeros(dim), sigma=cov).rvs(
                                    1).reshape(dim)
                            tmp = R2RelativeGaussianLikelihoodFactor(
                                var1=var1, var2=var2,
                                observation=obs,
                                covariance=cov)
                            self._factor_list.append(tmp)
                line = fp.readline()
                cnt += 1

        plt.plot([self._true_location_mapping[node][0] for node in
                  self._node_list], [self._true_location_mapping[node][1]
                                     for node in
                                     self._node_list], c='k')
        plt.savefig(file_path+'.png', dpi=300)
        plt.show()

    def dataForSolver(self, prior_cov_scale=.1):
        var0 = self._node_list[0]
        if var0.dim == 2:
            prior_factor = UnaryR2GaussianPriorFactor(var=var0,
                                                      mu=self._true_location_mapping[var0],
                                                      covariance=prior_cov_scale * np.identity(var0.dim))
        elif var0.dim == 3:
            prior_factor = UnarySE2ApproximateGaussianPriorFactor(var=var0,
                                                                  prior_pose=SE2Pose.by_array(self._true_location_mapping[var0]),
                                                                  covariance=prior_cov_scale * np.identity(var0.dim))

        factors = [prior_factor] + self._factor_list
        return self._node_list, factors, self._true_location_mapping

    def getFileType(self):
        i = 0
        for type in G2oToroPoseGraphReader.file_type_list:
            if self._file_path.endswith(type):
                return type, G2oToroPoseGraphReader.node_header_list[i], \
                       G2oToroPoseGraphReader.factor_header_list[i], \
                       G2oToroPoseGraphReader.info_mat_format_list[i]
            i += 1
        raise ValueError("Can not recognize the suffix of input file")

    @property
    def file_type(self):
        return self._file_type

    @property
    def node_head(self):
        return self._node_head

    @property
    def factor_head(self):
        return self._factor_head

    @property
    def info_mat_format(self):
        return self._info_mat_format

    @property
    def file_path(self):
        return self._file_path

    @property
    def node_list(self):
        return self._node_list

    @property
    def factor_list(self):
        return self._factor_list


