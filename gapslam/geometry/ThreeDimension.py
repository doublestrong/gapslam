from copy import deepcopy

import gtsam

from utils.Functions import none_to_zero, theta_to_pipi
import numpy as np
import math
from typing import List, Tuple, Union
from utils.Units import _DEG_TO_RAD_FACTOR, _RAD_TO_DEG_FACTOR
from scipy.spatial.transform import Rotation as sciR

def quat2mat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    # check quaternion
    assert abs(qw ** 2 + qx ** 2 + qy ** 2 + qz ** 2 - 1.0) < 1e-4
    return np.array([[1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                     [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qz * qy - 2 * qx * qw],
                     [2 * qx * qz - 2 * qy * qw, 2 * qz * qy + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]])

def skewSymMat(rot_axis):
    return np.array([[0, -rot_axis[2], rot_axis[1]],
              [rot_axis[2], 0, -rot_axis[0]],
              [-rot_axis[1], rot_axis[0], 0]])

class Point3(object):
    dim = 3
    def __init__(self, x: float = None, y: float = None, z: float = None) -> None:
        """
        Create a 3D point
        : please use += -+ ... to do modify self properties
        """
        self._x = none_to_zero(x)
        self._y = none_to_zero(y)
        self._z = none_to_zero(z)
        assert np.isscalar(self._x) and \
               np.isscalar(self._y) and \
               np.isscalar(self._z)

    @classmethod
    def by_array(cls, other: Union[List[float], Tuple[float], np.ndarray]
                 ) -> "Point3":
        return cls(other[0], other[1], other[2])

    @property
    def x(self) -> float:
        return self._x

    @property
    def z(self) -> float:
        return self._z

    @property
    def y(self) -> float:
        return self._y

    @property
    def norm(self) -> float:
        return np.sqrt(self.x ** 2 + self.y ** 2 + self.z **2)

    @property
    def array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def inverse(self) -> "Point3":
        return Point3(-self.x, -self.y, -self.z)

    def set_xyz(self, x: float = None, y: float = None, z: float = None) -> "Point3":
        if x is not None:
            self._x = x
        if y is not None:
            self._y = y
        if z is not None:
            self._z = z
        return self

    def copy(self) -> "Point3":
        return Point3(self.x, self.y, self.z)

    def transform_to(self, other: "Point3") -> "Point3":
        return other - self

    def distance(self, other: "Point3") -> float:
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def __add__(self, other: "Point3") -> "Point3":
        if isinstance(other, Point3):
            return Point3(self.x + other.x, self.y + other.y, self.z + other.z)
        raise ValueError("Not a Point3 type to add.")

    def __sub__(self, other: "Point3") -> "Point3":
        """
        Subtraction
        """
        if isinstance(other, Point3):
            return Point3(self.x - other.x, self.y - other.y, self.z - other.z)
        raise ValueError("Not a Point3 type to minus.")

    def __mul__(self, other: Union[int, float]) -> "Point3":
        """
        Scalar multiplication
        """
        if np.isscalar(other):
            return Point3(self.x * other, self.y * other, self.z * other)
        raise ValueError("Not an int or float type to multiply.")

    def __rmul__(self, other: Union[int, float]) -> "Point3":
        if np.isscalar(other):
            return Point3(self.x * other, self.y * other, self.z * other)
        raise ValueError("Not an int or float type to multiply.")

    def __truediv__(self, other: Union[int, float]) -> "Point3":
        if np.isscalar(other):
            if other == 0.0:
                raise ValueError("Cannot divide by zeros.")
            else:
                return Point3(self.x / other, self.y / other, self.z / other)
        raise ValueError("Not an int or float type to divide.")

    def __iadd__(self, other: "Point3") -> "Point3":
        if isinstance(other, Point3):
            self._x += other.x
            self._y += other.y
            self._z += other.z
            return self
        else:
            raise ValueError("Not a Point3 type to add.")

    def __isub__(self, other: "Point3") -> "Point3":
        if isinstance(other, Point3):
            self._x -= other.x
            self._y -= other.y
            self._z -= other.z
            return self
        else:
            raise ValueError("Not a Point3 type to minus.")

    def __imul__(self, other: Union[int, float]) -> "Point3":
        if np.isscalar(other):
            self._x *= other
            self._y *= other
            self._z *= other
            return self
        else:
            raise ValueError("Not an int or float type to multiply.")

    def __itruediv__(self, other: Union[int, float]) -> "Point3":
        if np.isscalar(other):
            if other == 0.0:
                raise ValueError("Cannot divide by zeros.")
            else:
                self._x /= other
                self._y /= other
                self._z /= other
                return self
        else:
            raise ValueError("Not an int or float type to multiply.")

    def __neg__(self) -> "Point3":
        return self.inverse()

    def __str__(self) -> str:
        return "Point3{x: " + str(self.x) + ", y: " + str(self.y) + ", z: " + str(self.z) + "}"

    def __eq__(self, other: "Point3") -> bool:
        if isinstance(other, Point3):
            return abs(self.x - other.x) < 1e-8 and abs(self.y - other.y) < 1e-8 and abs(self.z - other.z) < 1e-8
        return False

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z))

class SE3Pose:
    dim = 6
    def __init__(self, mat: np.ndarray):
        """
        :param mat: transformation matrix 4 x 4
        """
        self.mat = mat
        self.rot_vec = sciR.from_matrix(self.mat[:3,:3]).as_rotvec()

    @classmethod
    def by_transQuat(cls, x: float, y: float, z: float, qw: float, qx: float, qy: float, qz: float):
        """
        Construction using translation vector and quaternion
        :param x:
        :param y:
        :param z:
        :param qw:
        :param qx:
        :param qy:
        :param qz:
        :return:
        """
        # convert to quaternion
        mat = np.eye(4)
        mat[:3, :3] = quat2mat(qw, qx, qy, qz)
        mat[:3, 3] = np.array([x, y, z])
        return cls(mat)

    @classmethod
    def by_trans_rotvec(cls, arr):
        """
        Construction using translation vector and rotation vector
        :param arr:
        :return:
        """
        # convert to quaternion
        mat = np.eye(4)
        mat[:3, :3] = sciR.from_rotvec(arr[3:]).as_matrix()
        mat[:3, 3] = arr[:3]
        return cls(mat)

    @classmethod
    def by_exp_map(cls, tangent_vec: np.ndarray):
        # see page 42 in Murray's book for the math
        # http://www.cse.lehigh.edu/~trink/Courses/RoboticsII/reading/murray-li-sastry-94-complete.pdf
        # same as the canonical form, Eq 174, in https://arxiv.org/pdf/1812.01537.pdf
        # implementation is same as gtsam pose3
        # https://github.com/devbharat/gtsam/blob/master/gtsam/geometry/Pose3.cpp~
        trans_vec = tangent_vec[3:]
        rot_vec = tangent_vec[:3]
        # create a rotation by the exponential map at identity
        theta = np.linalg.norm(rot_vec)
        rot_axis = rot_vec / theta
        rot_mat = sciR.from_rotvec(rot_vec).as_matrix()
        parallel_trans = np.dot(trans_vec, rot_axis)
        cross_trans = np.cross(rot_axis, trans_vec)
        trans_point = (cross_trans-np.dot(rot_mat, cross_trans))/theta + parallel_trans*rot_axis
        mat = np.eye(4)
        mat[:3, :3] = rot_mat
        mat[:3, 3] = trans_point
        return cls(mat)

    def log_map(self):
        # implementation same as Pose3::Logmap in GTSAM
        # https://github.com/devbharat/gtsam/blob/master/gtsam/geometry/Pose3.cpp~
        # return a 1D vector with 6 elements where the first three are for translation
        rot_vec = deepcopy(self.rot_vec)
        theta = np.linalg.norm(rot_vec)
        rot_axis = rot_vec / theta
        rot_axis_skew_sym = skewSymMat(rot_axis)
        tan_theta = np.tan(0.5 * theta)
        skew_sym_trans = np.dot(rot_axis_skew_sym, self.translation)
        rho = self.translation - (0.5 * theta) * skew_sym_trans + \
              (1- theta/(2*tan_theta)) * np.dot(rot_axis_skew_sym, skew_sym_trans)
        return np.concatenate([rot_vec, rho])

    def det_grad_x_logmap(self):
        return np.linalg.det(gtsam.Pose3.LogmapDerivative(mat=gtsam.Pose3(self.mat)))

    def rangeBearing2point(self, distance, elevation, azimuth):
        """
        The
        :param distance:
        :param elevation:
        :param azimuth:
        :return:
        """
        ptInCam = [distance*np.cos(elevation)*np.sin(azimuth), -distance*np.sin(elevation), distance*np.cos(elevation)*np.cos(azimuth), 1]
        return np.dot(self.mat, ptInCam)[:3]

    def worldPtInCam(self, pt):
        h_pt = np.concatenate((pt, [1]))
        return np.dot(self.inverse().mat, h_pt)[:3]

    def camPtInWorld(self, pt):
        h_pt = np.concatenate((pt, [1]))
        return np.dot(self.mat, h_pt)[:3]

    def point2rangeBearing(self, point):
        """
        The
        :param distance:
        :param elevation:
        :param azimuth:
        :return:
        """
        ptInCam = self.worldPtInCam(point)
        r = np.linalg.norm(ptInCam)
        azimuth = np.arctan2(ptInCam[0], ptInCam[2])
        elevation = np.arctan2(-ptInCam[1], np.linalg.norm(ptInCam[[0,2]]))
        return elevation, azimuth, r

    def point2pixel(self, pt:np.ndarray, cam_mat:np.ndarray):
        ptInCam = self.worldPtInCam(pt)
        # assert ptInCam[-1] > 0
        ptInCam = ptInCam/ptInCam[-1]
        return np.dot(cam_mat, ptInCam)[:2]

    def depth2point(self, pixel_xy: np.ndarray, depth: float, cam_mat: np.ndarray):
        rx, ry = (pixel_xy - np.array([cam_mat[0, 2], cam_mat[1, 2]])) / np.array([cam_mat[0, 0], cam_mat[1, 1]])
        ptInCam = np.array([rx, ry, 1]) * depth
        return self.camPtInWorld(ptInCam)

    @classmethod
    def from2d(cls, x: float, y: float, theta: float):
        # convert to quaternion
        mat = np.eye(4)
        mat[:3, 3] = np.array([x, y, .0])
        mat[:3, 0] = [np.cos(theta), np.sin(theta), .0]
        mat[:3, 1] = [-np.sin(theta), np.cos(theta), .0]
        return cls(mat)

    @property
    def translation(self):
        """
        Translation vector
        :return:
        """
        return self.mat[:3, 3]

    @property
    def rotation(self):
        """
        Rotation matrix
        :return:
        """
        return self.mat[:3, :3]

    @property
    def transQuat(self):
        """
        :return: tx ty tz qw qx qy qz
        """
        tmp = sciR.from_matrix(self.rotation).as_quat() # xyzw
        tmp2 = np.zeros(4)
        tmp2[0] = tmp[-1]
        tmp2[1:] = tmp[:3]
        return np.concatenate((self.translation,tmp2))

    @property
    def array(self):
        """
        :return: tx ty tz and rotation vector
        """
        return np.concatenate((self.translation, self.rot_vec))

    def inverse(self):
        mat = self.mat.copy()
        mat[:3, :3] = mat[:3, :3].T
        mat[:3, 3] = -mat[:3, :3] @ mat[:3, 3]
        return SE3Pose(mat)

    def tf_point(self, arr: np.array):
        vec = np.array([arr[0], arr[1], arr[2], 1])
        return (self.mat @ vec)[:3]

    def tf_points(self, points: np.array):
        h_points = np.hstack([points, np.ones((len(points), 1))])
        return (self.mat @ h_points.T).T[:, :3]

    def __mul__(self, other):
        if isinstance(other, SE3Pose):
            return SE3Pose(self.mat @ other.mat)
        raise ValueError("Not a SE3Pose type to multiply.")

def angular_dist(pose1: SE3Pose, pose2: SE3Pose):
    """
    Angular distance between two poses
    :param pose1:
    :param pose2:
    :return: theta in [0, pi]
    """
    temp = np.clip((np.trace(pose1.rotation.T @ pose2.rotation) - 1) / 2, -1,
                   1)
    return abs(np.arccos(temp))
