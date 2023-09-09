from utils.Functions import none_to_zero, theta_to_pipi
import numpy as np
import math
from typing import List, Tuple, Union
from utils.Units import _DEG_TO_RAD_FACTOR, _RAD_TO_DEG_FACTOR
import gtsam

class Point2(object):
    dim = 2

    def __init__(self, x: float = None, y: float = None) -> None:
        """
        Create a 2D point
        : please use += -+ ... to do modify self properties
        """
        self._x = none_to_zero(x)
        self._y = none_to_zero(y)
        assert np.isscalar(self._x) and \
               np.isscalar(self._y)

    @classmethod
    def by_array(cls, other: Union[List[float], Tuple[float], np.ndarray]
                 ) -> "Point2":
        return cls(other[0], other[1])

    @staticmethod
    def dist(x1: np.ndarray, x2: np.ndarray):
        """Euclidean distance between two points"""
        return np.linalg.norm(x1 - x2)

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def norm(self) -> float:
        return np.sqrt(self.x ** 2 + self.y ** 2)

    @property
    def array(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def inverse(self) -> "Point2":
        return Point2(-self.x, -self.y)

    def set_x_y(self, x: float = None, y: float = None) -> "Point2":
        if x is not None:
            self._x = x
        if y is not None:
            self._y = y
        return self

    def copy(self) -> "Point2":
        return Point2(self.x, self.y)

    def transform_to(self, other: "Point2") -> "Point2":
        return other - self

    def distance(self, other: "Point2") -> float:
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __add__(self, other: "Point2") -> "Point2":
        if isinstance(other, Point2):
            return Point2(self.x + other.x, self.y + other.y)
        raise ValueError("Not a Point2 type to add.")

    def __sub__(self, other: "Point2") -> "Point2":
        """
        Subtraction
        """
        if isinstance(other, Point2):
            return Point2(self.x - other.x, self.y - other.y)
        raise ValueError("Not a Point2 type to minus.")

    def __mul__(self, other: Union[int, float]) -> "Point2":
        """
        Scalar multiplication
        """
        if np.isscalar(other):
            return Point2(self.x * other, self.y * other)
        raise ValueError("Not an int or float type to multiply.")

    def __rmul__(self, other: Union[int, float]) -> "Point2":
        if np.isscalar(other):
            return Point2(self.x * other, self.y * other)
        raise ValueError("Not an int or float type to multiply.")

    def __truediv__(self, other: Union[int, float]) -> "Point2":
        if np.isscalar(other):
            if other == 0.0:
                raise ValueError("Cannot divide by zeros.")
            else:
                return Point2(self.x / other, self.y / other)
        raise ValueError("Not an int or float type to divide.")

    def __iadd__(self, other: "Point2") -> "Point2":
        if isinstance(other, Point2):
            self._x += other.x
            self._y += other.y
            return self
        else:
            raise ValueError("Not a Point2 type to add.")

    def __isub__(self, other: "Point2") -> "Point2":
        if isinstance(other, Point2):
            self._x -= other.x
            self._y -= other.y
            return self
        else:
            raise ValueError("Not a Point2 type to minus.")

    def __imul__(self, other: Union[int, float]) -> "Point2":
        if np.isscalar(other):
            self._x *= other
            self._y *= other
            return self
        else:
            raise ValueError("Not an int or float type to multiply.")

    def __itruediv__(self, other: Union[int, float]) -> "Point2":
        if np.isscalar(other):
            if other == 0.0:
                raise ValueError("Cannot divide by zeros.")
            else:
                self._x /= other
                self._y /= other
                return self
        else:
            raise ValueError("Not an int or float type to multiply.")

    def __neg__(self) -> "Point2":
        return self.inverse()

    def __str__(self) -> str:
        return "Point2{x: " + str(self.x) + ", y: " + str(self.y) + "}"

    def __eq__(self, other: "Point2") -> bool:
        if isinstance(other, Point2):
            return abs(self.x - other.x) < 1e-8 and abs(self.y - other.y) < 1e-8
        return False

    def __hash__(self) -> int:
        return hash((self.x, self.y))


class Rot2(object):
    dim = 1

    def __init__(self, theta: float = None) -> None:
        """
        Create a 2D rotation
        : theta is in radians
        :
        """
        # enforcing _theta in [-pi, pi] as a state
        self._theta = theta_to_pipi(none_to_zero(theta))
        # free theta
        # self._theta = none_to_zero(theta)

    @classmethod
    def by_degrees(cls, degrees: float = None) -> "Rot2":
        return cls(none_to_zero(degrees) * _DEG_TO_RAD_FACTOR)

    @classmethod
    def by_xy(cls, x: float = None, y: float = None) -> "Rot2":
        if x is None and y is None:
            return cls()
        else:
            return cls(math.atan2(none_to_zero(y), none_to_zero(x)))

    @classmethod
    def by_matrix(cls, matrix: np.ndarray = None) -> "Rot2":
        """
        : the maxtrix should be
        : np.array([ [self.cos, -self.sin],
                     [self.sin,  self.cos]])
        """
        return cls() if matrix is None else cls(math.atan2(matrix[1, 0],
                                                           matrix[0, 0]))

    @classmethod
    def exp_map(cls, vector: np.array = None) -> "Rot2":
        """
        :expect vector is a 1*1 array for 2D
        """
        if vector is None:
            return cls()
        assert len(vector) == 1
        return cls(vector[0])

    @staticmethod
    def dist(x1: np.ndarray, x2: np.ndarray):
        """chordal distance between x1 and x2 which are np.ndarray forms of Rot2"""
        return np.linalg.norm((Rot2.by_array(x1).inverse() *
                        Rot2.by_array(x2)).log_map())

    def log_map(self):
        """
        Logarithmic map
        """
        return np.array([self.theta])

    @property
    def theta(self) -> float:
        return self._theta

    @property
    def degrees(self) -> float:
        return self.theta * _RAD_TO_DEG_FACTOR

    @property
    def cos(self) -> float:
        return math.cos(self.theta)

    @property
    def sin(self) -> float:
        return math.sin(self.theta)

    @property
    def matrix(self):
        return np.array([[self.cos, -self.sin],
                         [self.sin, self.cos]])

    @property
    def dmatdth(self):
        return np.array([[-self.sin, -self.cos],
                         [self.cos, -self.sin]])

    def set_theta(self, theta: float = None) -> "Rot2":
        if theta is not None:
            assert np.isscalar(theta)
            self._theta = theta
        return self

    def bearing(self, global_pt: Point2):
        local_pt = self.unrotate_point(global_pt)
        return math.atan2(local_pt.y, local_pt.x)

    def inverse(self) -> "Rot2":
        return Rot2(-self.theta)

    def copy(self) -> "Rot2":
        return Rot2(self.theta)

    def transform_to(self, other: "Rot2") -> "Rot2":
        return other / self

    def rotate_point(self, local_pt: Point2):
        """
        get a point in world frame
        """
        return self * local_pt

    def unrotate_point(self, global_pt: Point2):
        """
        get a point in local frame
        """
        return self.inverse() * global_pt

    def __mul__(self, other: Union["Rot2", "Point2"]
                ) -> Union["Rot2", "Point2"]:
        if isinstance(other, Rot2):
            return Rot2(self.theta + other.theta)
        elif isinstance(other, Point2):
            x = self.cos * other.x - self.sin * other.y
            y = self.sin * other.x + self.cos * other.y
            return Point2(x, y)
        raise ValueError("Not a Point2 or Rot2 type to multiply.")

    def __truediv__(self, other: "Rot2") -> "Rot2":
        if isinstance(other, Rot2):
            return Rot2(self.theta - other.theta)
        raise ValueError("Not a Rot2 type to divide.")

    def __imul__(self, other: "Rot2") -> "Rot2":
        if isinstance(other, Rot2):
            self._theta += other.theta
            return self
        raise ValueError("Not a Rot2 type to multiply.")

    def __itruediv__(self, other: "Rot2") -> "Rot2":
        if isinstance(other, Rot2):
            self._theta -= other.theta
            return self
        raise ValueError("Not a Rot2 type to divide.")

    def __str__(self) -> str:
        string = "Rot2{theta: " + str(self.theta) + "}"
        return string

    def __eq__(self, other: "Rot2") -> bool:
        if isinstance(other, Rot2):
            return abs(self.theta - other.theta) < 1e-8
        return False

    def __hash__(self) -> int:
        return hash(self.theta)


class SE2Pose(object):
    rot_pi_2 = Rot2(np.pi / 2)
    dim = 3
    def __init__(self, x: float = None, y: float = None, theta: float = None
                 ) -> None:
        """
        Create a 2D pose
        : theta is in radians
        : _theta should be in [-pi, pi] as a state
        """
        self._point = Point2(x=x, y=y)
        self._rot = Rot2(theta=theta)
        self._gtsam_obj = gtsam.Pose2(x=self._point.x, y=self._point.y, theta=self._rot.theta)

    @classmethod
    def by_pt_rt(cls, pt: Point2 = None, rt: Rot2 = None) -> "SE2Pose":
        return cls(pt.x, pt.y, rt.theta)

    @classmethod
    def by_matrix(cls, matrix: np.ndarray = None) -> "SE2Pose":
        """
        : the maxtrix should be
        : np.array([ [self.cos, -self.sin, x],
                     [self.sin,  self.cos, y],
                     [       0,         0, 1]])
        """
        if matrix is None:
            return cls()
        else:
            assert isinstance(matrix, np.ndarray)
            pt = Point2.by_array(matrix[0:2, 2])
            rt = Rot2.by_matrix(matrix[0:2, 0:2])
            return SE2Pose.by_pt_rt(pt=pt, rt=rt)

    @classmethod
    def by_exp_map(cls, vector: np.array = None) -> "SE2Pose":
        """
        :return a 1*3 array for 2D
        """
        if vector is None:
            return cls()
        assert len(vector) == 3
        w = vector[2]
        if abs(w) < 1e-10:
            return SE2Pose(vector[0],
                           vector[1],
                           w)
        else:
            pt = Point2(vector[0], vector[1])
            w_rot = Rot2(w)
            v_ortho = SE2Pose.rot_pi_2 * pt
            t = (v_ortho - w_rot.rotate_point(v_ortho)) / w
            return cls(x=t.x, y=t.y, theta=w)

    @classmethod
    def by_array(cls, other: Union[List[float], Tuple[float], np.ndarray]
                 ) -> "SE2Pose":
        return cls(other[0], other[1], other[2])

    @staticmethod
    def dist(x1: np.ndarray, x2: np.ndarray):
        """chordal distance between x1 and x2 which are np.ndarray forms of SE2 poses"""
        return np.linalg.norm((SE2Pose.by_array(x1).inverse() *
                        SE2Pose.by_array(x2)).log_map())

    # @property
    # def dim(self):
    #     return self._rot.dim + self._point.dim

    @property
    def theta(self):
        return self._rot.theta

    @property
    def x(self):
        return self._point.x

    @property
    def y(self):
        return self._point.y

    @property
    def rotation(self):
        return self._rot

    @property
    def translation(self):
        return self._point

    @property
    def matrix(self):
        r_c = self._rot.cos
        r_s = self._rot.sin
        x = self._point.x
        y = self._point.y
        return np.array([[r_c, -r_s, x],
                         [r_s, r_c, y],
                         [0, 0, 1]])

    @property
    def array(self):
        return np.array([self.x, self.y, self.theta])

    def log_map(self):
        # TODO: simplify this closed form
        r = self._rot
        t = self._point
        w = r.theta
        if (abs(w) < 1e-10):
            return np.array([t.x, t.y, w])
        else:
            c_1 = r.cos - 1.0
            s = r.sin
            det = c_1 * c_1 + s * s
            p = SE2Pose.rot_pi_2 * (r.unrotate_point(t) - t)
            v = (w / det) * p
            return np.array([v.x, v.y, w])

    def grad_x_logmap(self):
        """
        d(v1, v2, alpha)/d(x, y, theta) where (v1, v2, alpha) is a tangent space vector
        """
        return gtsam.Pose2.LogmapDerivative(v=self._gtsam_obj)

    def det_grad_x_logmap(self):
        # note this is just the determinant, not absolute values
        if (abs(self.theta) < 1e-5):
            return 1.0
        return (self.theta**2/4/(np.sin(self.theta/2)**2))

    def grad_x_det_grad_x_logmap(self):
        # note this is just the determinant, not absolute values
        if (abs(self.theta) < 1e-5):
            return np.array([.0,.0,.0])
        h = self.theta / 2.0
        dfdth = h/np.sin(h)**2 - np.cos(h) * h**2 / np.sin(h)**3
        return np.array([.0,.0, dfdth])

    def grad_xi_expmap(self):
        """
        Informally, d(x, y, theat)/d(v1, v2, alpha) where (v1, v2, alpha) is a tangent space vector
        J_r on page 36 of
        Stochastic Models, Information Theory, and Lie Groups: Volume 2 Analytic Methods and Modern Applications. G. S. Chirikjian. Boston: Birkhäuser. Nov. 2011.
        """
        return gtsam.Pose2.ExpmapDerivative(v=self.log_map())

    def range_and_bearing(self, pt: Point2):
        d = pt - self._point
        range = d.norm
        bearing = self._rot.bearing(d)
        return range, bearing

    def inverse(self) -> "SE2Pose":
        inv_t = - (self._rot.unrotate_point(self._point))
        return SE2Pose.by_pt_rt(pt=inv_t, rt=self._rot.inverse())

    def copy(self) -> "SE2Pose":
        return SE2Pose(x=self.x, y=self.y, theta=self.theta)

    def transform_to(self, other):
        """
        transform from self to other
        """
        return other / self

    def transform_point(self, local_point: Point2):
        """
        get a point in world frame
        """
        return self * local_point

    def __mul__(self, other):
        if isinstance(other, SE2Pose):
            r = self._rot * other.rotation
            t = self._point + self._rot * other.translation
            return SE2Pose.by_pt_rt(pt=t, rt=r)
        if isinstance(other, Point2):
            return self._rot * other + self._point
        raise ValueError("Not a Point2 or Pose2 type to multiply.")

    def __truediv__(self, other):
        if isinstance(other, SE2Pose):
            return self * other.inverse()
        raise ValueError("Not a Pose2 type to divide.")

    def __imul__(self, other):
        if isinstance(other, SE2Pose):
            pos = self * other
            self._point = self._point.set_x_y(x=pos.x,
                                              y=pos.y)
            self._rot = self._rot.set_theta(pos.theta)
            return self
        raise ValueError("Not a Pose2 type to multiply.")

    def __itruediv__(self, other):
        if isinstance(other, SE2Pose):
            pos = self / other
            self._point = self._point.set_x_y(x=pos.x,
                                              y=pos.y)
            self._rot = self._rot.set_theta(pos.theta)
            return self
        raise ValueError("Not a Pose2 type to divide.")

    def __str__(self) -> str:
        string = "Pose2{" + \
                 self._point.__str__() + \
                 ", " + \
                 self._rot.__str__() + \
                 "}"
        return string

    def __eq__(self, other: "SE2Pose") -> bool:
        if isinstance(other, SE2Pose):
            return abs(self._rot.theta - other.theta) < 1e-8 and \
                   abs(self._point.x - other.x) < 1e-8 and \
                   abs(self._point.y - other.y) < 1e-8
        return False

    def __hash__(self):
        return hash((self._point.x,
                     self._point.y,
                     self._rot.theta))
