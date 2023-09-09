from typing import Hashable
from enum import Enum
from typing import List, Set

import gtsam
import numpy as np


class VariableType(Enum):
    Pose = "Pose"
    Landmark = "Landmark"
    Measurement = "Measurement"


class Variable(object):
    def __init__(self, name: Hashable, dim: int,
                 variable_type: VariableType = VariableType.Pose,
                 rotational_dims: Set[int] = None) -> None:
        """
        Create a Variable object that is uniquely identifiable by its name
        :param name: the identifier of the variable
        :param dim: the dimensionality of the variable
        :param variable_type: whether the variable is a pose, a landmark, or a
            measurement
        :param rotational_dims: dimensions that represent rotational, instead
            of translational components
        """
        self._type = variable_type
        if dim <= 0:
            raise ValueError("Dimensionality must be positive")
        self._dim = dim
        self._name = name
        if not rotational_dims:
            self._rotational_dims = {}
        elif not 0 <= min(rotational_dims) <= max(rotational_dims) < dim:
            raise ValueError("rotational_dims is incorrect")
        else:
            self._rotational_dims = rotational_dims
        if name[0] != 'O':
            self.key = gtsam.Symbol(name[0], int(name[1:])).key()
        else:
            self.key = None

    @classmethod
    def construct_from_text(cls, line: str) -> "Variable":
        line = line.strip().split()
        cls_type = eval(line[2] + "Variable")
        var_type = eval("VariableType." + line[1])
        return cls_type(name=line[3], variable_type=var_type)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> Hashable:
        return self._name

    def __copy__(self) -> "Variable":
        """
        Make a copy of the Variable object with the same name
        :return: the copy of the current Variable object
        """
        return Variable(name=self._name, dim=self._dim)

    def __str__(self) -> str:
        return " ".join(["Variable", self.type.value,
                         self.__class__.__name__.replace("Variable", ""),
                         str(self.name)])

    def __hash__(self) -> int:
        return hash(self._name)

    def __eq__(self, other: "Variable") -> bool:
        """
        Two variables are the same if they have the same names
        """
        return self._name == other._name

    def __ne__(self, other: "Variable") -> bool:
        """
        Two variables are different if they have different names
        """
        return self._name != other._name

    def __le__(self, other: "Variable") -> bool:
        """
        Compare two variables by their names
        """
        return self._name <= other._name

    def __lt__(self, other: "Variable") -> bool:
        """
        Compare two variables by their names
        """
        return self._name < other._name

    def __ge__(self, other: "Variable") -> bool:
        """
        Compare two variables by their names
        """
        return self._name >= other._name

    def __gt__(self, other: "Variable") -> bool:
        """
        Compare two variables by their names
        """
        return self._name > other._name

    @property
    def translational_dim(self) -> int:
        return self._dim - len(self._rotational_dims)

    @property
    def rotational_dim(self) -> int:
        return len(self._rotational_dims)

    @property
    def circular_dim_list(self) -> List[bool]:
        """
        A list indicating whether a dimension is periodic (True) or Euclidean
            (False)
        convention of dim order is translation first like x y z r p y
        """
        return [True if i in self._rotational_dims else False
                for i in range(self.dim)]

    @property
    def type(self) -> VariableType:
        return self._type

    @property
    def t_dim_indices(self):
        """
        indices of translational dim;
        convention of dim order is translation first like x y z r p y
        """
        return list(range(self.translational_dim))
    @property
    def R_dim_indices(self):
        """
        indices of rotational dim;
        convention of dim order is translation first like x y z r p y
        """
        return list(range(self.dim))[self.translational_dim:]

    @staticmethod
    def file2vars(order_file: str, pose_space: str = "SE2", lmk_prefix="L"):
        var_list = []
        order = np.loadtxt(order_file, dtype='str',ndmin=1)
        for var in order:
            if var[0] == lmk_prefix:
                var_list.append(R2Variable(name=var, variable_type=VariableType.Landmark))
            else:
                if pose_space == "SE2":
                    var_list.append(SE2Variable(name=var, variable_type=VariableType.Pose))
                elif pose_space == "R2":
                    var_list.append(R2Variable(name=var, variable_type=VariableType.Pose))
        return var_list

class R2Variable(Variable):
    def __init__(self, name: Hashable,
                 variable_type: VariableType = VariableType.Pose) -> None:
        super().__init__(name=name, dim=2, variable_type=variable_type,
                         rotational_dims=None)


class R1Variable(Variable):
    def __init__(self, name: Hashable,
                 variable_type: VariableType = VariableType.Pose) -> None:
        super().__init__(name=name, dim=1, variable_type=variable_type,
                         rotational_dims=None)

class Bearing2DVariable(Variable):
    def __init__(self, name: Hashable,
                 variable_type: VariableType = VariableType.Measurement) -> None:
        super().__init__(name=name, dim=1, variable_type=variable_type,
                         rotational_dims={0})

class Bearing3DVariable(Variable):
    def __init__(self, name: Hashable,
                 variable_type: VariableType = VariableType.Measurement) -> None:
        super().__init__(name=name, dim=2, variable_type=variable_type,
                         rotational_dims={0,1})

class SE2Variable(Variable):
    def __init__(self, name: Hashable,
                 variable_type: VariableType = VariableType.Pose) -> None:
        super().__init__(name=name, dim=3, variable_type=variable_type,
                         rotational_dims={2})

class SE3Variable(Variable):
    def __init__(self, name: Hashable,
                 variable_type: VariableType = VariableType.Pose) -> None:
        super().__init__(name=name, dim=6, variable_type=variable_type,
                         rotational_dims={3,4,5})

class R3Variable(Variable):
    def __init__(self, name: Hashable,
                 variable_type: VariableType = VariableType.Pose) -> None:
        super().__init__(name=name, dim=3, variable_type=variable_type,
                         rotational_dims=None)
