# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
Coil and coil grouping objects
"""

from enum import Enum, EnumMeta, auto
from typing import Iterable, Optional, Union

# import matplotlib.pyplot as plt
import numpy as np

from bluemira.equilibria.coils._field import CoilFieldsMixin
from bluemira.equilibria.constants import I_MIN, NBTI_B_MAX, NBTI_J_MAX
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.plotting import CoilPlotter, CoilSetPlotter
from bluemira.utilities.tools import is_num

# from scipy.interpolate import RectBivariateSpline


__all__ = ["CoilType", "Coil"]


class CoilTypeEnumMeta(EnumMeta):
    """
    Allow override of KeyError error string
    """

    def __getitem__(self, name):
        try:
            return super().__getitem__(name)
        except KeyError:
            raise KeyError(f"Unknown CoilType {name}") from None


class CoilType(Enum, metaclass=CoilTypeEnumMeta):
    """
    CoilType Enum
    """

    PF = auto()
    CS = auto()
    NONE = auto()


__ITERABLE_FLOAT = Union[float, Iterable[float]]
__ITERABLE_COILTYPE = Union[str, CoilType, Iterable[Union[str, CoilType]]]
__ANY_ITERABLE = Union[__ITERABLE_COILTYPE, __ITERABLE_FLOAT]


class CoilNumber:
    """
    Coil naming-numbering utility class. Coil naming convention is not enforced here.
    """

    __PF_counter: int = 1
    __CS_counter: int = 1
    __no_counter: int = 1

    @staticmethod
    def generate(ctype: CoilType) -> int:
        """
        Generate a coil name based on its type and indexing if specified. If no index is
        specified, an encapsulated global counter assigns an index.

        Parameters
        ----------
        coil: Any
            Object to name

        Returns
        -------
        name: str
            Coil name
        """
        if ctype == CoilType.NONE:
            idx = CoilNumber.__no_counter
            CoilNumber.__no_counter += 1
        elif ctype == CoilType.CS:
            idx = CoilNumber.__CS_counter
            CoilNumber.__CS_counter += 1
        elif ctype == CoilType.PF:
            idx = CoilNumber.__PF_counter
            CoilNumber.__PF_counter += 1

        return idx


class Coil(CoilFieldsMixin):
    def __init__(
        self,
        x: float,
        z: float,
        dx: float = None,
        dz: float = None,
        name: Optional[str] = None,
        ctype: Union[str, CoilType] = CoilType.NONE,
        current: float = 0,
        j_max: float = np.nan,
        b_max: float = np.nan,
        discretisation: float = 1,
        n_turns: int = 1,
    ):
        self._dx = None
        self._dz = None
        self._discretisation = 1
        self._flag_sizefix = None

        self.x = x
        self.z = z
        self.dx = dx
        self.dz = dz
        self.discretisation = discretisation
        self.current = current
        self.j_max = j_max
        self.b_max = b_max
        self.ctype = ctype
        self.name = name
        self.n_turns = n_turns

        self._number = CoilNumber.generate(self.ctype)
        if self.name is None:
            self.name = f"{self._ctype.name}_{self._number}"

        # check if dx and not dz set
        # check of j max set
        self._validate_size()
        if not self._flag_sizefix and None in (self.dx, self.dz):
            self._dx, self._dz = 0, 0
            self._discretise()

    def __repr__(self):
        return (
            f"{type(self).__name__}({self.name} ctype={self.ctype.name} x={self.x:.2g}"
            f" z={self.z:.2g} dx={self.dx:.2g} dz={self.dz:.2g} current={self.current:.2g}"
            f" j_max={self.j_max:.2g} b_max={self.b_max:.2g}"
            f" discretisation={self.discretisation:.2g})"
        )

    def n_coils(self):
        return 1

    @property
    def x(self) -> float:
        return self._x

    @property
    def z(self) -> float:
        return self._z

    @property
    def ctype(self) -> CoilType:
        return self._ctype

    @property
    def dx(self) -> float:
        return self._dx

    @property
    def dz(self) -> float:
        return self._dz

    @property
    def current(self):
        return self._current

    @property
    def j_max(self) -> float:
        return self._j_max

    @property
    def b_max(self) -> float:
        return self._b_max

    @property
    def discretisation(self) -> float:
        return self._discretisation

    @property
    def area(self) -> np.ndarray:
        """
        The cross-sectional area of the coil

        Returns
        -------
        area: float
            The cross-sectional area of the coil [m^2]
        """
        return 4 * self.dx * self.dz

    @property
    def volume(self) -> np.ndarray:
        """
        The volume of the coil

        Returns
        -------
        volume: float
            The volume of the coil [m^3]
        """
        return self.area * 2 * np.pi * self.x

    @property
    def x_boundary(self):
        if getattr(self, "_x_boundary") is not None:
            return self._x_boundary
        return self._make_boundary(self.x, self.z, self.dx, self.dz)[0]

    @property
    def z_boundary(self):
        if getattr(self, "_z_boundary") is not None:
            return self._z_boundary
        return self._make_boundary(self.x, self.z, self.dx, self.dz)[1]

    @x.setter
    def x(self, value: float):
        self._x = float(value)
        if None not in (self.dx, self.dz):
            self._discretise()
            self._set_coil_attributes()

    @z.setter
    def z(self, value: float):
        self._z = float(value)
        if None not in (self.dx, self.dz):
            self._discretise()
            self._set_coil_attributes()

    @ctype.setter
    def ctype(self, value: Union[str, CoilType]):
        self._ctype = (
            value
            if isinstance(value, CoilType)
            else CoilType[value[0] if isinstance(value, np.ndarray) else value]
        )

    @dx.setter
    def dx(self, value: float):
        self._dx = None if value is None else float(value)
        if None not in (self.dx, self.dz):
            self._discretise()
            self._set_coil_attributes()

    @dz.setter
    def dz(self, value: float):
        self._dz = None if value is None else float(value)
        if None not in (self.dx, self.dz):
            self._discretise()
            self._set_coil_attributes()

    @current.setter
    def current(self, value: Union[str, CoilType]):
        self._current = float(value)

    @j_max.setter
    def j_max(self, value: float):
        self._j_max = float(value)
        if None not in (self.dx, self.dz):
            self.resize()

    @b_max.setter
    def b_max(self, value: float):
        self._b_max = float(value)

    @discretisation.setter
    def discretisation(self, value: float):
        self._discretisation = float(value)
        self._discretise()

    def assign_material(
        self,
        j_max=NBTI_J_MAX,
        b_max=NBTI_B_MAX,
    ) -> None:
        """
        Assigns EM material properties to coil

        Parameters
        ----------
        j_max: float
            Overwrite default constant material max current density [A/m^2]
        b_max: float
            Overwrite default constant material max field [T]

        """
        self.j_max = j_max
        self.b_max = b_max

    def _discretise(self):
        """
        Discretise a coil for greens function magnetic field calculations

        Notes
        -----
        Only discretisation method currently implemented is rectangular fraction

        Possible improvement: multiple discretisations for different coils

        """
        self._quad_x = np.array([self.x])
        self._quad_z = np.array([self.z])
        self._quad_dx = np.array([self.dx])
        self._quad_dz = np.array([self.dz])
        self._quad_weighting = np.ones_like(self._quad_x)
        self._einsum_str = "...j, ...j -> ..."

        if self.discretisation < 1:
            # How fancy do we want the mesh or just smaller rectangles?
            self._rectangular_discretisation()

    def _validate_size(self):
        dx_spec = is_num(self.dx)
        dz_spec = is_num(self.dz)
        dxdz_spec = dx_spec and dz_spec

        if (dx_spec ^ dz_spec) and not dxdz_spec:
            # Check that we don't have dx = None and dz = float or vice versa
            raise EquilibriaError("Must specify either dx and dz or neither.")
        if dxdz_spec:
            # If dx and dz are specified, we presume the coil size should
            # remain fixed
            if not self._flag_sizefix:
                self._flag_sizefix = True

            self._set_coil_attributes()
            self._discretise()
        else:
            if not is_num(self.j_max):
                # Check there is a viable way to size the coil
                raise EquilibriaError("Must specify either dx and dz or j_max.")

            if self._flag_sizefix:
                self._flag_sizefix = False

    def _set_coil_attributes(self):
        self._current_radius = 0.5 * np.hypot(self.dx, self.dz)
        self._x_boundary, self._z_boundary = self._make_boundary(
            self.x, self.z, self.dx, self.dz
        )

    def _rectangular_discretisation(self):
        """
        Discretise a coil into smaller rectangles based on fraction of
        coil dimensions

        Parameters
        ----------
        d_coil: float
            Target discretisation fraction

        Returns
        -------
        weighting: np.ndarray

        """
        nx = np.maximum(1, np.ceil(self.dx * 2 / self.discretisation))
        nz = np.maximum(1, np.ceil(self.dz * 2 / self.discretisation))

        if not nx * nz == 1:
            sc_dx, sc_dz = self.dx / nx, self.dz / nz

            # Calculate sub-coil centroids
            x_sc = (self.x - self.dx) + sc_dx * np.arange(1, 2 * nx, 2)
            z_sc = (self.z - self.dz) + sc_dz * np.arange(1, 2 * nz, 2)
            x_sc, z_sc = np.meshgrid(x_sc, z_sc)

            self._quad_x = x_sc.flatten()
            self._quad_z = z_sc.flatten()
            self._quad_dx = np.ones(x_sc.size) * sc_dx
            self._quad_dz = np.ones(x_sc.size) * sc_dz

            self._quad_weighting = np.ones(x_sc.size) / x_sc.size

    def fix_size(self):
        """
        Fixes the size of all coils
        """
        self._flag_sizefix = True

    def resize(self, current: Optional[float] = None):
        if not self._flag_sizefix:
            # Adjust the size of the coil
            self.dx, self.dz = self._make_size(current)
            self._set_coil_attributes()

    def _make_size(self, current=None):
        """
        Size the coil based on a current and a current density.
        """
        if current is None:
            current = self.current
        if not np.isnan(self.j_max):
            half_width = 0.5 * np.sqrt(abs(current) / self.j_max)
            return half_width, half_width
        else:
            return self.dx, self.dz

    @staticmethod
    def _make_boundary(x_c: float, z_c: float, dx: float, dz: float):
        """
        Makes the coil boundary vectors

        Parameters
        ----------
        x_c: float
            x coordinate of centre
        z_c: float
            z coordinate of centre
        dx: float
            dx of coil
        dz: float
            dz of coil

        Returns
        -------
        x_boundary, z_boundary: np.array

        Note
        ----
        Only rectangular coils

        """
        xx, zz = (np.ones((4, 1)) * x_c).T, (np.ones((4, 1)) * z_c).T
        x_boundary = xx + (dx * np.array([-1, 1, 1, -1])[:, None]).T
        z_boundary = zz + (dz * np.array([-1, -1, 1, 1])[:, None]).T
        return x_boundary, z_boundary
