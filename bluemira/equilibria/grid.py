# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
Grid object and operations for equilibria.
"""
import numba as nb
import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.equilibria.constants import MIN_N_DISCR, X_AXIS_MIN
from bluemira.equilibria.error import EquilibriaError
from bluemira.geometry.coordinates import get_area_2d, get_centroid_2d

__all__ = ["Grid", "integrate_dx_dz", "volume_integral", "revolved_volume"]


class Grid:
    """
    A rectangular Grid object with regular rectangular cells for use in finite
    difference calculations.

    Parameters
    ----------
    x_min: float > 0
        Minimum x grid coordinate [m]
    x_max: float
        Maximum x grid coordinate [m]
    z_min: float
        Minimum z grid coordinate [m]
    z_max: float
        Maximum z grid coordinate [m]
    nx: int
        Number of x grid points
    nz: int
        Number of z grid points
    """

    __slots__ = [
        "x_min",
        "x_max",
        "x_size",
        "x_mid",
        "x_1d",
        "x",
        "z_min",
        "z_max",
        "z_size",
        "z_mid",
        "z_1d",
        "z",
        "dx",
        "dz",
        "nx",
        "nz",
        "bounds",
        "edges",
    ]

    def __init__(self, x_min, x_max, z_min, z_max, nx, nz):
        if x_min == x_max or z_min == z_max:
            raise EquilibriaError("Invalid Grid dimensions specified.")

        if x_min > x_max:
            print("")  # stdout flusher
            bluemira_warn(
                f"x_min should be < x_max {x_min:.2f} > {x_max:.2f}. Switching x_min and"
                " x_max."
            )
            x_min, x_max = x_max, x_min

        if z_min > z_max:
            print("")  # stdout flusher
            bluemira_warn(
                f"z_min should be < z_max {z_min:.2f} > {z_max:.2f}. Switching z_min and"
                " z_max."
            )
            z_min, z_max = z_max, z_min

        if x_min <= 0:  # Cannot calculate flux on machine axis - (divide by 0)
            x_min = X_AXIS_MIN

        if nx < MIN_N_DISCR:
            print("")  # stdout flusher
            bluemira_warn(
                f"Insufficient nx discretisation: {nx}, setting to {MIN_N_DISCR}."
            )
            nx = MIN_N_DISCR

        if nz < MIN_N_DISCR:
            print("")  # stdout flusher
            bluemira_warn(
                f"Insufficient nx discretisation: {nz}, setting to {MIN_N_DISCR}."
            )
            nz = MIN_N_DISCR

        self.x_min = x_min
        self.x_max = x_max
        self.x_size = x_max - x_min
        self.x_mid = (x_max - x_min) / 2
        self.z_min = z_min
        self.z_max = z_max
        self.z_size = abs(z_max) + abs(z_min)
        self.z_mid = z_min + (z_max - z_min) / 2
        self.x_1d = np.linspace(x_min, x_max, nx)
        self.z_1d = np.linspace(z_min, z_max, nz)
        self.x, self.z = np.meshgrid(self.x_1d, self.z_1d, indexing="ij")
        self.dx = np.diff(self.x_1d[:2])[0]  # Grid sizes
        self.dz = np.diff(self.z_1d[:2])[0]
        self.nx, self.nz = nx, nz
        self.bounds = [
            [x_min, x_max, x_max, x_min, x_min],  # Grid corners
            [z_min, z_min, z_max, z_max, z_min],
        ]
        self.edges = np.concatenate(
            [
                [(x, 0) for x in range(nx)],  # Grid edges
                [(x, nz - 1) for x in range(nx)],
                [(0, z) for z in range(nz)],
                [(nx - 1, z) for z in range(nz)],
            ]
        )

    @classmethod
    def from_eqdict(cls, e):
        """
        Initialise a Grid object from an EQDSK dictionary.

        Parameters
        ----------
        e: dict
            EQDSK dictionary
        """
        return cls(
            e["xgrid1"],
            e["xgrid1"] + e["xdim"],
            e["zmid"] - 0.5 * e["zdim"],
            e["zmid"] + 0.5 * e["zdim"],
            e["nx"],
            e["nz"],
        )

    @classmethod
    def from_eqdsk(cls, e):
        """
        Initialise a Grid object from an EQDSKInterface.

        Parameters
        ----------
        e: EQDSKInterface

        """
        return cls(
            e.xgrid1,
            e.xgrid1 + e.xdim,
            e.zmid - 0.5 * e.zdim,
            e.zmid + 0.5 * e.zdim,
            e.nx,
            e.nz,
        )

    def point_inside(self, x, z=None):
        """
        Determine if a point is inside the rectangular grid (includes edges).

        Parameters
        ----------
        x: Union[float, Iterable]
            The x coordinate of the point. Or the 2-D point.
        z: Optional[float]
            The z coordinate of the point

        Returns
        -------
        inside: bool
            Whether or not the point is inside the grid
        """
        if z is None:
            x, z = x
        return (
            (x >= self.x_min)
            and (x <= self.x_max)
            and (z >= self.z_min)
            and (z <= self.z_max)
        )

    def distance_to(self, x, z=None):
        """
        Get the distances of a point to the edges of the Grid.

        Parameters
        ----------
        x: Union[float, Iterable]
            The x coordinate of the point. Or the 2-D point.
        z: Optional[float]
            The z coordinate of the point

        Returns
        -------
        distances: np.ndarray
            Distances to the edges of the Grid.
        """
        if z is None:
            x, z = x
        return np.abs(
            [
                x - self.x_min,
                x - self.x_max,
                z - self.z_min,
                z - self.z_max,
            ]
        )

    def plot(self, ax=None, **kwargs):
        """
        Plot the Grid object onto an ax.
        """
        from bluemira.equilibria.plotting import GridPlotter

        return GridPlotter(self, ax=ax, **kwargs)


@nb.jit(nopython=True, cache=True)
def integrate_dx_dz(func, d_x, d_z):
    """
    Get the double-integral of a function over the space.

    \t:math:`\\int_Z\\int_X f(x, z) dXdZ`

    Parameters
    ----------
    func: np.array(N, M)
        A 2-D function map
    d_x: float
        The discretisation size in the X coordinate
    d_z: float
        The discretisation size in the Z coordinate

    Returns
    -------
    integral: float
        The integral value of the field in 2-D
    """
    return np.trapz(np.trapz(func)) * d_x * d_z


@nb.jit(nopython=True, cache=True)
def volume_integral(func, x, d_x, d_z):
    """
    Calculate the volume integral of a field in quasi-cylindrical coordinates.

    Parameters
    ----------
    func: 2-D np.array
        Field to volume integrate
    x: 2-D np.array
        X coordinate grid
    d_x: float
        Grid X cell size
    d_z: float
        Grid Z cell size

    Returns
    -------
    integral: float
        The integral value of the field in 3-D space
    """
    return 2 * np.pi * integrate_dx_dz(func * x, d_x, d_z)


def revolved_volume(x, z):
    """
    Calculate the revolved volume of a set of x, z coordinates. Revolution about
    [0, 0, 1].

    Parameters
    ----------
    x: np.array
        The x coordinates
    z: np.array
        The z coordinates

    Returns
    -------
    volume: float
        The volume of the revolved x, z coordinates
    """
    area = get_area_2d(x, z)
    cx, _ = get_centroid_2d(x, z)
    return 2 * np.pi * cx * area
