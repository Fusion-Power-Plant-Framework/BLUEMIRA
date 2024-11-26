# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Grid object and operations for equilibria.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba as nb
import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.equilibria.constants import MIN_N_DISCR, X_AXIS_MIN
from bluemira.equilibria.error import EquilibriaError
from bluemira.geometry.coordinates import get_area_2d, get_centroid_2d

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy.typing as npt
    from eqdsk import EQDSKInterface


__all__ = ["Grid", "integrate_dx_dz", "revolved_volume", "volume_integral"]


class Grid:
    """
    A rectangular Grid object with regular rectangular cells for use in finite
    difference calculations.

    Parameters
    ----------
    x_min:
        Minimum x grid coordinate (>0) [m]
    x_max:
        Maximum x grid coordinate [m]
    z_min:
        Minimum z grid coordinate [m]
    z_max:
        Maximum z grid coordinate [m]
    nx:
        Number of x grid points
    nz:
        Number of z grid points
    """

    __slots__ = (
        "bounds",
        "dx",
        "dz",
        "edges",
        "nx",
        "nz",
        "step",
        "x",
        "x_1d",
        "x_max",
        "x_mid",
        "x_min",
        "x_size",
        "z",
        "z_1d",
        "z_max",
        "z_mid",
        "z_min",
        "z_size",
    )

    def __init__(
        self, x_min: float, x_max: float, z_min: float, z_max: float, nx: int, nz: int
    ):
        if x_min == x_max or z_min == z_max:
            raise EquilibriaError("Invalid Grid dimensions specified.")

        if x_min > x_max:
            bluemira_warn(
                f"x_min should be < x_max {x_min:.2f} > {x_max:.2f}. Switching x_min and"
                " x_max."
            )
            x_min, x_max = x_max, x_min

        if z_min > z_max:
            bluemira_warn(
                f"z_min should be < z_max {z_min:.2f} > {z_max:.2f}. Switching z_min and"
                " z_max."
            )
            z_min, z_max = z_max, z_min

        if x_min <= 0:  # Cannot calculate flux on machine axis - (divide by 0)
            x_min = X_AXIS_MIN

        if nx < MIN_N_DISCR:
            bluemira_warn(
                f"Insufficient nx discretisation: {nx}, setting to {MIN_N_DISCR}."
            )
            nx = MIN_N_DISCR

        if nz < MIN_N_DISCR:
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
        self.z_size = abs(z_max - z_min)
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
        self.edges = np.concatenate([
            [(x, 0) for x in range(nx)],  # Grid edges
            [(x, nz - 1) for x in range(nx)],
            [(0, z) for z in range(nz)],
            [(nx - 1, z) for z in range(nz)],
        ])
        self.step = abs(self.x_1d[0] - self.x_1d[1]) * abs(self.z_1d[0] - self.z_1d[1])

    @classmethod
    def from_eqdict(cls, e) -> Grid:
        """
        Initialise a Grid object from an EQDSK dictionary.

        Parameters
        ----------
        e: dict
            EQDSK dictionary
        """  # noqa: DOC201
        return cls(
            e["xgrid1"],
            e["xgrid1"] + e["xdim"],
            e["zmid"] - 0.5 * e["zdim"],
            e["zmid"] + 0.5 * e["zdim"],
            e["nx"],
            e["nz"],
        )

    @classmethod
    def from_eqdsk(cls, e: EQDSKInterface) -> Grid:
        """
        Initialise a Grid object from an EQDSKInterface.

        Parameters
        ----------
        e:

        """  # noqa: DOC201
        return cls(
            e.xgrid1,
            e.xgrid1 + e.xdim,
            e.zmid - 0.5 * e.zdim,
            e.zmid + 0.5 * e.zdim,
            e.nx,
            e.nz,
        )

    def point_inside(self, x: float | Iterable[float], z: float | None = None) -> bool:
        """
        Determine if a point is inside the rectangular grid (includes edges).

        Parameters
        ----------
        x:
            The x coordinate of the point. Or the 2-D point.
        z:
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

    def distance_to(
        self, x: float | Iterable[float], z: float | None = None
    ) -> npt.NDArray[np.float64]:
        """
        Get the distances of a point to the edges of the Grid.

        Parameters
        ----------
        x:
            The x coordinate of the point. Or the 2-D point.
        z:
            The z coordinate of the point

        Returns
        -------
        distances:
            Distances to the edges of the Grid.
        """
        if z is None:
            x, z = x
        return np.abs([
            x - self.x_min,
            x - self.x_max,
            z - self.z_min,
            z - self.z_max,
        ])

    def plot(self, ax=None, **kwargs):
        """
        Plot the Grid object onto an ax.

        Returns
        -------
        :
            The plot axis
        """
        from bluemira.equilibria.plotting import GridPlotter  # noqa: PLC0415

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
    return np.trapz(np.trapz(func)) * d_x * d_z  # noqa: NPY201


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
