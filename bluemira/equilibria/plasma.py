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
Representation of the plasma
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from bluemira.equilibria.grid import Grid

import numpy as np
from scipy.interpolate import RectBivariateSpline

from bluemira.equilibria.constants import J_TOR_MIN
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.plotting import PlasmaCoilPlotter
from bluemira.magnetostatics.greens import greens_Bx, greens_Bz, greens_psi


def treat_xz_array(func):
    """
    Decorator for handling array calls to PlasmaCoil methods.
    """

    def wrapper(self, x=None, z=None):
        if x is None or z is None:
            if z is None and x is None:
                return func(self, x, z)
            raise EquilibriaError("Only one of x and z specified.")

        x = np.array(x)
        z = np.array(z)
        if x.shape != z.shape:
            raise EquilibriaError("x and z arrays of different dimension.")

        values = np.zeros(x.size)

        for i, (xx, zz) in enumerate(zip(x.flat, z.flat)):
            values[i] = func(self, xx, zz)

        return values.reshape(x.shape)

    return wrapper


class PlasmaCoil:
    """
    PlasmaCoil object for finite difference representation of toroidal current
    carrying plasma.

    Parameters
    ----------
    plasma_psi:
        Psi contribution from the plasma on the grid
    j_tor:
        Toroidal current density distribution from the plasma on the grid
    grid:
        Grid object on which the finite difference representation of the plasma should be
        constructed

    Notes
    -----
    Uses direct summing of Green's functions to avoid SIGKILL and MemoryErrors
    when using very dense grids (e.g. CREATE).
    """

    def __init__(self, plasma_psi: np.ndarray, j_tor: Optional[np.ndarray], grid: Grid):
        self._grid = grid
        self._set_j_tor(j_tor)
        self._set_funcs(plasma_psi)

    def _set_j_tor(self, j_tor: Optional[np.ndarray]):
        self._j_tor = j_tor
        if j_tor is not None:
            self._ii, self._jj = np.nonzero(j_tor > J_TOR_MIN)
        else:
            self._ii, self._jj = None, None

    def _set_funcs(self, plasma_psi: np.ndarray):
        self._plasma_psi = plasma_psi
        self._psi_func = RectBivariateSpline(
            self._grid.x[:, 0], self._grid.z[0, :], plasma_psi
        )
        self._plasma_Bx = self._Bx_func(self._grid.x, self._grid.z)
        self._plasma_Bz = self._Bz_func(self._grid.x, self._grid.z)
        self._plasma_Bp = np.hypot(self._plasma_Bx, self._plasma_Bz)

    def _Bx_func(self, x, z):
        return -self._psi_func(x, z, dy=1, grid=False) / x

    def _Bz_func(self, x, z):
        return self._psi_func(x, z, dx=1, grid=False) / x

    def _check_in_grid(self, x, z):
        return self._grid.point_inside(x, z)

    def _convolve(self, func, x, z):
        """
        Map a Green's function across the grid at a point, without crashing or
        running out of memory.
        """
        if self._j_tor is None:
            raise EquilibriaError(
                "Cannot calculate value off grid; there is no known toroidal current"
                " distribution."
            )

        array = np.zeros_like(x, dtype=float)
        for i, j in zip(self._ii, self._jj):
            current = self._j_tor[i, j] * self._grid.dx * self._grid.dz
            array += current * func(self._grid.x[i, j], self._grid.z[i, j], x, z)
        return array

    @treat_xz_array
    def psi(
        self,
        x: Optional[Union[float, np.ndarray]] = None,
        z: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Poloidal magnetic flux at x, z

        Parameters
        ----------
        x:
            Radial coordinates at which to calculate
        z:
            Vertical coordinates at which to calculate.

        Notes
        -----
        If both x and z are None, defaults to the full map on the grid.

        Returns
        -------
        Poloidal magnetic flux at the points [V.s/rad]
        """
        if x is None and z is None:
            return self._plasma_psi

        if not self._check_in_grid(x, z):
            return self._convolve(greens_psi, x, z)
        return self._psi_func(x, z)

    @treat_xz_array
    def Bx(
        self,
        x: Optional[Union[float, np.ndarray]] = None,
        z: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Radial magnetic field at x, z

        Parameters
        ----------
        x:
            Radial coordinates at which to calculate
        z:
            Vertical coordinates at which to calculate.

        Notes
        -----
        If both x and z are None, defaults to the full map on the grid.

        Returns
        -------
        Radial magnetic field at the points [T]
        """
        if x is None and z is None:
            return self._plasma_Bx

        if not self._check_in_grid(x, z):
            return self._convolve(greens_Bx, x, z)
        return self._Bx_func(x, z)

    @treat_xz_array
    def Bz(
        self,
        x: Optional[Union[float, np.ndarray]] = None,
        z: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Vertical magnetic field at x, z

        Parameters
        ----------
        x:
            Radial coordinates at which to calculate
        z:
            Vertical coordinates at which to calculate.

        Notes
        -----
        If both x and z are None, defaults to the full map on the grid.

        Returns
        -------
        Vertical magnetic field at the points [T]
        """
        if x is None and z is None:
            return self._plasma_Bz

        if not self._check_in_grid(x, z):
            return self._convolve(greens_Bz, x, z)
        return self._Bz_func(x, z)

    def Bp(
        self,
        x: Optional[Union[float, np.ndarray]] = None,
        z: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Poloidal magnetic field at x, z

        Parameters
        ----------
        x:
            Radial coordinates at which to calculate
        z:
            Vertical coordinates at which to calculate.

        Notes
        -----
        If both x and z are None, defaults to the full map on the grid.

        Returns
        -------
        Poloidal magnetic field at the points [T]
        """
        if x is None and z is None:
            return self._plasma_Bp
        return np.hypot(self.Bx(x, z), self.Bz(x, z))

    def plot(self, ax=None):
        """
        Plot the PlasmaCoil.

        Parameters
        ----------
        ax:
            The matplotlib axes on which to plot the PlasmaCoil
        """
        return PlasmaCoilPlotter(self, ax=ax)

    def __repr__(self):
        """
        Get a simple string representation of the PlasmaCoil.
        """
        n_filaments = len(np.nonzero(self._j_tor > 0)[0])
        return f"{self.__class__.__name__}: {n_filaments} filaments"


class NoPlasmaCoil:
    """
    NoPlasmaCoil object for dummy representation of a plasma-less state.

    Parameters
    ----------
    grid:
        Grid object on which the finite difference representation of the plasma should be
        constructed
    """

    def __init__(self, grid: Grid):
        self.grid = grid

    def psi(
        self,
        x: Optional[Union[float, np.ndarray]] = None,
        z: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Poloidal magnetic flux at x, z

        Parameters
        ----------
        x:
            Radial coordinates at which to calculate
        z:
            Vertical coordinates at which to calculate.

        Notes
        -----
        If both x and z are None, defaults to the full map on the grid.

        Returns
        -------
        Poloidal magnetic flux at the points [V.s/rad]
        """
        return self._return_zeros(x, z)

    def Bx(
        self,
        x: Optional[Union[float, np.ndarray]] = None,
        z: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Radial magnetic field at x, z

        Parameters
        ----------
        x:
            Radial coordinates at which to calculate
        z:
            Vertical coordinates at which to calculate.

        Notes
        -----
        If both x and z are None, defaults to the full map on the grid.

        Returns
        -------
        Radial magnetic field at the points [T]
        """
        return self._return_zeros(x, z)

    def Bz(
        self,
        x: Optional[Union[float, np.ndarray]] = None,
        z: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Vertical magnetic field at x, z

        Parameters
        ----------
        x:
            Radial coordinates at which to calculate
        z:
            Vertical coordinates at which to calculate.

        Notes
        -----
        If both x and z are None, defaults to the full map on the grid.

        Returns
        -------
        Vertical magnetic field at the points [T]
        """
        return self._return_zeros(x, z)

    def Bp(
        self,
        x: Optional[Union[float, np.ndarray]] = None,
        z: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Poloidal magnetic field at x, z

        Parameters
        ----------
        x:
            Radial coordinates at which to calculate
        z:
            Vertical coordinates at which to calculate.

        Notes
        -----
        If both x and z are None, defaults to the full map on the grid.

        Returns
        -------
        Poloidal magnetic field at the points [T]
        """
        return self._return_zeros(x, z)

    def _return_zeros(self, x, z):
        if x is None and z is None:
            return np.zeros_like(self.grid.x)

        return np.zeros_like(x)
