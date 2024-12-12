# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Representation of the plasma
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import RectBivariateSpline

from bluemira.equilibria.constants import J_TOR_MIN
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.plotting import PlasmaCoilPlotter
from bluemira.magnetostatics.greens import (
    greens_Bx,
    greens_Bz,
    greens_dbz_dx,
    greens_psi,
)
from bluemira.utilities.tools import floatify

if TYPE_CHECKING:
    import numpy.typing as npt

    from bluemira.equilibria.grid import Grid


def treat_xz_array(func):
    """
    Decorator for handling array calls to PlasmaCoil methods.

    Returns
    -------
    :
        Wrapped function

    Raises
    ------
    EquilibriaError
        x and z must be specified and the same dimension
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

        for i, (xx, zz) in enumerate(zip(x.flat, z.flat, strict=False)):
            values[i] = floatify(func(self, xx, zz))

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

    def __init__(
        self,
        plasma_psi: npt.NDArray[np.float64],
        j_tor: npt.NDArray[np.float64] | None,
        grid: Grid,
    ):
        self._grid = grid
        self._set_j_tor(j_tor)
        self._set_funcs(plasma_psi)

    def _set_j_tor(self, j_tor: npt.NDArray[np.float64] | None):
        self._j_tor = j_tor
        if j_tor is not None:
            self._ii, self._jj = np.nonzero(j_tor > J_TOR_MIN)
        else:
            self._ii, self._jj = None, None

    def _set_funcs(self, plasma_psi: npt.NDArray[np.float64]):
        self._plasma_psi = plasma_psi
        self._psi_func = RectBivariateSpline(
            self._grid.x[:, 0], self._grid.z[0, :], plasma_psi
        )
        self._plasma_Bx = self._Bx_func(self._grid.x, self._grid.z)
        self._plasma_dBx = self._dBx_func(self._grid.x, self._grid.z)
        self._plasma_Bz = self._Bz_func(self._grid.x, self._grid.z)
        self._plasma_dBz = self._dBz_func(self._grid.x, self._grid.z)
        self._plasma_Bp = np.hypot(self._plasma_Bx, self._plasma_Bz)

    def _Bx_func(self, x, z):
        return -self._psi_func(x, z, dy=1, grid=False) / x

    def _dBx_func(self, x, z):
        return -self._psi_func(x, z, dy=2, grid=False) / x**2

    def _Bz_func(self, x, z):
        return self._psi_func(x, z, dx=1, grid=False) / x

    def _dBz_func(self, x, z):
        return -self._psi_func(x, z, dx=2, grid=False) / x**2

    def _check_in_grid(self, x, z):
        return self._grid.point_inside(x, z)

    def _convolve(self, func, x, z):
        """
        Map a Green's function across the grid at a point, without crashing or
        running out of memory.

        Raises
        ------
        EquilibriaError
            No known toroidal current distribution

        Returns
        -------
        :
            Mapped greens function
        """
        if self._j_tor is None:
            raise EquilibriaError(
                "Cannot calculate value off grid; there is no known toroidal current"
                " distribution."
            )

        return np.sum(
            self._j_tor[self._ii, self._jj]
            * self._grid.dx
            * self._grid.dz
            * func(
                self._grid.x[self._ii, self._jj], self._grid.z[self._ii, self._jj], x, z
            ),
            axis=0,
        )

    @treat_xz_array
    def psi(
        self,
        x: npt.ArrayLike | None = None,
        z: npt.ArrayLike | None = None,
    ) -> float | npt.NDArray[np.float64]:
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
        :
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
        x: npt.ArrayLike | None = None,
        z: npt.ArrayLike | None = None,
    ) -> float | npt.NDArray[np.float64]:
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
        :
            Radial magnetic field at the points [T]
        """
        if x is None and z is None:
            return self._plasma_Bx

        if not self._check_in_grid(x, z):
            return self._convolve(greens_Bx, x, z)
        return self._Bx_func(x, z)

    @treat_xz_array
    def dBx(
        self,
        x: npt.ArrayLike | None = None,
        z: npt.ArrayLike | None = None,
    ) -> float | npt.NDArray[np.float64]:
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
        :
            Radial magnetic field at the points [T]
        """
        if x is None and z is None:
            return self._plasma_dBx

        if not self._check_in_grid(x, z):
            # greens_dbx_dz not implemented, for vacuum calculations this is equivalent
            return self._convolve(greens_dbz_dx, x, z)
        return self._dBx_func(x, z)

    @treat_xz_array
    def Bz(
        self,
        x: npt.ArrayLike | None = None,
        z: npt.ArrayLike | None = None,
    ) -> float | npt.NDArray[np.float64]:
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
        :
            Vertical magnetic field at the points [T]
        """
        if x is None and z is None:
            return self._plasma_Bz

        if not self._check_in_grid(x, z):
            return self._convolve(greens_Bz, x, z)
        return self._Bz_func(x, z)

    @treat_xz_array
    def dBz(
        self,
        x: npt.ArrayLike | None = None,
        z: npt.ArrayLike | None = None,
    ) -> float | npt.NDArray[np.float64]:
        """
        Vertical magnetic field at x, z

        Parameters
        ----------
        x:
            Vertical coordinates at which to calculate
        z:
            Vertical coordinates at which to calculate.

        Notes
        -----
        If both x and z are None, defaults to the full map on the grid.

        Returns
        -------
        :
            Vertical magnetic field at the points [T]
        """
        if x is None and z is None:
            return self._plasma_dBz

        if not self._check_in_grid(x, z):
            return self._convolve(greens_dbz_dx, x, z)
        return self._dBz_func(x, z)

    def Bp(
        self,
        x: npt.ArrayLike | None = None,
        z: npt.ArrayLike | None = None,
    ) -> float | npt.NDArray[np.float64]:
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
        :
            Poloidal magnetic field at the points [T]
        """
        if x is None and z is None:
            return self._plasma_Bp
        return np.hypot(self.Bx(x, z), self.Bz(x, z))

    def plot(self, ax=None) -> PlasmaCoilPlotter:
        """
        Plot the PlasmaCoil.

        Parameters
        ----------
        ax:
            The matplotlib axes on which to plot the PlasmaCoil

        Returns
        -------
        :
            The plasma coil plotter object
        """
        return PlasmaCoilPlotter(self, ax=ax)

    def __repr__(self) -> str:
        """
        Returns
        -------
        :
            A simple string representation of the PlasmaCoil.
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
        x: npt.ArrayLike | None = None,
        z: npt.ArrayLike | None = None,
    ) -> float | npt.NDArray[np.float64]:
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
        :
            Poloidal magnetic flux at the points [V.s/rad]
        """
        return self._return_zeros(x, z)

    def Bx(
        self,
        x: npt.ArrayLike | None = None,
        z: npt.ArrayLike | None = None,
    ) -> float | npt.NDArray[np.float64]:
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
        :
            Radial magnetic field at the points [T]
        """
        return self._return_zeros(x, z)

    def Bz(
        self,
        x: npt.ArrayLike | None = None,
        z: npt.ArrayLike | None = None,
    ) -> float | npt.NDArray[np.float64]:
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
        :
            Vertical magnetic field at the points [T]
        """
        return self._return_zeros(x, z)

    def Bp(
        self,
        x: npt.ArrayLike | None = None,
        z: npt.ArrayLike | None = None,
    ) -> float | npt.NDArray[np.float64]:
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
        :
            Poloidal magnetic field at the points [T]
        """
        return self._return_zeros(x, z)

    def dBx(
        self,
        x: npt.ArrayLike | None = None,
        z: npt.ArrayLike | None = None,
    ) -> float | npt.NDArray[np.float64]:
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
        :
            Radial magnetic field at the points [T]
        """
        return self._return_zeros(x, z)

    def dBz(
        self,
        x: npt.ArrayLike | None = None,
        z: npt.ArrayLike | None = None,
    ) -> float | npt.NDArray[np.float64]:
        """
        Vertical magnetic field at x, z

        Parameters
        ----------
        x:
            Vertical coordinates at which to calculate
        z:
            Vertical coordinates at which to calculate.

        Notes
        -----
        If both x and z are None, defaults to the full map on the grid.

        Returns
        -------
        :
            Vertical magnetic field at the points [T]
        """
        return self._return_zeros(x, z)

    def _return_zeros(self, x, z):
        if x is None and z is None:
            return np.zeros_like(self.grid.x)

        return np.zeros_like(x)
