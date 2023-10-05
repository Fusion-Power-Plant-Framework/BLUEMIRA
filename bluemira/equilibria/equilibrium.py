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
Plasma MHD equilibrium and state objects
"""
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import tabulate
from scipy.optimize import minimize

from bluemira.base.constants import MU_0
from bluemira.base.file import get_bluemira_path
from bluemira.base.look_and_feel import bluemira_print_flush
from bluemira.equilibria.boundary import FreeBoundary, apply_boundary
from bluemira.equilibria.coils import CoilSet, symmetrise_coilset
from bluemira.equilibria.constants import PSI_NORM_TOL
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.file import EQDSKInterface
from bluemira.equilibria.find import (
    Opoint,
    Xpoint,
    find_LCFS_separatrix,
    find_OX_points,
    find_flux_surf,
    in_plasma,
    in_zone,
)
from bluemira.equilibria.flux_surfaces import (
    ClosedFluxSurface,
    CoreResults,
    analyse_plasma_core,
)
from bluemira.equilibria.grad_shafranov import GSSolver
from bluemira.equilibria.grid import Grid, integrate_dx_dz
from bluemira.equilibria.limiter import Limiter
from bluemira.equilibria.num_control import DummyController, VirtualController
from bluemira.equilibria.physics import calc_li3minargs, calc_psi_norm, calc_summary
from bluemira.equilibria.plasma import NoPlasmaCoil, PlasmaCoil
from bluemira.equilibria.plotting import (
    BreakdownPlotter,
    CorePlotter,
    CorePlotter2,
    EquilibriumPlotter,
    FixedPlasmaEquilibriumPlotter,
)
from bluemira.equilibria.profiles import BetaLiIpProfile, CustomProfile, Profile
from bluemira.geometry.coordinates import Coordinates
from bluemira.optimisation._tools import process_scipy_result
from bluemira.utilities.tools import abs_rel_difference


class MHDState:
    """
    Base class for magneto-hydrodynamic states
    """

    def __init__(self):
        # Constructors
        self.x: Union[None, np.ndarray] = None
        self.z: Union[None, np.ndarray] = None
        self.dx: Union[None, float] = None
        self.dz: Union[None, float] = None
        self.grid: Union[None, Grid] = None
        self.limiter: Union[None, Limiter] = None

    def set_grid(self, grid: Grid):
        """
        Sets a Grid object for an Equilibrium, and sets the G-S operator and
        G-S solver on the grid.

        Parameters
        ----------
        grid:
            The grid upon which to solve the Equilibrium
        """
        self.grid = grid
        self.x, self.z = self.grid.x, self.grid.z
        self.dx, self.dz = self.grid.dx, self.grid.dz

    @classmethod
    def _get_eqdsk(
        cls,
        filename: str,
    ) -> Tuple[EQDSKInterface, np.ndarray, CoilSet, Grid, Optional[Limiter]]:
        """
        Get eqdsk data from file for read in

        Parameters
        ----------
        filename:
            Filename
        force_symmetry:
            Whether or not to force symmetrisation in the CoilSet

        Returns
        -------
        e:
            Instance if EQDSKInterface with the EQDSK file read in
        psi:
            psi array
        coilset:
            Coilset from eqdsk
        grid:
            Grid from eqdsk
        limiter:
            Limiter instance if any limiters are in file
        """
        e = EQDSKInterface.from_file(filename)
        if "equilibria" in e.name:
            psi = e.psi
        elif "SCENE" in e.name and not isinstance(cls, Breakdown):
            psi = e.psi
            e.dxc = e.dxc / 2
            e.dzc = e.dzc / 2
        elif "fiesta" in e.name.lower():
            psi = e.psi
        else:  # CREATE
            psi = e.psi / (2 * np.pi)  # V.s as opposed to V.s/rad
            e.dxc = e.dxc / 2
            e.dzc = e.dzc / 2
            e.cplasma = abs(e.cplasma)

        grid = Grid.from_eqdsk(e)

        return e, psi, grid

    def to_eqdsk(
        self,
        data: Dict[str, Any],
        filename: Union[Path, str],
        header: str = "bluemira_equilibria",
        directory: Optional[str] = None,
        filetype: str = "json",
        **kwargs,
    ):
        """
        Writes the Equilibrium Object to an eqdsk file
        """
        data["name"] = f"{filename}_{header}"

        if not filename.endswith(f".{filetype}"):
            filename += f".{filetype}"

        if directory is None:
            try:
                filename = Path(
                    get_bluemira_path("eqdsk/equilibria", subfolder="data"), filename
                )
            except ValueError as error:
                raise ValueError(
                    f"Unable to find default data directory: {error}"
                ) from None
        else:
            filename = Path(directory, filename)

        self.filename = filename  # Convenient
        eqdsk = EQDSKInterface(**data)
        eqdsk.write(filename, file_format=filetype, **kwargs)


class FixedPlasmaEquilibrium(MHDState):
    """
    Class for loading a fixed boundary plasma equilibrium.
    """

    def __init__(
        self,
        grid: Grid,
        lcfs: Coordinates,
        profiles: Profile,
        psi: np.ndarray,
        psi_ax: float,
        psi_b: float,
        filename: Optional[str] = None,
    ):
        super().__init__()
        self.set_grid(grid)
        # We just need the flux values, not the locations
        o_points = [Opoint(0.0, 0.0, psi_ax)]
        x_points = [Xpoint(0.0, 0.0, psi_b)]
        j_tor = profiles.jtor(
            grid.x, grid.z, psi, o_points=o_points, x_points=x_points, lcfs=lcfs.xz.T
        )
        self._psi = psi
        self._jtor = j_tor
        self.profiles = profiles
        self.plasma = PlasmaCoil(psi, j_tor, self.grid)
        self._lcfs = lcfs
        self.filename = filename

    @classmethod
    def from_eqdsk(cls, filename: str):
        """
        Initialises a Breakdown Object from an eqdsk file. Note that this
        will involve recalculation of the magnetic flux.

        Parameters
        ----------
        filename:
            Filename
        force_symmetry:
            Whether or not to force symmetrisation in the CoilSet
        """
        e, psi, grid = super()._get_eqdsk(filename)
        psi_ax = e.psimag
        psi_b = e.psibdry
        lcfs = Coordinates({"x": e.xbdry, "z": e.zbdry})
        lcfs.close()

        profiles = CustomProfile.from_eqdsk(filename)

        cls._eqdsk = e
        return cls(
            grid, lcfs, profiles, psi=psi, psi_ax=psi_ax, psi_b=psi_b, filename=filename
        )

    def get_LCFS(self) -> Coordinates:
        """
        Get the Last Closed FLux Surface (LCFS).

        Returns
        -------
        The Coordinates of the LCFS
        """
        return deepcopy(self._lcfs)

    def Bx(
        self,
        x: Optional[Union[float, np.ndarray]] = None,
        z: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Total radial magnetic field at point (x, z) from coils

        Parameters
        ----------
        x:
            Radial coordinates for which to return Bx. If None, returns values
            at all grid points
        z:
            Vertical coordinates for which to return Bx. If None, returns values
            at all grid points

        Returns
        -------
        Radial magnetic field at x, z
        """
        return self.plasma.Bx(x, z)

    def Bz(
        self,
        x: Optional[Union[float, np.ndarray]] = None,
        z: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Total vertical magnetic field at point (x, z) from coils

        Parameters
        ----------
        x:
            Radial coordinates for which to return Bz. If None, returns values
            at all grid points
        z:
            Vertical coordinates for which to return Bz. If None, returns values
            at all grid points

        Returns
        -------
        Vertical magnetic field at x, z
        """
        return self.plasma.Bz(x, z)

    def Bp(
        self,
        x: Optional[Union[float, np.ndarray]] = None,
        z: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Total poloidal magnetic field at point(s) (x, z)

        Parameters
        ----------
        x:
            Radial coordinates for which to return Bp. If None, returns values
            at all grid points
        z:
            Vertical coordinates for which to return Bp. If None, returns values
            at all grid points

        Returns
        -------
        Poloidal magnetic field at x, z
        """
        return np.hypot(self.Bx(x, z), self.Bz(x, z))

    def psi(
        self,
        x: Optional[Union[float, np.ndarray]] = None,
        z: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Total poloidal magnetic flux, either for the whole grid, or for
        specified x, z coordinates.

        Parameters
        ----------
        x:
            Radial coordinates for which to return psi. If None, returns values
            at all grid points
        z:
            Vertical coordinates for which to return psi. If None, returns values
            at all grid points

        Returns
        -------
        Poloidal magnetic flux at x, z
        """
        if x is None and z is None:
            return self._psi

        return self.plasma.psi(x, z)

    def plot(self, ax: Optional[plt.Axes] = None, field: bool = False):
        """
        Plots the FixedPlasmaEquilibrium object onto `ax`
        """
        return FixedPlasmaEquilibriumPlotter(self, ax, field=field)


class CoilSetMHDState(MHDState):
    """
    Base class for magneto-hydrodynamic states with a CoilSet
    """

    def __init__(self):
        super().__init__()
        self._psi_green = None
        self._bx_green = None
        self._bz_green = None
        self.coilset = None

    @classmethod
    def _get_eqdsk(
        cls, filename: str, force_symmetry: bool = False
    ) -> Tuple[EQDSKInterface, np.ndarray, CoilSet, Grid, Optional[Limiter]]:
        """
        Get eqdsk data from file for read in

        Parameters
        ----------
        filename:
            Filename
        force_symmetry:
            Whether or not to force symmetrisation in the CoilSet

        Returns
        -------
        e:
            Instance if EQDSKInterface with the EQDSK file read in
        psi:
            psi array
        coilset:
            Coilset from eqdsk
        grid:
            Grid from eqdsk
        limiter:
            Limiter instance if any limiters are in file
        """
        e, psi, grid = super()._get_eqdsk(filename)
        coilset = CoilSet.from_group_vecs(e)
        if force_symmetry:
            coilset = symmetrise_coilset(coilset)

        if e.nlim == 0:
            limiter = None
        elif e.nlim < 5:  # noqa: PLR2004
            limiter = Limiter(e.xlim, e.zlim)
        else:
            limiter = None  # CREATE..

        return e, psi, coilset, grid, limiter

    def _remap_greens(self):
        """
        Stores Green's functions arrays in a dictionary of coils. Used upon
        initialisation and must be called after meshing of coils.

        Notes
        -----
        Modifies:

            ._pgreen:
                Greens function coil mapping for psi
            ._bxgreen:
                Greens function coil mapping for Bx
            .bzgreen:
                Greens function coil mapping for Bz
        """
        self._psi_green = self.coilset.psi_response(self.x, self.z)
        self._bx_green = self.coilset.Bx_response(self.x, self.z)
        self._bz_green = self.coilset.Bz_response(self.x, self.z)

    def get_coil_forces(self) -> np.ndarray:
        """
        Returns the Fx and Fz force at the centre of the control coils

        Returns
        -------
        Fx, Fz array of forces on coils [N]

        Notes
        -----
        Will not work for symmetric circuits
        """
        no_coils = self.coilset.n_coils()
        plasma = self.plasma
        non_zero_current = np.nonzero(self.coilset.current)[0]
        response = self.coilset.control_F(self.coilset)
        background = (
            self.coilset.F(plasma)[non_zero_current]
            / self.coilset.current[non_zero_current]
        )

        forces = np.zeros((no_coils, 2))
        currents = self.coilset.get_control_coils().current
        forces[:, 0] = currents * (response[:, :, 0] @ currents + background[:, 0])
        forces[:, 1] = currents * (response[:, :, 1] @ currents + background[:, 1])

        return forces

    def get_coil_fields(self) -> np.ndarray:
        """
        Returns the poloidal magnetic fields on the control coils
        (approximate peak at the middle inner radius of the coil)

        Returns
        -------
        The Bp array of fields on coils [T]
        """
        return self.Bp(self.coilset.x - self.coilset.dx, self.coilset.z)

    def reset_grid(self, grid: Grid, psi: Optional[np.ndarray] = None):
        """
        Reset the grid for the MHDState.

        Parameters
        ----------
        grid:
            The grid to set the MHDState on
        psi:
            Initial psi array to use
        """
        self.set_grid(grid)
        self._set_init_plasma(grid, psi)

    def _set_init_plasma(self, grid: Grid, psi: Optional[np.ndarray] = None):
        zm = 1 - grid.z_max / (grid.z_max - grid.z_min)
        if psi is None:  # Initial psi guess
            # Normed 0-1 grid
            x, z = self.x / grid.x_max, (self.z - grid.z_min) / (grid.z_max - grid.z_min)
            # Factor has an important effect sometimes... good starting
            # solutions matter
            psi = 100 * np.exp(-((x - 0.5) ** 2 + (z - zm) ** 2) / 0.1)
            apply_boundary(psi, 0)

        self._remap_greens()
        return psi


class Breakdown(CoilSetMHDState):
    """
    Represents the breakdown state

    Parameters
    ----------
    coilset:
        The set of coil objects which the equilibrium will be solved with
    grid:
        The grid which to solve over
    psi:
        The initial psi array (optional)
    filename:
        The filename of the breakdown (optional)
    """

    def __init__(
        self,
        coilset: CoilSet,
        grid: Grid,
        psi: Optional[np.ndarray] = None,
        filename: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.coilset = coilset
        self.set_grid(grid)
        self._set_init_plasma(grid, psi)
        self.plasma = NoPlasmaCoil(grid)
        self.limiter = kwargs.get("limiter", None)

        # Set default breakdown point to grid centre
        x_mid = grid.x_min + 0.5 * (grid.x_max + grid.x_min)
        self.breakdown_point = kwargs.get("breakdown_point", (x_mid, 0))
        self.filename = filename

    @classmethod
    def from_eqdsk(cls, filename: str, force_symmetry: bool):
        """
        Initialises a Breakdown Object from an eqdsk file. Note that this
        will involve recalculation of the magnetic flux.

        Parameters
        ----------
        filename:
            Filename
        force_symmetry:
            Whether or not to force symmetrisation in the CoilSet
        """
        cls._eqdsk, psi, coilset, grid, limiter = super()._get_eqdsk(
            filename, force_symmetry=force_symmetry
        )
        return cls(coilset, grid, limiter=limiter, psi=psi, filename=filename)

    def to_dict(self) -> Dict[str, Any]:
        """
        Creates a dictionary for a Breakdown object

        Returns
        -------
        A dictionary for the Breakdown object
        """
        xc, zc, dxc, dzc, currents = self.coilset.to_group_vecs()
        return {
            "nx": self.grid.nx,
            "nz": self.grid.nz,
            "xdim": self.grid.x_size,
            "zdim": self.grid.z_size,
            "x": self.grid.x_1d,
            "z": self.grid.z_1d,
            "xgrid1": self.grid.x_min,
            "zmid": self.grid.z_mid,
            "cplasma": 0.0,
            "psi": self.psi(),
            "Bx": self.Bx(),
            "Bz": self.Bz(),
            "Bp": self.Bp(),
            "ncoil": self.coilset.n_coils(),
            "xc": xc,
            "zc": zc,
            "dxc": dxc,
            "dzc": dzc,
            "Ic": currents,
        }

    def to_eqdsk(
        self,
        filename: str,
        header: str = "bluemira_equilibria",
        directory: Optional[str] = None,
        filetype: str = "json",
        **kwargs,
    ) -> EQDSKInterface:
        """
        Writes the Equilibrium Object to an eqdsk file
        """
        data = self.to_dict()
        data["xcentre"] = 0
        data["bcentre"] = 0
        super().to_eqdsk(data, filename, header, directory, filetype, **kwargs)

    def set_breakdown_point(self, x_bd: float, z_bd: float):
        """
        Set the point at which the centre of the breakdown region is defined.

        Parameters
        ----------
        x_bd:
            The x coordinate of the centre of the breakdown region
        z_bd:
            The z coordinate of the centre of the breakdown region
        """
        self.breakdown_point = (x_bd, z_bd)

    @property
    def breakdown_psi(self) -> float:
        """
        The poloidal magnetic flux at the centre of the breakdown region.

        Returns
        -------
        The minimum poloidal magnetic flux at the edge of the breakdown
        region [V.s/rad]
        """
        return self.psi(*self.breakdown_point)

    def Bx(
        self,
        x: Optional[Union[float, np.ndarray]] = None,
        z: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Total radial magnetic field at point (x, z) from coils

        Parameters
        ----------
        x:
            Radial coordinates for which to return Bx. If None, returns values
            at all grid points
        z:
            Vertical coordinates for which to return Bx. If None, returns values
            at all grid points

        Returns
        -------
        Radial magnetic field at x, z
        """
        if x is None and z is None:
            return self.coilset._Bx_greens(self._bx_green)

        return self.coilset.Bx(x, z)

    def Bz(
        self,
        x: Optional[Union[float, np.ndarray]] = None,
        z: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Total vertical magnetic field at point (x, z) from coils

        Parameters
        ----------
        x:
            Radial coordinates for which to return Bz. If None, returns values
            at all grid points
        z:
            Vertical coordinates for which to return Bz. If None, returns values
            at all grid points

        Returns
        -------
        Vertical magnetic field at x, z
        """
        if x is None and z is None:
            return self.coilset._Bz_greens(self._bz_green)

        return self.coilset.Bz(x, z)

    def Bp(
        self,
        x: Optional[Union[float, np.ndarray]] = None,
        z: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Total poloidal magnetic field at point (x, z) from coils

        Parameters
        ----------
        x:
            Radial coordinates for which to return Bp. If None, returns values
            at all grid points
        z:
            Vertical coordinates for which to return Bp. If None, returns values
            at all grid points

        Returns
        -------
        Poloidal magnetic field at x, z
        """
        if x is None and z is None:
            return np.hypot(
                self.coilset._Bx_greens(self._bx_green),
                self.coilset._Bz_greens(self._bz_green),
            )

        return np.hypot(self.Bx(x, z), self.Bz(x, z))

    def psi(
        self,
        x: Optional[Union[float, np.ndarray]] = None,
        z: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Returns the poloidal magnetic flux, either for the whole grid, or for
        specified x, z coordinates, including contributions from the coilset.

        Parameters
        ----------
        x:
            Radial coordinates for which to return psi. If None, returns values
            at all grid points
        z:
            Vertical coordinates for which to return psi. If None, returns values
            at all grid points

        Returns
        -------
        Poloidal magnetic flux at x, z
        """
        if x is None and z is None:
            return self.coilset._psi_greens(self._psi_green)

        return self.coilset.psi(x, z)

    def get_coil_Bp(self) -> np.ndarray:
        """
        Returns the poloidal field within each coil
        """
        b = np.zeros(self.coilset.n_coils())
        dx_mask = np.zeros_like(self.coilset.dx)
        dx_mask[self.coilset.dx > 0] = True
        mask = in_zone(
            self.x[dx_mask],
            self.z[dx_mask],
            np.array([self.x[dx_mask], self.z[dx_mask]]).T,
        )
        b[dx_mask] = np.max(self.Bp()[dx_mask] * mask[dx_mask], axis=-1)
        b[~dx_mask] = np.max(self.Bp(self.x, self.z)[~dx_mask] * mask[~dx_mask], axis=-1)
        return b

    def plot(self, ax: Optional[plt.Axes] = None, Bp: bool = False):
        """
        Plots the breakdown object onto `ax`
        """
        return BreakdownPlotter(self, ax, Bp=Bp)


class QpsiCalcMode(Enum):
    """
    Modes for how to calculate qpsi

    Parameters
    ----------
    0:
        Don't Calculate qpsi
    1:
        Calculate qpsi
    2:
        Fill qpsi grid with Zeros
    """

    NO_CALC = 0
    CALC = 1
    ZEROS = 2


class Equilibrium(CoilSetMHDState):
    """
    Represents the equilibrium state, including plasma and coil currents

    Parameters
    ----------
    coilset:
        The set of coil objects which the equilibrium will be solved with
    grid:
        The grid on which to calculate the Equilibrium
    profiles:
        The plasma profiles to use in the Equilibrium
    force_symmetry:
        Controls whether symmetry of the plasma contribution to psi across z=0
        is strictly enforced in the linear system formed during solve step.
    vcontrol:
        Type of virtual plasma control to enact
    limiter:
        Limiter conditions to apply to equilibrium
    psi:
        Magnetic flux [V.s] applied to X, Z grid
    jtor:
        The toroidal current density array of the plasma. Default = None will
        cause the jtor array to be constructed later as necessary.
    filename:
        The filename of the Equilibrium. Default = None (no file)
    """

    def __init__(
        self,
        coilset: CoilSet,
        grid: Grid,
        profiles: Profile,
        force_symmetry: bool = False,
        vcontrol: Optional[str] = None,
        limiter: Optional[Limiter] = None,
        psi: Optional[np.ndarray] = None,
        jtor: Optional[np.ndarray] = None,
        filename: Optional[str] = None,
    ):
        super().__init__()
        # Constructors
        self._jtor = jtor
        self.profiles = profiles
        self._o_points = None
        self._x_points = None
        self._solver = None
        self._eqdsk = None

        self._li_flag = False
        if isinstance(profiles, BetaLiIpProfile):
            self._li_flag = True
            self._li = profiles._l_i_target  # target plasma normalised inductance
            self._li_iter = 0  # li iteration count
            self._li_temp = None

        self.plasma = None

        self.force_symmetry = force_symmetry
        self.controller = None
        self.coilset = coilset

        self.set_grid(grid)
        self._set_init_plasma(grid, psi, jtor)
        self.boundary = FreeBoundary(self.grid)
        self.set_vcontrol(vcontrol)
        self.limiter = limiter
        self.filename = filename

        self._kwargs = {"vcontrol": vcontrol}

    @classmethod
    def from_eqdsk(cls, filename: str, force_symmetry: bool = False):
        """
        Initialises an Equilibrium Object from an eqdsk file. Note that this
        will involve recalculation of the magnetic flux. Because of the nature
        of the (non-linear) Grad-Shafranov equation, values of psi may differ
        from those stored in eqdsk.

        NOTE: Need to solve again with some profiles in order to refind...

        Parameters
        ----------
        filename:
            Filename
        force_symmetry:
            Whether or not to force symmetrisation in the CoilSet
        """
        e, psi, coilset, grid, limiter = super()._get_eqdsk(
            filename, force_symmetry=force_symmetry
        )

        profiles = CustomProfile.from_eqdsk(filename)

        cls._eqdsk = e

        o_points, x_points = find_OX_points(grid.x, grid.z, psi, limiter=limiter)
        jtor = profiles.jtor(grid.x, grid.z, psi, o_points=o_points, x_points=x_points)

        return cls(
            coilset,
            grid,
            profiles=profiles,
            vcontrol=None,
            limiter=limiter,
            psi=psi,
            jtor=jtor,
            filename=filename,
        )

    def to_dict(self, qpsi_calcmode: int = 0) -> Dict[str, Any]:
        """
        Creates dictionary for equilibrium object, in preparation for saving
        to a file format

        Parameters
        ----------
        qpsi_calcmode:
          don't calculate: 0, calculate qpsi: 1, fill with zeros: 2

        Returns
        -------
        A dictionary of the Equilibrium object values, sufficient for EQDSK
        """
        qpsi_calcmode = QpsiCalcMode(qpsi_calcmode)

        psi = self.psi()
        n_x, n_z = psi.shape
        opoints, xpoints = self.get_OX_points(psi)
        opoint = opoints[0]  # Primary points
        # It is possible to have an EQDSK with no X-point...
        psi_bndry = xpoints[0][2] if xpoints else np.amin(psi)
        psinorm = np.linspace(0, 1, n_x)

        if qpsi_calcmode is QpsiCalcMode.CALC:
            # This is too damn slow..
            q = self.q(psinorm, o_points=opoints, x_points=xpoints)
        elif qpsi_calcmode is QpsiCalcMode.ZEROS:
            q = np.zeros(n_x)

        lcfs = self.get_LCFS(psi)
        nbdry = lcfs.xz.shape[1]
        x_c, z_c, dxc, dzc, currents = self.coilset.to_group_vecs()

        result = {
            "nx": n_x,
            "nz": n_z,
            "xdim": self.grid.x_size,
            "zdim": self.grid.z_size,
            "x": self.grid.x_1d,
            "z": self.grid.z_1d,
            "xcentre": self.profiles.R_0,
            "bcentre": self.profiles._B_0,
            "xgrid1": self.grid.x_min,
            "zmid": self.grid.z_mid,
            "xmag": opoint[0],
            "zmag": opoint[1],
            "psimag": opoint[2],
            "psibdry": psi_bndry,
            "cplasma": self.profiles.I_p,
            "psi": psi,
            "fpol": self.fRBpol(psinorm),
            "ffprime": self.ffprime(psinorm),
            "pprime": self.pprime(psinorm),
            "pressure": self.pressure(psinorm),
            "psinorm": psinorm,
            "nbdry": nbdry,
            "xbdry": lcfs.x,
            "zbdry": lcfs.z,
            "ncoil": self.coilset.n_coils(),
            "xc": x_c,
            "zc": z_c,
            "dxc": dxc,
            "dzc": dzc,
            "Ic": currents,
        }
        if qpsi_calcmode is not QpsiCalcMode.NO_CALC:
            result["qpsi"] = q

        if self.limiter is None:  # Needed for eqdsk file format
            result["nlim"] = 0
            result["xlim"] = np.ndarray([])
            result["zlim"] = np.ndarray([])
        else:
            result["nlim"] = len(self.limiter)
            result["xlim"] = self.limiter.x
            result["zlim"] = self.limiter.z
        return result

    def to_eqdsk(
        self,
        filename: str,
        header: str = "BP_equilibria",
        directory: Optional[str] = None,
        filetype: str = "json",
        qpsi_calcmode: int = 0,
        **kwargs,
    ):
        """
        Writes the Equilibrium Object to an eqdsk file
        """
        if "eqdsk" in filetype and qpsi_calcmode == 0:
            qpsi_calcmode = 2

        super().to_eqdsk(
            self.to_dict(qpsi_calcmode),
            filename,
            header,
            directory,
            filetype,
            **kwargs,
        )

    def __getstate__(self):
        """
        Get the state of the Equilibrium object. Used in pickling.
        """
        d = dict(self.__dict__)
        d.pop("_solver", None)
        return d

    def __setstate__(self, d):
        """
        Get the state of the Equilibrium object. Used in unpickling.
        """
        self.__dict__ = d
        if "grid" in d:
            self.set_grid(self.grid)

    def set_grid(self, grid: Grid):
        """
        Sets a Grid object for an Equilibrium, and sets the G-S operator and
        G-S solver on the grid.

        Parameters
        ----------
        grid:
            The grid upon which to solve the Equilibrium
        """
        super().set_grid(grid)

        self._solver = GSSolver(grid, force_symmetry=self.force_symmetry)

    def reset_grid(self, grid: Grid, **kwargs):
        """
        Yeah, yeah...
        """
        super().reset_grid(grid, **kwargs)
        vcontrol = kwargs.get("vcontrol", self._kwargs["vcontrol"])
        self.set_vcontrol(vcontrol)
        # TODO: reinit psi and jtor?

    def _set_init_plasma(
        self,
        grid: Grid,
        psi: Optional[np.ndarray] = None,
        j_tor: Optional[np.ndarray] = None,
    ):
        psi = super()._set_init_plasma(grid, psi)

        # This is necessary when loading an equilibrium from an EQDSK file (we
        # hide the coils to get the plasma psi)
        psi -= self.coilset.psi(self.x, self.z)
        self._update_plasma(psi, j_tor)

    def set_vcontrol(self, vcontrol: Optional[str] = None):
        """
        Sets the vertical position controller

        Parameters
        ----------
        vcontrol:
            Vertical control strategy
        """
        if vcontrol == "virtual":
            self.controller = VirtualController(self, gz=2.2)
        elif vcontrol == "feedback":
            raise NotImplementedError
        elif vcontrol is None:
            self.controller = DummyController(self.plasma.psi())
        else:
            raise ValueError(
                "Please select a numerical stabilisation strategy"
                ' from: 1) "virtual" \n 2) "feedback" 3) None.'
            )

    def solve(self, jtor: Optional[np.ndarray] = None, psi: Optional[np.ndarray] = None):
        """
        Re-calculates the plasma equilibrium given new profiles

        Linear Grad-Shafranov solve

        Parameters
        ----------
        jtor:
            The toroidal current density on the finite difference grid [A/m^2]
        psi:
            The poloidal magnetic flux on the finite difference grid [V.s/rad]

        Note
        ----
        Modifies the following in-place:
            .plasma_psi
            .psi_func
            ._I_p
            ._Jtor
        """
        self._clear_OX_points()

        if jtor is None:
            if psi is None:
                psi = self.psi()
            o_points, x_points = self.get_OX_points(psi=psi, force_update=True)

            if not o_points:
                raise EquilibriaError("No O-point found in equilibrium.")
            jtor = self.profiles.jtor(self.x, self.z, psi, o_points, x_points)

        plasma_psi = self.plasma.psi()
        self.boundary(plasma_psi, jtor)
        rhs = -MU_0 * self.x * jtor  # RHS of GS equation
        apply_boundary(rhs, plasma_psi)

        plasma_psi = self._solver(rhs)
        self._update_plasma(plasma_psi, jtor)

        self._jtor = jtor
        self._plasmacoil = None

    def solve_li(
        self,
        jtor: Optional[np.ndarray] = None,  # noqa: ARG002
        psi: Optional[np.ndarray] = None,
    ):
        """
        Optimises profiles to match input li
        Re-calculates the plasma equilibrium given new profiles

        Linear Grad-Shafranov solve

        Parameters
        ----------
        jtor:
            The 2-D array toroidal current at each (x, z) point (optional)
        psi:
            The 2-D array of poloidal magnetic flux at each (x, z) point (optional)

        Note
        ----
        Modifies the following in-place:

            .plasma_psi
            .psi_func
            ._I_p
            ._Jtor

        jtor argument input is not used but kept for consistency with `solve`
        """
        if not self._li_flag:
            raise EquilibriaError("Cannot use solve_li without the BetaLiIpProfile.")
        self._clear_OX_points()
        self._li_iter = 0
        if psi is None:
            psi = self.psi()
        # Speed optimisations
        o_points, x_points = self.get_OX_points(psi=psi, force_update=True)
        mask = in_plasma(self.x, self.z, psi, o_points=o_points, x_points=x_points)
        print("")  # flusher

        def minimise_dli(x):
            """
            The minimisation function to obtain the correct l_i
            """
            self.profiles.shape.adjust_parameters(x)
            jtor_opt = self.profiles.jtor(self.x, self.z, psi, o_points, x_points)
            plasma_psi = self.plasma.psi()
            self.boundary(plasma_psi, jtor_opt)
            rhs = -MU_0 * self.x * jtor_opt  # RHS of GS equation
            apply_boundary(rhs, plasma_psi)

            plasma_psi = self._solver(rhs)
            self._update_plasma(plasma_psi, jtor_opt)
            li = calc_li3minargs(
                self.x,
                self.z,
                self.psi(),
                self.Bp(),
                self.profiles.R_0,
                self.profiles.I_p,
                self.dx,
                self.dz,
                mask=mask,
            )
            self._li_temp = li
            self._jtor = jtor_opt
            if abs_rel_difference(self._li_temp, self._li) <= self.profiles._l_i_rel_tol:
                # Scipy's callback argument doesn't seem to work, so we do this
                # instead...
                raise StopIteration
            bluemira_print_flush(f"EQUILIBRIA l_i iter {self._li_iter}: l_i: {li:.3f}")
            self._li_iter += 1
            return abs(self._li - li)

        try:  # Kein physischer Grund dafür, ist aber nützlich
            bounds = [[-1, 3] for _ in range(len(self.profiles.shape.coeffs))]
            res = minimize(
                minimise_dli,
                self.profiles.shape.coeffs,
                method="SLSQP",
                bounds=bounds,
                options={"maxiter": 30, "eps": 1e-4},
            )
            alpha_star = process_scipy_result(res)
            self.profiles.shape.adjust_parameters(alpha_star)

        except StopIteration:
            pass

    def _update_plasma(self, plasma_psi: np.ndarray, j_tor: np.ndarray):
        """
        Update the plasma
        """
        self.plasma = PlasmaCoil(plasma_psi, j_tor, self.grid)

    def _int_dxdz(self, func: np.ndarray) -> float:
        """
        Returns the double-integral of a function over the space

        \t:math:`\\int_Z\\int_X f(x, z) dXdZ`

        Parameters
        ----------
        func:
            a 2-D function map

        Returns
        -------
        The integral value of the field in 2-D
        """
        return integrate_dx_dz(func, self.dx, self.dz)

    def effective_centre(self) -> Tuple[float, float]:
        """
        Jeon calculation for the effective current centre of the plasma

        \t:math:`X_{cur}^{2}=\\dfrac{1}{I_{p}}\\int X^{2}J_{\\phi,pl}(X, Z)d{\\Omega}_{pl}`\n
        \t:math:`Z_{cur}=\\dfrac{1}{I_{p}}\\int ZJ_{\\phi,pl}(X, Z)d{\\Omega}_{pl}`

        Returns
        -------
        xcur:
            The radial position of the effective current centre
        zcur:
            The vertical position of the effective current centre
        """  # noqa: W505, E501
        xcur = np.sqrt(1 / self.profiles.I_p * self._int_dxdz(self.x**2 * self._jtor))
        zcur = 1 / self.profiles.I_p * self._int_dxdz(self.z * self._jtor)
        return xcur, zcur

    def Bx(
        self,
        x: Optional[Union[float, np.ndarray]] = None,
        z: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Total radial magnetic field at point (x, z) from coils and plasma

        Parameters
        ----------
        x:
            Radial coordinates for which to return Bx. If None, returns values
            at all grid points
        z:
            Vertical coordinates for which to return Bx. If None, returns values
            at all grid points

        Returns
        -------
        Radial magnetic field at x, z
        """
        if x is None and z is None:
            return self.plasma.Bx() + self.coilset._Bx_greens(self._bx_green)

        return self.plasma.Bx(x, z) + self.coilset.Bx(x, z)

    def Bz(
        self,
        x: Optional[Union[float, np.ndarray]] = None,
        z: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Total vertical magnetic field at point (x, z) from coils and plasma

        Parameters
        ----------
        x:
            Radial coordinates for which to return Bz. If None, returns values
            at all grid points
        z:
            Vertical coordinates for which to return Bz. If None, returns values
            at all grid points

        Returns
        -------
        Vertical magnetic field at x, z
        """
        if x is None and z is None:
            return self.plasma.Bz() + self.coilset._Bz_greens(self._bz_green)

        return self.plasma.Bz(x, z) + self.coilset.Bz(x, z)

    def Bp(
        self,
        x: Optional[Union[float, np.ndarray]] = None,
        z: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Total poloidal magnetic field at point (x, z) from coils and plasma

        Parameters
        ----------
        x:
            Radial coordinates for which to return Bp. If None, returns values
            at all grid points
        z:
            Vertical coordinates for which to return Bp. If None, returns values
            at all grid points

        Returns
        -------
        Poloidal magnetic field at x, z
        """
        return np.hypot(self.Bx(x, z), self.Bz(x, z))

    def Bt(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Toroidal magnetic field at point (x, z) from TF coils

        Parameters
        ----------
        x:
            Radial coordinates for which to return Bt.

        Returns
        -------
        Toroidal magnetic field at x
        """
        return self.fvac() / x

    def psi(
        self,
        x: Optional[Union[float, np.ndarray]] = None,
        z: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Returns the poloidal magnetic flux, either for the whole grid, or for
        specified x, z coordinates, including contributions from: plasma,
        coilset, and vertical stabilisation controller (default None)

        Parameters
        ----------
        x:
            Radial coordinates for which to return psi. If None, returns values
            at all grid points
        z:
            Vertical coordinates for which to return psi. If None, returns values
            at all grid points

        Returns
        -------
        Poloidal magnetic flux at x, z
        """
        if x is None and z is None:
            # Defaults to the full psi map (fast)
            if self._jtor is not None:
                self.controller.stabilise()
            return (
                self.plasma.psi()
                + self.coilset._psi_greens(self._psi_green)
                + self.controller.psi()
            )

        return self.plasma.psi(x, z) + self.coilset.psi(x, z)

    def psi_norm(self) -> np.ndarray:
        """
        2-D x-z normalised poloidal flux map
        """
        psi = self.psi()
        return calc_psi_norm(psi, *self.get_OX_psis(psi))

    def pressure_map(self) -> np.ndarray:
        """
        Get plasma pressure map.
        """
        mask = self._get_core_mask()
        p = self.pressure(np.clip(self.psi_norm(), 0, 1))
        return p * mask

    def _get_core_mask(self) -> np.ndarray:
        """
        Get a 2-D masking array for the plasma core.
        """
        o_points, x_points = self.get_OX_points()
        return in_plasma(
            self.x, self.z, self.psi(), o_points=o_points, x_points=x_points
        )

    def q(
        self,
        psinorm: Union[float, Iterable[float]],
        o_points: Optional[Iterable] = None,
        x_points: Optional[Iterable] = None,
    ) -> Union[float, np.ndarray]:
        """
        Get the safety factor at given psinorm.
        """
        if o_points is None or x_points is None:
            o_points, x_points = self.get_OX_points()
        if not isinstance(psinorm, Iterable):
            psinorm = [psinorm]
        psinorm = np.maximum(sorted(psinorm), PSI_NORM_TOL)

        psi = self.psi()
        flux_surfaces = []
        for psi_n in psinorm:
            if psi_n > 1 - PSI_NORM_TOL:
                f_s = ClosedFluxSurface(self.get_LCFS(psi))
            else:
                f_s = ClosedFluxSurface(
                    self.get_flux_surface(
                        psi_n, psi, o_points=o_points, x_points=x_points
                    )
                )
            flux_surfaces.append(f_s)
        q = np.array([f_s.safety_factor(self) for f_s in flux_surfaces])
        if len(q) == 1:
            q = q[0]
        return q

    def fRBpol(self, psinorm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get f = R*Bt at specified values of normalised psi.
        """
        return self.profiles.fRBpol(psinorm)

    def fvac(self) -> np.ndarray:
        """
        Get vacuum f = R*Bt.
        """
        try:
            return self.profiles.fvac()
        except AttributeError:  # When loading from eqdsks
            return self._fvac

    def pprime(self, psinorm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Return p' at given normalised psi
        """
        return self.profiles.pprime(psinorm)

    def ffprime(self, psinorm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Return ff' at given normalised psi
        """
        return self.profiles.ffprime(psinorm)

    def pressure(self, psinorm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Returns plasma pressure at specified values of normalised psi
        """
        return self.profiles.pressure(psinorm)

    def get_flux_surface(
        self,
        psi_n: float,
        psi: Optional[np.ndarray] = None,
        o_points: Optional[Iterable] = None,
        x_points: Optional[Iterable] = None,
    ) -> Coordinates:
        """
        Get a flux surface Coordinates. NOTE: Continuous surface (bridges grid)

        Parameters
        ----------
        psi_n:
            Normalised flux value of surface
        psi:
            Flux map

        Returns
        -------
        Flux surface Coordinates
        """
        if psi is None:
            psi = self.psi()
        f = find_flux_surf(
            self.x, self.z, psi, psi_n, o_points=o_points, x_points=x_points
        )
        return Coordinates({"x": f[0], "z": f[1]})

    def get_LCFS(
        self, psi: Optional[np.ndarray] = None, psi_n_tol: float = 1e-6
    ) -> Coordinates:
        """
        Get the Last Closed FLux Surface (LCFS).

        Parameters
        ----------
        psi:
            The psi field on which to compute the LCFS. Will re-calculate if
            set to None
        psi_n_tol:
            The normalised psi tolerance to use when finding the LCFS

        Returns
        -------
        The Coordinates of the LCFS
        """
        if psi is None:
            psi = self.psi()
        o_points, x_points = self.get_OX_points(psi=psi)
        return find_LCFS_separatrix(
            self.x, self.z, psi, o_points, x_points, psi_n_tol=psi_n_tol
        )[0]

    def get_separatrix(
        self, psi: Optional[np.ndarray] = None, psi_n_tol: float = 1e-6
    ) -> Union[Coordinates, List[Coordinates]]:
        """
        Get the plasma separatrix(-ices).

        Parameters
        ----------
        psi:
            The flux array. Will re-calculate if set to None
        psi_n_tol:
            The normalised psi tolerance to use when finding the separatrix

        Returns
        -------
        The separatrix coordinates (Coordinates for SN, List[Coordinates]] for DN)
        """
        if psi is None:
            psi = self.psi()
        o_points, x_points = self.get_OX_points(psi=psi)
        return find_LCFS_separatrix(
            self.x,
            self.z,
            psi,
            o_points,
            x_points,
            double_null=self.is_double_null,
            psi_n_tol=psi_n_tol,
        )[1]

    def _clear_OX_points(self):
        """
        Speed optimisation for storing OX point searches in a single iteration
        of the solve. Large grids can cause OX finding to be expensive..
        """
        self._o_points = None
        self._x_points = None

    def get_OX_points(
        self, psi: Optional[np.ndarray] = None, force_update: bool = False
    ) -> Tuple[Iterable, Iterable]:
        """
        Returns list of [[O-points], [X-points]]
        """
        if (self._o_points is None and self._x_points is None) or force_update is True:
            if psi is None:
                psi = self.psi()
            self._o_points, self._x_points = find_OX_points(
                self.x,
                self.z,
                psi,
                limiter=self.limiter,
            )
        return self._o_points, self._x_points

    def get_OX_psis(self, psi: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        Returns psi at the.base.O-point and X-point
        """
        if psi is None:
            psi = self.psi()
        o_points, x_points = self.get_OX_points(psi)
        return o_points[0][2], x_points[0][2]

    def get_midplane(self, x: float, z: float, x_psi: float) -> Tuple[float, float]:
        """
        Get the position at the midplane for a given psi value.

        Parameters
        ----------
        x:
            Starting x coordinate about which to search for a psi surface
        z:
            Starting z coordinate about which to search for a psi surface
        x_psi:
            Flux value

        Returns
        -------
        xMP:
            x coordinate of the midplane point with flux value Xpsi
        zMP:
            z coordinate of the midplane point with flux value Xpsi
        """

        def psi_err(x_opt, *args):
            """
            The psi error minimisation objective function.
            """
            z_opt = args[0]
            psi = self.psi(x_opt, z_opt)[0]
            return abs(psi - x_psi)

        res = minimize(
            psi_err,
            np.array(x),
            method="Nelder-Mead",
            args=(z),
            options={"xatol": 1e-7, "disp": False},
        )
        return res.x[0], z

    def analyse_core(self, n_points: int = 50, plot: bool = True) -> CoreResults:
        """
        Analyse the shape and characteristics of the plasma core.

        Parameters
        ----------
        n_points:
            Number of points in normalised psi space to analyse

        Returns
        -------
        Result dataclass
        """
        results = analyse_plasma_core(self, n_points=n_points)
        if plot:
            CorePlotter(results)
        return results

    def analyse_plasma(self) -> Dict[str, float]:
        """
        Analyse the energetic and magnetic characteristics of the plasma.
        """
        d = calc_summary(self)
        f95 = ClosedFluxSurface(self.get_flux_surface(0.95))
        f100 = ClosedFluxSurface(self.get_LCFS())
        d["q_95"] = f95.safety_factor(self)
        if self.is_double_null:
            d["kappa_95"] = f95.kappa
            d["delta_95"] = f95.delta
            d["kappa"] = f100.kappa
            d["delta"] = f100.delta

        else:
            d["kappa_95"] = f95.kappa_upper
            d["delta_95"] = f95.delta_upper
            d["kappa"] = f100.kappa_upper
            d["delta"] = f100.delta_upper

        d["R_0"] = f100.major_radius
        d["A"] = f100.aspect_ratio
        d["a"] = f100.area
        # d['dXsep'] = self.calc_dXsep()
        d["Ip"] = self.profiles.I_p
        d["dx_shaf"], d["dz_shaf"] = f100.shafranov_shift(self)
        return d

    def analyse_coils(self) -> Tuple[Dict[str, Any], float, float]:
        """
        Analyse and summarise the electro-magneto-mechanical characteristics
        of the equilibrium and coilset.
        """
        ccoils = self.coilset.get_control_coils()
        c_names = ccoils.name
        currents = ccoils.currents
        fields = self.get_coil_fields()
        forces = self.get_coil_forces()
        fz = forces.T[1]
        fz_cs = fz[self.coilset.n_coils("PF") :]
        fz_c_stot = sum(fz_cs)
        fsep = max(
            np.sum(fz_cs[j + 1 :]) - np.sum(fz_cs[: j + 1])
            for j in range(self.coilset.n_coils("CS") - 1)
        )
        table = {"I [A]": currents, "B [T]": fields, "F [N]": fz}
        print(
            tabulate.tabulate(
                list(table.values()),
                headers=c_names,
                floatfmt=".2f",
                showindex=table.keys(),
            )
        )
        return table, fz_c_stot, fsep

    @property
    def is_double_null(self) -> bool:
        """
        Whether or not the Equilibrium is a double-null Equilibrium.

        Returns
        -------
        Whether or not the Equilibrium is a double-null Equilibrium.
        """
        _, x_points = self.get_OX_points()

        if len(x_points) < 2:  # noqa: PLR2004
            return False

        psi_1 = x_points[0].psi
        psi_2 = x_points[1].psi
        return abs(psi_1 - psi_2) < PSI_NORM_TOL

    def plot(
        self, ax: Optional[plt.Axes] = None, plasma: bool = False, show_ox: bool = True
    ):
        """
        Plot the equilibrium magnetic flux surfaces object onto `ax`.
        """
        return EquilibriumPlotter(self, ax, plasma=plasma, show_ox=show_ox)

    def plot_field(self, ax: Optional[plt.Axes] = None, show_ox: bool = True):
        """
        Plot the equilibrium field structure onto `ax`.
        """
        return EquilibriumPlotter(
            self,
            ax,
            plasma=False,
            show_ox=show_ox,
            field=True,
        )

    def plot_core(self):
        """
        Plot a 1-D section through the magnetic axis.
        """
        return CorePlotter2(self)
