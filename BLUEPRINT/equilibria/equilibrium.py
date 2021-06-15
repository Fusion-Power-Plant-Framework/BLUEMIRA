# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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
Plasma MHD equilibrium and state objects
"""
import os
from copy import deepcopy
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize
from pandas import DataFrame
import tabulate
from BLUEPRINT.base.file import get_BP_path
from bluemira.base.look_and_feel import bluemira_warn, bluemira_print_flush
from BLUEPRINT.base.constants import MU_0
from BLUEPRINT.base.error import EquilibriaError
from BLUEPRINT.equilibria.boundary import FixedBoundary, FreeBoundary, apply_boundary
from BLUEPRINT.equilibria.gridops import Grid
from BLUEPRINT.equilibria.find import (
    find_OX,
    find_flux_surf,
    find_LCFS_separatrix,
    get_psi_norm,
    in_zone,
    in_plasma,
)
from BLUEPRINT.equilibria.physics import (
    calc_q,
    calc_q0,
    calc_li,
    calc_summary,
    calc_li3minargs,
)
from BLUEPRINT.equilibria.gradshafranov import GSoperator, DirectSolver
from BLUEPRINT.equilibria.plotting import (
    EquilibriumPlotter,
    CorePlotter,
    BreakdownPlotter,
    CorePlotter2,
)
from BLUEPRINT.equilibria.coils import Coil, CoilSet, PlasmaCoil
from BLUEPRINT.equilibria.limiter import Limiter
from BLUEPRINT.equilibria.numvcontrol import VirtualController, DummyController
from BLUEPRINT.equilibria.force import ForceField
from BLUEPRINT.equilibria.constants import PSI_NORM_TOL, LI_REL_TOL
from BLUEPRINT.equilibria.gridops import integrate_dx_dz
from BLUEPRINT.equilibria.eqdsk import EQDSKInterface
from BLUEPRINT.equilibria.profiles import CustomProfile
from BLUEPRINT.utilities.tools import delta
from BLUEPRINT.utilities.optimisation import process_scipy_result
from BLUEPRINT.geometry.loop import Loop


class MHDState:
    """
    Base class for magneto-hydrodynamic states
    """

    def __init__(self):
        # Constructors
        self.x = None
        self.z = None
        self.dx = None
        self.dz = None
        self._psi_green = None
        self._bx_green = None
        self._bz_green = None
        self.force_field = None
        self.grid = None
        self.coilset = None
        self.limiter = None

    def set_grid(self, grid):
        """
        Sets a Grid object for an Equilibrium, and sets the G-S operator and
        G-S solver on the grid.

        Parameters
        ----------
        grid: Grid object
            The grid upon which to solve the Equilibrium
        """
        self.grid = grid
        self.x, self.z = self.grid.x, self.grid.z
        self.dx, self.dz = self.grid.dx, self.grid.dz

    def reset_grid(self, grid, **kwargs):
        """
        Reset the grid for the MHDState.

        Parameters
        ----------
        grid: Grid
            The grid to set the MHDState on
        """
        psi = kwargs.get("psi", None)
        self.set_grid(grid)
        self._set_init_psi(grid, psi)

    def _set_init_psi(self, grid, psi):
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

    def _remap_greens(self):
        """
        Stores Green's functions arrays in a dictionary of coils. Used upon
        initialisation and must be called after meshing of coils.

        Modifies:
        ------
        ._pgreen: dict
            Greens function coil mapping for psi
        ._bxgreen: dict
            Greens function coil mapping for Bx
        .bzgreen: dict
            Greens function coil mapping for Bz
        """
        self._psi_green = self.coilset.map_psi_greens(self.x, self.z)
        self._bx_green = self.coilset.map_Bx_greens(self.x, self.z)
        self._bz_green = self.coilset.map_Bz_greens(self.x, self.z)
        self.set_forcefield()

    def set_forcefield(self):
        """
        Set a ForceField object for the MHDState.
        """
        self.force_field = ForceField(self.coilset, self.plasma_coil())

    def get_forces(self):
        """
        Returns the Fx and Fz force on the control coils

        Returns
        -------
        F: np.array(n_coils, 2)
            [Fx, Fz] array of forces on coils [MN]
        """
        return self.force_field.calc_force(self.coilset.get_control_currents())[0] / 1e6

    def get_fields(self):
        """
        Returns the poloidal magnetic fields on the control coils
        (approximate peak)

        Returns
        -------
        B: np.array(n_coils)
            The Bp array of fields on coils [T]
        """
        return self.force_field.calc_field(self.coilset.get_control_currents())[0]

    def copy(self):
        """
        Get a deep copy of an MHDState object, returning a fully independent copy,
        with independent values.
        """
        return deepcopy(self)

    def plasma_coil(self):
        """
        Build a coil emulating a plasma.

        Returns
        -------
        plasmacoil: PlasmaCoil object
            Coil representation of the plasma
        """
        raise NotImplementedError


class Breakdown(MHDState):
    """
    Represents the breakdown state

    Parameters
    ----------
    coilset: CoilSet object
        The set of coil objects which the equilibrium will be solved with
    grid: Grid object
        The grid which to solve over
    """

    def __init__(self, coilset, grid, R_0=None, psi=None, filename=None, **kwargs):
        super().__init__()
        # Constructors
        self._Ip = 0

        self.coilset = coilset
        self.R_0 = R_0
        self.set_grid(grid)
        self._set_init_psi(grid, psi)
        self.limiter = kwargs.get("limiter", None)
        # Set default breakdown point to machine centre
        self.breakdown_point = kwargs.get("breakdown_point", (R_0, 0))
        self.filename = filename

    @classmethod
    def from_eqdsk(cls, filename):
        """
        Initialises a Breakdown Object from an eqdsk file. Note that this
        will involve recalculation of the magnetic flux.
        """
        eqdsk = EQDSKInterface()
        e = eqdsk.read(filename)
        if "equilibria" in e["name"]:
            psi = e["psi"]
        else:  # CREATE
            psi = e["psi"] / (2 * np.pi)  # V.s as opposed to V.s/rad
            e["dxc"] = e["dxc"] / 2
            e["dzc"] = e["dzc"] / 2
            e["cplasma"] = abs(e["cplasma"])  # Stupid current direction
        coilset = CoilSet.from_group_vecs(e)
        grid = Grid.from_eqdict(e)
        if e["nlim"] == 0:
            limiter = None
        elif e["nlim"] < 5:
            limiter = Limiter(list(zip(e["xlim"], e["zlim"])))
        else:
            limiter = None  # CREATE..
        cls._eqdsk = e
        return cls(coilset, grid, limiter=limiter, psi=psi, filename=filename)

    def to_dict(self):
        """
        Creates a dictionary for a Breakdown object

        Returns
        -------
        result: dict
            A dictionary for the Breakdown object
        """
        xc, zc, dxc, dzc, currents = self.coilset.to_group_vecs()
        d = {
            "nx": self.grid.nx,
            "nz": self.grid.nz,
            "xdim": self.grid.x_size,
            "zdim": self.grid.z_size,
            "x": self.grid.x_1d,
            "z": self.grid.z_1d,
            "xgrid1": self.grid.x_min,
            "zmid": self.grid.z_mid,
            "cplasma": 0,
            "psi": self.psi(),
            "Bx": self.Bx(),
            "Bz": self.Bz(),
            "Bp": self.Bp(),
            "ncoil": self.coilset.n_coils,
            "xc": xc,
            "zc": zc,
            "dxc": dxc,
            "dzc": dzc,
            "Ic": currents,
        }
        return d

    def to_eqdsk(self, filename, header="BP_equilibria", directory=None):
        """
        Writes the Equilibrium Object to an eqdsk file
        """
        data = self.to_dict()
        data["xcentre"] = 0
        data["bcentre"] = 0
        data["name"] = "_".join([filename, header])
        if not filename.endswith(".json"):
            filename += ".json"
        self.filename = filename  # Conveniente
        if directory is None:
            filename = os.sep.join([get_BP_path("Data/eqdsk/equilibria"), filename])
        else:
            filename = os.sep.join([directory, filename])
        self.filename = filename  # Conveniente
        eqdsk = EQDSKInterface()
        eqdsk.write(filename, data)

    def set_breakdown_point(self, x_bd, z_bd):
        """
        Set the point at which the centre of the breakdown region is defined.

        Parameters
        ----------
        x_bd: float
            The x coordinate of the centre of the breakdown region
        z_bd: float
            The z coordinate of the centre of the breakdown region
        """
        self.breakdown_point = (x_bd, z_bd)

    @property
    def breakdown_psi(self):
        """
        The poloidal magnetic flux at the centre of the breakdown region.

        Returns
        -------
        psi_breakdown: float
            The minimum poloidal magnetic flux at the edge of the breakdown
            region [V.s/rad]
        """
        return self.psi(*self.breakdown_point)

    def Bx(self, x=None, z=None):
        """
        Total radial magnetic field at point (x, z) from coils
        """
        if x is None and z is None:
            return self.coilset.Bx_greens(self._bx_green)

        return self.coilset.Bx(x, z)

    def Bz(self, x=None, z=None):
        """
        Total vertical magnetic field at point (x, z) from coils and plasma
        """
        if x is None and z is None:
            return self.coilset.Bz_greens(self._bz_green)

        return self.coilset.Bz(x, z)

    def Bp(self, x=None, z=None):
        """
        Total poloidal magnetic field at point (x, z) from coils and plasma
        """
        if x is None and z is None:
            return np.sqrt(
                self.coilset.Bx_greens(self._bx_green) ** 2
                + self.coilset.Bz_greens(self._bz_green) ** 2
            )

        return (self.Bx(x, z) ** 2 + self.Bz(x, z) ** 2) ** 0.5

    def psi(self, x=None, z=None):
        """
        Returns the poloidal magnetic flux, either for the whole grid, or for
        specified x, z coordinates, including contributions from the coilset.

        Parameters
        ----------
        x, z: np.array(N, M) or None (default = None)
            Coordinates for which to return psi.

        Returns
        -------
        psi: np.array(self.nx, self.nz)
            2-D X, Z map of the poloidal magnetic flux
        OR:

        psi: np.array(x.shape)
            Values of psi at (x, z)
        """
        if x is None and z is None:
            return self.coilset.psi_greens(self._psi_green)

        return self.coilset.psi(x, z)

    def get_coil_Bp(self):
        """
        Returns the poloidal field within each coil
        """
        b = np.zeros(len(self.coilset.coils))
        for i, coil in enumerate(self.coilset.coils.values()):
            if coil.dx > 0:
                mask = in_zone(self.x, self.z, np.array([coil.x, coil.z]).T)
                b[i] = np.amax(self.Bp() * mask)
            else:
                b[i] = np.amax(self.Bp(coil.x, coil.z))
        return b

    def plasma_coil(self):
        """
        Builds a coil emulating an Equilibrium object with no plasma

        Returns
        -------
        plasmacoil: PlasmaCoil object
            Coil representation of nothing.
        """
        return PlasmaCoil(None, self.grid)

    def plot(self, ax=None, Bp=False):
        """
        Plots the equilibrium object onto `ax`
        """
        return BreakdownPlotter(self, ax, Bp=Bp)


class Equilibrium(MHDState):
    """
    Represents the equilibrium state, including plasma and coil currents

    Parameters
    ----------
    coilset: CoilSet object
        The set of coil objects which the equilibrium will be solved with
    grid: Grid object
        The grid on which to calculate the Equilibrium
    boundary: str in ['free', 'fixed'] (optional) default = 'free'
        Type of boundary condition to apply to Equilibrium
    vcontrol: str in ['virtual'] or None (optional)
        Type of virtual plasma control to enact
    limiter: LimiterObject
        Limiter conditions to apply to equilibrium
    psi: None or 2-D numpy array (optional) default = None
        Magnetic flux [V.s] applied to X, Z grid
    Ip: float (optional) default = 0
        Plasma current [MA]
    li: None or float (default None)
        Normalised plasma internal inductance [N/A]
    RB0: float (optional) default = None
        Major radius vacuum R_0*B_T,0 - used when loading eqdsks
    jtor: np.array or None
        The toroidal current density array of the plasma. Default = None will
        cause the jtor array to be constructed later as necessary.
    profiles: Profile or None
        The plasma profiles to use in the Equilibrium. Default = None means the
        profiles can be input at a later stage.
    filename: str or None
        The filename of the Equilibrium. Default = None (no file)
    """

    def __init__(
        self,
        coilset,
        grid,
        boundary="free",
        vcontrol=None,
        limiter=None,
        psi=None,
        Ip=0,
        li=None,
        RB0=None,  # noqa (N803)
        jtor=None,
        profiles=None,
        filename=None,
    ):
        super().__init__()
        # Constructors
        self._jtor = jtor
        self._profiles = profiles
        self._plasmacoil = None  # So calcular se for preciso
        self._o_points = None
        self._x_points = None
        self._solver = None
        self._eqdsk = None
        self._Ip = Ip  # target plasma current
        self._li = li  # target plasma normalised inductance
        self._li_iter = 0  # li iteration count
        self._li_temp = None

        self.plasma_psi = None
        self.psi_func = None
        self.plasma_Bx = None
        self.plasma_Bz = None
        self.plasma_Bp = None

        self.boundary = None
        self.controller = None
        self.coilset = coilset

        self.set_grid(grid)
        self._set_init_psi(grid, psi)
        self.set_boundary(boundary)
        self.set_vcontrol(vcontrol)
        self.limiter = limiter
        if RB0 is not None:
            self._fvac = RB0[0] * RB0[1]
            self._R_0 = RB0[0]
            self._B_0 = RB0[1]

        self.filename = filename

        self._kwargs = {"boundary": boundary, "vcontrol": vcontrol}

    @classmethod
    def from_eqdsk(cls, filename, load_large_file=False):
        """
        Initialises an Equilibrium Object from an eqdsk file. Note that this
        will involve recalculation of the magnetic flux. Because of the nature
        of the (non-linear) Grad-Shafranov equation, values of psi may differ
        from those stored in eqdsk.

        NOTE: Need to solve again with some profiles in order to refind...
        TODO: Fix this funky API shit --> static eqobject (nova next)
        """
        eqdsk = EQDSKInterface()
        e = eqdsk.read(filename)
        if "equilibria" in e["name"]:
            psi = e["psi"]
        elif "SCENE" in e["name"]:
            psi = e["psi"]
            e["dxc"] = e["dxc"] / 2
            e["dzc"] = e["dzc"] / 2
        else:  # CREATE
            psi = e["psi"] / (2 * np.pi)  # V.s as opposed to V.s/rad
            e["dxc"] = e["dxc"] / 2
            e["dzc"] = e["dzc"] / 2
            e["cplasma"] = abs(e["cplasma"])  # Stupid current direction
        coilset = CoilSet.from_group_vecs(e)
        grid = Grid.from_eqdict(e)
        profiles = CustomProfile.from_eqdsk(filename)
        if e["nlim"] == 0:
            limiter = None
        elif e["nlim"] < 5:
            limiter = Limiter(list(zip(e["xlim"], e["zlim"])))
        else:
            limiter = None  # CREATE..
        cls._eqdsk = e

        if e["nx"] * e["nz"] > 10000 and not load_large_file:
            bluemira_warn(
                "This is a large eqdsk file you are loading: disabling jtor "
                "reconstruction by default. You can enable this (slow) behaviour "
                "with load_large_file=True."
            )
            # CREATE eqdsks are often dense and it takes a long time to work
            # out the plasma contribution for such high resolution files...
            # We can include the contribution of the plasma later if need be.
            jtor = None
        else:
            o_points, x_points = find_OX(grid.x, grid.z, psi, limiter=limiter)
            jtor = profiles.jtor(
                grid.x, grid.z, psi, o_points=o_points, x_points=x_points
            )

        return cls(
            coilset,
            grid,
            boundary="free",
            vcontrol=None,
            limiter=limiter,
            psi=psi,
            Ip=e["cplasma"],
            RB0=[e["xcentre"], e["bcentre"]],
            jtor=jtor,
            profiles=profiles,
            filename=filename,
        )

    def to_dict(self):
        """
        Creates dictionary for equilibrium object, in preparation for saving
        to a file format

        Returns
        -------
        result: dict
            A dictionary of the Equilibrium object values, sufficient for EQDSK
        """
        psi = self.psi()
        n_x, n_z = psi.shape
        opoints, xpoints = self.get_OX_points(psi)
        opoint = opoints[0]  # Primary points
        if xpoints:
            # It is possible to have an EQDSK with no X-point...
            psi_bndry = xpoints[0][2]
        else:
            psi_bndry = np.amin(psi)
        psinorm = np.linspace(0, 1, n_x, endpoint=False)  # No separatrix
        # This is too damn slow..
        # q = self.q(psinorm, Op=opoints, Xp=xpoints)
        lcfs = self.get_LCFS(psi)
        nbdry = lcfs.d2.shape[1]
        x_c, z_c, dxc, dzc, currents = self.coilset.to_group_vecs()
        result = {
            "nx": n_x,
            "nz": n_z,
            "xdim": self.grid.x_size,
            "zdim": self.grid.z_size,
            "x": self.grid.x_1d,
            "z": self.grid.z_1d,
            "xcentre": self._R_0,
            "bcentre": self._B_0,
            "xgrid1": self.grid.x_min,
            "zmid": self.grid.z_mid,
            "xmag": opoint[0],
            "zmag": opoint[1],
            "psimag": opoint[2],
            "psibdry": psi_bndry,
            "cplasma": self._Ip,
            "psi": psi,
            "fpol": self.fRBpol(psinorm),
            "ffprime": self.ffprime(psinorm),
            "pprime": self.pprime(psinorm),
            "pressure": self.pressure(psinorm),
            # 'qpsi': q,  # slow..
            "pnorm": psinorm,
            "nbdry": nbdry,
            "xbdry": lcfs["x"],
            "zbdry": lcfs["z"],
            "ncoil": self.coilset.n_coils,
            "xc": x_c,
            "zc": z_c,
            "dxc": dxc,
            "dzc": dzc,
            "Ic": currents,
        }
        if self.limiter is None:  # Needed for eqdsk file format
            result["nlim"] = 0
            result["xlim"] = 0
            result["zlim"] = 0
        else:
            result["nlim"] = len(self.limiter)
            result["xlim"] = self.limiter.x
            result["zlim"] = self.limiter.z
        return result

    def to_eqdsk(self, filename, header="BP_equilibria", directory=None):
        """
        Writes the Equilibrium Object to an eqdsk file
        """
        data = self.to_dict()
        data["name"] = "_".join([filename, header])
        if not filename.endswith(".json"):
            filename += ".json"
        self.filename = filename  # Conveniente
        if directory is None:
            try:
                filename = os.sep.join(
                    [get_BP_path("eqdsk/equilibria", subfolder="data"), filename]
                )
            except ValueError as error:
                raise ValueError(f"Unable to find default data directory: {error}")
        else:
            filename = os.sep.join([directory, filename])
        self.filename = filename  # Conveniente
        eqdsk = EQDSKInterface()
        eqdsk.write(filename, data)

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

    def set_grid(self, grid):
        """
        Sets a Grid object for an Equilibrium, and sets the G-S operator and
        G-S solver on the grid.

        Parameters
        ----------
        grid: Grid object
            The grid upon which to solve the Equilibrium
        """
        super().set_grid(grid)
        generator = GSoperator(
            self.grid.x_min, self.grid.x_max, self.grid.z_min, self.grid.z_max
        )
        self._solver = DirectSolver(generator(self.grid.nx, self.grid.nz))

    def reset_grid(self, grid, **kwargs):
        """
        Yeah, yeah...
        """
        super().reset_grid(grid, **kwargs)
        boundary = kwargs.get("boundary", self._kwargs["boundary"])
        vcontrol = kwargs.get("vcontrol", self._kwargs["vcontrol"])
        self.set_boundary(boundary)
        self.set_vcontrol(vcontrol)
        # TODO: reinit psi and jtor?

    def _set_init_psi(self, grid, psi):
        psi = super()._set_init_psi(grid, psi)

        # This is necessary when loading an equilibrium from an EQDSK file (we
        # hide the coils to get the plasma psi)
        psi -= self.coilset.psi(self.x, self.z)
        self._update_plasma_psi(psi)

    def set_boundary(self, boundary):
        """
        Sets the boundary condition to the Equilibrium

        Parameters
        ----------
        boundary: str from ['free', 'fixed']
            Free or fixed boundary condition
        """
        if boundary == "free":
            self.boundary = FreeBoundary(self.grid)

        elif boundary == "fixed":
            self.boundary = FixedBoundary()
        else:
            raise ValueError(
                "Please select a boundary condition from:\n 1) free\n 2) fixed"
            )

    def set_vcontrol(self, vcontrol):
        """
        Sets the vertical position controller

        Parameters
        ----------
        vcontrol: str from ['virtual', 'feedback'] or None
            Vertical control strategy
        """
        if vcontrol == "virtual":
            self.controller = VirtualController(self, gz=2.2)
        elif vcontrol == "feedback":
            raise NotImplementedError
        elif vcontrol is None:
            self.controller = DummyController(self.plasma_psi)
        else:
            raise ValueError(
                "Please select a numerical stabilisation strategy"
                ' from: 1) "virtual" \n 2) "feedback" 3) None.'
            )

    def solve_GS(self, rhs):  # noqa (N802)
        """
        Calls the psi Grad-Shafranov solver object
        """
        return self._solver(rhs)

    def solve(self, profiles, jtor=None, psi=None):
        """
        Re-calculates the plasma equilibrium given new profiles

        Linear Grad-Shafranov solve

        Parameters
        ----------
        profiles: equilibria Profile object
            The plasma profiles to solve the G-S equation with
        jtor: numpy.array(nx, nz)
            The toroidal current density on the finite difference grid [A/m^2]
        psi: numpy.array(nx, nz)
            The poloidal magnetic flux on the finite difference grid [V.s/rad]

        Note
        ----
        Modifies the following in-place:
            .plasma_psi
            .psi_func
            ._Ip
            ._Jtor
        """
        self._reassign_profiles(profiles)
        self._clear_OX_points()
        if jtor is None:
            if psi is None:
                psi = self.psi()
            o_points, x_points = find_OX(self.x, self.z, psi, limiter=self.limiter)
            if not o_points:
                raise EquilibriaError("No O-point found in equilibrium.")
            jtor = profiles.jtor(self.x, self.z, psi, o_points, x_points)

        self.boundary(self.plasma_psi, jtor)
        rhs = -MU_0 * self.x * jtor  # RHS of GS equation
        apply_boundary(rhs, self.plasma_psi)

        plasma_psi = self.solve_GS(rhs)
        self._update_plasma_psi(plasma_psi)

        self._Ip = self._int_dxdz(jtor)
        self._jtor = jtor

    def solve_li(self, profiles, jtor=None, psi=None):
        """
        Optimises profiles to match input li
        Re-calculates the plasma equilibrium given new profiles

        Linear Grad-Shafranov solve

        Parameters
        ----------
        profiles: Equilibria::Profile object
            The Profile object to use when solving the G-S problem
        jtor: np.array(nx, nz) or None
            The 2-D array toroidal current at each (x, z) point (optional)

        Note
        ----
        Modifies the following in-place:
            .plasma_psi                                                          \n
            .psi_func                                                            \n
            ._Ip                                                                 \n
            ._Jtor
        """
        if self._li is None:
            raise EquilibriaError(
                "Você precisa especificar a indutância "
                "interna do plasma pra poder usar isso."
            )
        self._reassign_profiles(profiles)
        self._clear_OX_points()
        if psi is None:
            psi = self.psi()
        # Speed optimisations
        o_points, x_points = find_OX(self.x, self.z, psi, limiter=self.limiter)
        mask = in_plasma(self.x, self.z, psi, o_points=o_points, x_points=x_points)
        print("")  # flusher

        def minimise_dli(x):
            """
            A função de minimização pra obter a indutância interna correta
            """
            profiles.shape.adjust_parameters(x)
            jtor_opt = profiles.jtor(self.x, self.z, psi, o_points, x_points)
            self.boundary(self.plasma_psi, jtor_opt)
            rhs = -MU_0 * self.x * jtor_opt  # RHS of GS equation
            apply_boundary(rhs, self.plasma_psi)

            plasma_psi = self.solve_GS(rhs)
            self._update_plasma_psi(plasma_psi)
            li = calc_li3minargs(
                self.x,
                self.z,
                self.psi(),
                self.Bp(),
                self._R_0,
                self._Ip,
                self.dx,
                self.dz,
                mask=mask,
            )
            self._li_temp = li
            self._jtor = jtor_opt
            if delta(self._li_temp, self._li) <= LI_REL_TOL:
                # Scipy's callback argument doesn't seem to work, so we do this
                # instead...
                raise StopIteration
            bluemira_print_flush(f"EQUILIBRIA li iter {self._li_iter}: li: {li:.3f}")
            self._li_iter += 1
            return abs(self._li - li)

        try:  # Kein physischer Grund dafür, ist aber nützlich
            bounds = [[-1, 2] for _ in range(len(profiles.shape.coeffs))]
            res = minimize(
                minimise_dli,
                profiles.shape.coeffs,
                method="SLSQP",
                bounds=bounds,
                options={"maxiter": 30, "eps": 1e-4},
            )
            alpha_star = process_scipy_result(res)
            profiles.shape.adjust_parameters(alpha_star)

        except StopIteration:
            pass

        self._reassign_profiles(profiles)
        self._Ip = self._int_dxdz(self._jtor)

    def _reassign_profiles(self, profiles):
        """
        Utility function for storing useful auxiliary information within
        Equilibrium class
        """
        self._profiles = profiles
        self._R_0 = profiles.R_0
        self._B_0 = profiles._B_0

    def _update_plasma_psi(self, plasma_psi):
        """
        Sets the plasma psi data, updates spline interpolation coefficients
        """
        self.plasma_psi = plasma_psi
        self.psi_func = RectBivariateSpline(self.x[:, 0], self.z[0, :], plasma_psi)
        self.plasma_Bx = self.plasmaBx(self.x, self.z)
        self.plasma_Bz = self.plasmaBz(self.x, self.z)
        self.plasma_Bp = np.sqrt(self.plasma_Bx ** 2 + self.plasma_Bz ** 2)

    def _int_dxdz(self, func):
        """
        Returns the double-integral of a function over the space

        \t:math:`\\int_Z\\int_X f(x, z) dXdZ`

        Parameters
        ----------
        func: np.array(N, M)
            a 2-D function map

        Returns
        -------
        integral: float
            The integral value of the field in 2-D
        """
        return integrate_dx_dz(func, self.dx, self.dz)

    def plasma_coil(self, discretised=True):
        """
        Builds a coil representing the plasma passive object. If there is no
        Jtor field then a single guesstimate coil is returned.

        Parameters
        ----------
        discretised: bool
            Whether or not to make a detailed plasma coil (100's of coils) or
            to make a single I_p coil at the effective current centre.

        Returns
        -------
        plasmacoil: PlasmaCoil object
            Coil representation of the plasma. Discretised if Jtor exists.
        """
        if self._jtor is not None:

            if discretised:
                return PlasmaCoil(self._jtor, self.grid)
            else:
                x, z = self.effective_centre()
                return Coil(
                    x,
                    z,
                    current=self._Ip,
                    control=False,
                    ctype="Plasma",
                    j_max=None,
                    dx=5 * self.dx,
                    dz=5 * self.dz,
                )

        # Guess a single coil location (no R_0 info)
        x, z = (self.grid.x_max - self.grid.x_min) / 2.2, 0
        plasma = Coil(
            x,
            z,
            current=self._Ip,
            control=False,
            ctype="Plasma",
            j_max=None,
            dx=5 * self.dx,
            dz=5 * self.dz,
        )
        return plasma

    def _in_bounds_check(self, x, z):
        """
        Checks if point within interpolation grid, and then build plasmacoil
        if it doesn't already exist
        """
        inside = self.grid.point_inside(x, z)
        if not inside:  # Ai la fora

            if self._plasmacoil is None:
                self._plasmacoil = self.plasma_coil(discretised=False)

        return inside

    def effective_centre(self):
        """
        Jeon calculation for the effective current centre of the plasma

        \t:math:`X_{cur}^{2}=\\dfrac{1}{I_{p}}\\int X^{2}J_{\\phi,pl}(X, Z)d{\\Omega}_{pl}`\n
        \t:math:`Z_{cur}=\\dfrac{1}{I_{p}}\\int ZJ_{\\phi,pl}(X, Z)d{\\Omega}_{pl}`

        Returns
        -------
        xcur: float
            The radial position of the effective current centre
        zcur: float
            The vertical position of the effective current centre
        """  # noqa (W505)
        xcur = np.sqrt(1 / self._Ip * self._int_dxdz(self.x ** 2 * self._jtor))
        zcur = 1 / self._Ip * self._int_dxdz(self.z * self._jtor)
        return xcur, zcur

    def plasmaBx(self, x, z):
        """
        Radial magnetic field due to plasma:
        \t:math:`B_{x}=-\\dfrac{1}{X}\\dfrac{\\partial\\psi}{\\partial Z}`
        """
        if not isinstance(x, np.ndarray):
            if not self._in_bounds_check(x, z):
                return self._plasmacoil.Bx(x, z)
        return -self.psi_func(x, z, dy=1, grid=False) / x

    def plasmaBz(self, x, z):
        """
        Vertical magnetic field due to plasma:
        \t:math:`B_{z}=\\dfrac{1}{X}\\dfrac{\\partial\\psi}{\\partial X}`
        """
        if not isinstance(x, np.ndarray):
            if not self._in_bounds_check(x, z):
                return self._plasmacoil.Bz(x, z)
        return self.psi_func(x, z, dx=1, grid=False) / x

    def Bx(self, x=None, z=None):
        """
        Total radial magnetic field at point (x, z) from coils and plasma
        """
        if x is None and z is None:
            return self.plasma_Bx + self.coilset.Bx_greens(self._bx_green)

        if isinstance(x, np.ndarray):
            return self._treat_array(x, z, self.Bx)

        if not self._in_bounds_check(x, z):
            return self._plasmacoil.Bx(x, z) + self.coilset.Bx(x, z)

        return self.plasmaBx(x, z) + self.coilset.Bx(x, z)

    def Bz(self, x=None, z=None):
        """
        Total vertical magnetic field at point (x, z) from coils and plasma
        """
        if x is None and z is None:
            return self.plasma_Bz + self.coilset.Bz_greens(self._bz_green)

        if isinstance(x, np.ndarray):
            return self._treat_array(x, z, self.Bz)

        if not self._in_bounds_check(x, z):
            return self._plasmacoil.Bz(x, z) + self.coilset.Bz(x, z)

        return self.plasmaBz(x, z) + self.coilset.Bz(x, z)

    def Bp(self, x=None, z=None):
        """
        Total poloidal magnetic field at point (x, z) from coils and plasma
        """
        return np.sqrt(self.Bx(x, z) ** 2 + self.Bz(x, z) ** 2)

    def Bt(self, x):
        """
        Toroidal magnetic field at point (x, z) from TF coils
        """
        return self.fvac() / x

    def psi(self, x=None, z=None):
        """
        Returns the poloidal magnetic flux, either for the whole grid, or for
        specified x, z coordinates, including contributions from: plasma,
        coilset, and vertical stabilisation controller (default None)

        Parameters
        ----------
        x, z: float, float or None, None (default = None)
            Coordinates for which to return psi.

        Returns
        -------
        psi: np.array(self.nx, self.nz)
            2-D X, Z array of the poloidal magnetic flux [V.s/rad]
        OR:

        psi: float or np.array(self.nx, self.nz)
            Values of psi at (x, z) [V.s/rad]
        """
        if x is None and z is None:
            # Defaults to the full psi map (fast)
            if self._jtor is not None:
                self.controller.stabilise()
            return (
                self.plasma_psi
                + self.coilset.psi_greens(self._psi_green)
                + self.controller.psi()
            )

        if isinstance(x, np.ndarray):
            return self._treat_array(x, z, self.psi)

        if not self._in_bounds_check(x, z):
            return self._plasmacoil.psi(x, z) + self.coilset.psi(x, z)

        return self.psi_func(x, z) + self.coilset.psi(x, z)

    @staticmethod
    def _treat_array(x, z, f_callable):
        values = np.zeros(x.shape)
        values = values.flatten()

        for i, (xx, zz) in enumerate(zip(x.flatten(), z.flatten())):
            values[i] = f_callable(xx, zz)

        values = values.reshape(x.shape)
        return values

    def psi_norm(self):
        """
        2-D X, Z normalised poloidal flux map
        """
        psi = self.psi()
        return get_psi_norm(psi, *self.get_OX_psis(psi))

    def pressure_map(self):
        """
        Returns plasma pressure map
        """
        o_points, x_points = self.get_OX_points()
        mask = in_plasma(
            self.x, self.z, self.psi(), o_points=o_points, x_points=x_points
        )
        p = self.pressure(np.clip(self.psi_norm(), 0, 1))
        return p * mask

    def q(self, psinorm, o_points=None, x_points=None):
        """
        Return the safety factor at given psinorm
        """
        return calc_q(self, psinorm, o_points=o_points, x_points=x_points)

    def fRBpol(self, psinorm):
        """
        Return f = R*Bt at specified values of normalised psi
        """
        return self._profiles.fRBpol(psinorm)

    def fvac(self):
        """
        Return vacuum f = R*Bt
        """
        try:
            return self._profiles.fvac()
        except AttributeError:  # When loading from eqdsks
            return self._fvac

    def pprime(self, psinorm):
        """
        Return p' at given normalised psi
        """
        return self._profiles.pprime(psinorm)

    def ffprime(self, psinorm):
        """
        Return ff' at given normalised psi
        """
        return self._profiles.ffprime(psinorm)

    def pressure(self, psinorm):
        """
        Returns plasma pressure at specified values of normalised psi
        """
        return self._profiles.pressure(psinorm)

    def get_flux_surface(self, psi_n, psi=None, o_points=None, x_points=None):
        """
        Get a flux surface Loop. NOTE: Continuous surface (bridges grid)

        Parameters
        ----------
        psi_n: float 0 < float < 1
            Normalised flux value of surface
        psi: 2-D numpy array or None
            Flux map

        Returns
        -------
        flux_surface: Loop(x, z) object
            Flux surface Loop
        """
        if psi is None:
            psi = self.psi()
        f = find_flux_surf(
            self.x, self.z, psi, psi_n, o_points=o_points, x_points=x_points
        )
        return Loop(x=f[0], z=f[1])

    def get_flux_surface_through_point(self, x, z):
        """
        Get a flux surface loop passing through specified x, z coordinates.

        Parameters
        ----------
        x, z: float

        Returns
        -------
        flux surface: Loop
        """
        psi = self.psi(x, z)
        psi_n = get_psi_norm(psi, *self.get_OX_psis())
        return self.get_flux_surface(psi_n)

    def get_LCFS(self, psi=None):
        """
        Get the Last Closed FLux Surface (LCFS).

        Parameters
        ----------
        psi: Union[np.array(n, m), None]
            The psi field on which to compute the LCFS. Will re-calculate if
            set to None

        Returns
        -------
        lcfs: Loop
            The Loop of the LCFS
        """
        if psi is None:
            psi = self.psi()
        o_points, x_points = self.get_OX_points(psi=psi)
        return find_LCFS_separatrix(self.x, self.z, psi, o_points, x_points)[0]

    def get_B_boundary(self, n=21):  # noqa (N802)
        """
        WIP

        Get B_boundary

        Parameters
        ----------
        n: float

        Returns
        -------
        xb, zb:
        bdir:

        """
        from BLUEPRINT.utilities.tools import lengthnorm
        from scipy.interpolate import interp1d

        # TODO: Work in progress..
        x, z = self.get_LCFS().d2
        dx = -self.psi_func.ev(x, z, dx=1, dy=0)
        dz = -self.psi_func.ev(x, z, dx=0, dy=1)
        length_norm = lengthnorm(x, z)
        lc = np.linspace(0, 1, n + 1)[:-1]
        xb, zb = interp1d(length_norm, x)(lc), interp1d(length_norm, z)(lc)
        bdir = np.array([interp1d(length_norm, dx)(lc), interp1d(length_norm, dz)(lc)]).T
        return xb, zb, bdir

    def get_separatrix(self, psi=None):
        """
        Get the plasma separatrix(-ices).

        Parameters
        ----------
        psi: Union[np.array(n, m), None]
            The flux array. Will re-calculate if set to None

        Returns
        -------
        separatrix: Union[Loop, MultiLoop]
            The separatrix loop(s) (Loop for SN, MultiLoop for DN)
        """
        if psi is None:
            psi = self.psi()
        o_points, x_points = self.get_OX_points(psi=psi)
        return find_LCFS_separatrix(
            self.x, self.z, psi, o_points, x_points, double_null=self.is_double_null
        )[1]

    def _clear_OX_points(self):  # noqa (N802)
        """
        Speed optimisation for storing OX point searches in a single interation
        of the solve. Large grids can cause OX finding to be expensive..
        """
        self._o_points = None
        self._x_points = None

    def get_OX_points(self, psi=None, force_update=False):  # noqa (N802)
        """
        Returns list of [[O-points], [X-points]]
        """
        if (self._o_points is None and self._x_points is None) or force_update is True:
            if psi is None:
                psi = self.psi()
            self._o_points, self._x_points = find_OX(
                self.x, self.z, psi, limiter=self.limiter
            )
        return self._o_points, self._x_points

    def get_OX_psis(self, psi=None):  # noqa (N802)
        """
        Returns psi at the.base.O-point and X-point
        """
        if psi is None:
            psi = self.psi()
        o_points, x_points = self.get_OX_points(psi)
        return o_points[0][2], x_points[0][2]

    def get_midplane(self, x, z, x_psi):
        """
        Retorna a posição no midplane para um valor de psi

        Parameters
        ----------
        x: float
            Starting x coordinate about which to search for a psi surface
        z: float
            Starting z coordinate about which to search for a psi surface
        x_psi: float
            Flux value

        Returns
        -------
        xMP: float
            x coordinate of the midplane point with flux value Xpsi
        zMP: float
            z coordinate of the midplane point with flux value Xpsi
        """
        # NOTE: Moved from EquilibriumManipulator
        def psi_err(x_opt, *args):
            """
            A função de erro do psi
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

    def calc_dx_sep(self):
        """
        Calculates the magnitude of the minimum separation between the flux
        surfaces of null points in the equilibrium at the outboard midplane.

        Returns
        -------
        dXsep: float
            Separation distance at the outboard midplane between the active
            null and the next closest flux surface with a null [m]
        """
        o_points, x_points = self.get_OX_points()
        x, z = self.get_LCFS().d2
        lfs = np.argmax(x)
        lfp = self.get_midplane(x[lfs], z[lfs], x_points[0].psi)
        d_x = []
        count = 0  # Necessary because of retrieval of eqdsks with limiters
        for xp in x_points:
            if "Xpoint" in xp.__class__.__name__:
                if count > 0:
                    psinorm = get_psi_norm(xp.psi, o_points[0].psi, x_points[0].psi)
                    if psinorm > 1:
                        d_x.append(self.get_midplane(*lfp, xp.psi)[0])
                count += 1
        return np.min(d_x) - lfp[0]

    def calc_shaf_shift(self, R_0=None, Z_0=None):
        """
        Calculates the Shafranov shift vector (from the geometric centre to the
        magnetic axis)

        Returns
        -------
        shaf: [float, float]
            The Shafranov shift vector [dx, dz] [m]
        """
        opoint = self.get_OX_points()[0][0]  # magnetic axis
        if R_0 is None and Z_0 is None:
            lcfs = self.get_LCFS()
            au = np.argmax(lcfs["z"])
            al = np.argmin(lcfs["z"])
            ai = np.argmin(lcfs["x"])
            ao = np.argmax(lcfs["x"])
            u, l, i, o = lcfs.d2.T[au], lcfs.d2.T[al], lcfs.d2.T[ai], lcfs.d2.T[ao]
            R_0 = i[0] + (o[0] - i[0]) / 2
            Z_0 = l[1] + (u[1] - l[1]) / 2
        return [opoint.x - R_0, opoint.z - Z_0]

    def calc_li(self):
        """
        Calculates the normalised internal inductance of the plasma

        Returns
        -------
        li: float
            Nnormalised internal inductance of the plasma
        """
        return calc_li(self)

    def calc_q0(self):
        """
        Calculates the MHD safety factor on the plasma axis
        """
        opoint = self.get_OX_points()[0][0]
        psi_xx = self.psi_func(opoint.x, opoint.z, dx=2, grid=False)
        psi_zz = self.psi_func(opoint.x, opoint.z, dy=2, grid=False)
        b_0 = self.Bt(opoint.x)
        jfunc = RectBivariateSpline(self.x[:, 0], self.z[0, :], self._jtor)
        j_0 = jfunc(opoint.x, opoint.z, grid=False)
        return calc_q0(opoint.x, b_0, j_0, psi_xx, psi_zz)

    def _analyse_flux_surf(self, psi_n, psi=None):
        """
        Analyses shape parameters of plasma core at specified normalised flux

        Parameters
        ----------
        psi_n - Normalised psi value of flux surface 0<1

        Returns
        -------
        R_0: float
            Major radius [m]
        A: float
            Aspect ratio
        a: float
            Minor radius [m]
        V: float
            Volume [m^3]
        kappa_u: float
            Upper elongation
        kappa_l: float
            Lower elongation
        delta_u: float
            Upper triangularity
        delta_l: float
            Lower triangularity
        """
        if psi_n == 0:  # Handle linspace endpoint resolution here
            psi_n += PSI_NORM_TOL
        elif psi_n == 1:
            psi_n -= PSI_NORM_TOL
        elif psi_n < 0 or psi_n > 1:
            raise ValueError(f"Specify a psi_n value between 0 and 1: {psi_n}" "is not.")
        if psi is None:
            psi = self.psi()
        lcfs = self.get_flux_surface(psi_n, psi=psi)
        op, _ = self.get_OX_points(psi=psi)
        au = np.argmax(lcfs["z"])
        al = np.argmin(lcfs["z"])
        ai = np.argmin(lcfs["x"])
        ao = np.argmax(lcfs["x"])
        u, l, i, o = lcfs.d2.T[au], lcfs.d2.T[al], lcfs.d2.T[ai], lcfs.d2.T[ao]
        R_0 = i[0] + (o[0] - i[0]) / 2
        volume = lcfs.area * 2 * np.pi * R_0
        c = op[0]
        plasma_a = (o[0] - i[0]) / 2
        A = R_0 / plasma_a
        kappa_u = (u[1] - c[1]) / plasma_a
        kappa_l = abs(l[1] - c[1]) / plasma_a
        delta_u = (i[0] + plasma_a - u[0]) / plasma_a
        delta_l = (i[0] + plasma_a - l[0]) / plasma_a
        return R_0, A, plasma_a, volume, kappa_u, kappa_l, delta_u, delta_l

    def analyse_flux_surface(self, psi_n, psi=None):
        """
        Dictionary output version of a single flux surface analysis
        """
        (
            R_0,
            A,
            a,
            volume,
            kappa_u,
            kappa_l,
            delta_u,
            delta_l,
        ) = self._analyse_flux_surf(psi_n, psi)
        q = self.q(psi_n)
        return {
            "R_0": R_0,
            "A": A,
            "a": a,
            "V": volume,
            "q": q,
            "kappa_u": kappa_u,
            "kappa_l": kappa_l,
            "delta_u": delta_u,
            "delta_l": delta_l,
        }

    def analyse_separatrix(self, psi=None):
        """
        Analyses shape parameters of plasma separatrix
        """
        return self.analyse_flux_surface(1, psi=psi)

    def analyse_core(self, psi=None, n=50, plot=True, ax=None):
        """
        Analyses the shape and characteristics of the plasma core

        Parameters
        ----------
        psi: 2-D np.array (optional)
            Magnetic flux map
        n: int (optional)
            Number of points in normalised psi space to analyse

        Returns
        -------
        dict: dict
            R_0: major radius of flux surface
            A: aspect ratio of flux surface
            a: minor radius of flux surface
            V: volume inside flux surface
            kappa_u: upper elongation of flux surface
            kappa_l: lower elongation of flux surface
            delta_u: upper triangularity of flux surface
            delta_l: lower triangularity of flux surface
            psi_n: normalised psi of flux surface
        """
        if psi is None:
            psi = self.psi()
        p = np.linspace(0, 1, n)
        R_0 = np.zeros(n)
        A = np.zeros(n)
        a = np.zeros(n)
        volume = np.zeros(n)
        kappa_u = np.zeros(n)
        kappa_l = np.zeros(n)
        delta_u = np.zeros(n)
        delta_l = np.zeros(n)
        for i, v in enumerate(p):
            (
                R_0[i],
                A[i],
                a[i],
                volume[i],
                kappa_u[i],
                kappa_l[i],
                delta_u[i],
                delta_l[i],
            ) = self._analyse_flux_surf(v, psi)
        q = self.q(p)
        dictionary = {
            "R_0": R_0,
            "A": A,
            "a": a,
            "V": volume,
            "kappa_u": kappa_u,
            "kappa_l": kappa_l,
            "delta_u": delta_u,
            "delta_l": delta_l,
            "q": q,
            "psi_n": p,
        }
        if plot:
            CorePlotter(dictionary, ax)
        return dictionary

    def analyse_plasma(self):
        """
        Analyses the energetic and magnetic characteristics of the plasma
        """
        d = calc_summary(self)
        f95 = self.analyse_flux_surface(0.95)
        f100 = self.analyse_flux_surface(1)
        d["q_95"] = f95["q"]
        d["kappa_95"] = f95["kappa_u"]
        d["delta_95"] = f95["delta_u"]
        d["kappa"] = f100["kappa_u"]
        d["delta"] = f100["delta_u"]
        d["R_0"] = f100["R_0"]
        d["A"] = f100["A"]
        d["a"] = f100["a"]
        # d['dXsep'] = self.calc_dXsep()
        d["Ip"] = self._Ip
        d["shaf_shift"] = self.calc_shaf_shift(R_0=f100["R_0"], Z_0=0)
        return d

    def analyse_coils(self):
        """
        Analyses and summarises the electro-magneto-mechanical characteristics
        of the equilbrium and coilset
        """
        c_names = self.coilset.get_control_names()
        currents = self.coilset.get_control_currents() / 1e6
        fields = self.get_fields()
        forces = self.get_forces() / 1e6
        fz = forces.T[1]
        fz_cs = fz[self.coilset.n_PF :]
        fz_c_stot = sum(fz_cs)
        fsep = []
        for j in range(self.coilset.n_CS - 1):
            fsep.append(np.sum(fz_cs[j + 1 :]) - np.sum(fz_cs[: j + 1]))
        fsep = max(fsep)
        table = {"I [MA]": currents, "B [T]": fields, "F [MN]": fz}
        df = DataFrame(list(table.values()), index=list(table.keys()))
        df = df.applymap(lambda x: f"{x:.2f}")
        print(tabulate.tabulate(df, headers=c_names))
        return table, fz_c_stot, fsep

    @property
    def is_double_null(self):
        """
        Whether or not the Equilibrium is a double-null Equilibrium.

        Returns
        -------
        double_null: bool
            Whether or not the Equilibrium is a double-null Equilibrium.
        """
        _, x_points = self.get_OX_points()

        psi_1 = x_points[0].psi
        psi_2 = x_points[1].psi
        return abs(psi_1 - psi_2) < PSI_NORM_TOL

    def plot(self, ax=None, plasma=False, update_ox=False, show_ox=True):
        """
        Plots the equilibrium magnetic flux surfaces object onto `ax`
        """
        return EquilibriumPlotter(
            self, ax, plasma=plasma, update_ox=update_ox, show_ox=show_ox
        )

    def plot_field(self, ax=None, update_ox=False, show_ox=True):
        """
        Plots the equilibrium field structure onto `ax`
        """
        return EquilibriumPlotter(
            self,
            ax,
            plasma=False,
            update_ox=update_ox,
            show_ox=show_ox,
            field=True,
        )

    def plot_core(self, ax=None):
        """
        Plots a 1-D section through the magnetic axis
        """
        return CorePlotter2(self, ax)


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
