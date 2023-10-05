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
Bluemira module for the solution of a 2D magnetostatic problem with cylindrical symmetry
and toroidal current source using fenics FEM solver
"""
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Union

import dolfin
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from bluemira.base.constants import MU_0
from bluemira.base.file import try_get_bluemira_path
from bluemira.base.look_and_feel import bluemira_print_flush
from bluemira.display import plot_defaults
from bluemira.equilibria.constants import DPI_GIF, PLT_PAUSE
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.fem_fixed_boundary.utilities import (
    _interpolate_profile,
    find_magnetic_axis,
)
from bluemira.equilibria.plotting import PLOT_DEFAULTS
from bluemira.magnetostatics.finite_element_2d import FemMagnetostatic2d
from bluemira.utilities.plot_tools import make_gif, save_figure


def _parse_to_callable(profile_data: Union[None, np.ndarray]):
    if isinstance(profile_data, np.ndarray):
        x = np.linspace(0, 1, len(profile_data))
        return _interpolate_profile(x, profile_data)
    if profile_data is None:
        return None
    return None


@dataclass
class FixedBoundaryEquilibrium:
    """
    Simple minimal dataclass for a fixed boundary equilibrium.
    """

    # Solver information
    mesh: dolfin.Mesh
    psi: Callable[[float, float], float]

    # Profile information
    p_prime: np.ndarray
    ff_prime: np.ndarray
    R_0: float
    B_0: float
    I_p: float


class FemGradShafranovFixedBoundary(FemMagnetostatic2d):
    """
    A 2D fem Grad Shafranov solver. The solver is thought as support for the fem fixed
    boundary module.

    Parameters
    ----------
    p_prime:
        p' flux function. If callable, then used directly (50 points saved in file).
        If None, these must be specified later on, but before the solve.
    ff_prime:
        FF' flux function. If callable, then used directly (50 points saved in file).
        If None, these must be specified later on, but before the solve.
    mesh:
        Mesh to use when solving the problem.
        If None, must be specified later on, but before the solve.
    I_p:
        Plasma current [A]. If None, the plasma current is calculated, otherwise
        the source term is scaled to match the plasma current.
    B_0:
        Toroidal field at R_0 [T]. Used when saving to file.
    R_0:
        Major radius [m]. Used when saving to file.
    p_order:
        Order of the approximating polynomial basis functions
    max_iter:
        Maximum number of iterations
    iter_err_max:
        Convergence criterion value
    relaxation:
        Relaxation factor for the Picard iteration procedure
    """

    def __init__(
        self,
        p_prime: Optional[Callable[[float], float]] = None,
        ff_prime: Optional[Callable[[float], float]] = None,
        mesh: Optional[Union[dolfin.Mesh, str]] = None,
        I_p: Optional[float] = None,
        R_0: Optional[float] = None,
        B_0: Optional[float] = None,
        p_order: int = 2,
        max_iter: int = 10,
        iter_err_max: float = 1e-5,
        relaxation: float = 0.0,
    ):
        super().__init__(p_order)
        self._g_func = None
        self._psi_ax = None
        self._psi_b = None
        self._grad_psi = None
        self._pprime = None
        self._ffprime = None

        self._curr_target = I_p
        self._R_0 = R_0
        self._B_0 = B_0

        if (p_prime is not None) and (ff_prime is not None):
            self.set_profiles(p_prime, ff_prime)

        if mesh is not None:
            self.set_mesh(mesh)

        self.iter_err_max = iter_err_max
        self.max_iter = max_iter
        self.relaxation = relaxation
        self.k = 1

    @property
    def psi_ax(self) -> float:
        """Poloidal flux on the magnetic axis"""
        if self._psi_ax is None:
            self._psi_ax = self.psi(find_magnetic_axis(self.psi, self.mesh))
        return self._psi_ax

    @property
    def psi_b(self) -> float:
        """Poloidal flux on the boundary"""
        if self._psi_b is None:
            self._psi_b = 0.0
        return self._psi_b

    def grad_psi(self, point: np.ndarray) -> np.ndarray:
        """
        Calculate the gradients of psi at a point
        """
        if self._grad_psi is None:
            w = dolfin.VectorFunctionSpace(self.mesh, "CG", 1)
            dpsi_dx = self.psi.dx(0)
            dpsi_dz = self.psi.dx(1)
            self._grad_psi = dolfin.project(dolfin.as_vector((dpsi_dx, dpsi_dz)), w)
            self._grad_psi.set_allow_extrapolation(True)
        return self._grad_psi(point)

    @property
    def psi_norm_2d(self) -> Callable[[np.ndarray], np.ndarray]:
        """Normalized flux function in 2-D"""
        return lambda x: np.sqrt(
            np.abs((self.psi(x) - self.psi_ax) / (self.psi_b - self.psi_ax))
        )

    def set_mesh(self, mesh: Union[dolfin.Mesh, str]):
        """
        Set the mesh for the solver

        Parameters
        ----------
        mesh:
            Filename of the xml file with the mesh definition or a dolfin mesh
        """
        super().set_mesh(mesh=mesh)
        self._reset_psi_cache()

    def _create_g_func(
        self,
        pprime: Union[Callable[[np.ndarray], np.ndarray], float],
        ffprime: Union[Callable[[np.ndarray], np.ndarray], float],
        curr_target: Optional[float] = None,
    ) -> Callable[[np.ndarray], float]:
        """
        Return the density current function given pprime and ffprime.

        Parameters
        ----------
        pprime:
            pprime as function of psi_norm (1-D function)
        ffprime:
            ffprime as function of psi_norm (1-D function)
        curr_target:
            Target current (also used to initialize the solution in case self.psi is
            still 0 and pprime and ffprime are, then, not defined) [A]

        Returns
        -------
        Source current callable to solve the magnetostatic problem
        """
        area = dolfin.assemble(
            dolfin.Constant(1) * dolfin.Measure("dx", domain=self.mesh)()
        )

        j_target = curr_target / area if curr_target else 1.0

        def g(x):
            if self.psi_ax == 0:
                return j_target
            r = x[0]
            x_psi = self.psi_norm_2d(x)

            a = r * pprime(x_psi)
            b = 1 / MU_0 / r * ffprime(x_psi)

            return self.k * 2 * np.pi * (a + b)

        return g

    def define_g(self):
        """
        Return the density current DOLFIN function given pprime and ffprime.
        """
        self._g_func = self._create_g_func(
            self._pprime, self._ffprime, self._curr_target
        )

        # # This instruction seems to slow the calculation
        # super().define_g(ScalarSubFunc(self._g_func))

        # it has been replaced by this code
        dof_points = self.V.tabulate_dof_coordinates()
        self.g.vector()[:] = np.array([self._g_func(p) for p in dof_points])

    def set_profiles(
        self,
        p_prime: Callable[[float], float],
        ff_prime: Callable[[float], float],
        I_p: Optional[float] = None,
        B_0: Optional[float] = None,
        R_0: Optional[float] = None,
    ):
        """
        Set the profies for the FEM G-S solver.

        Parameters
        ----------
        pprime:
            pprime as function of psi_norm (1-D function)
        ffprime:
            ffprime as function of psi_norm (1-D function)
        I_p:
            Target current (also used to initialize the solution in case self.psi is
            still 0 and pprime and ffprime are, then, not defined).
            If None, plasma current is calculated and not constrained
        B_0:
            Toroidal field at R_0 [T]. Used when saving to file.
        R_0:
            Major radius [m]. Used when saving to file.
        """
        # Note: pprime and ffprime have been limited to a Callable,
        # because otherwise it is necessary to provide also psi_norm_1D
        # to which they refer.
        if callable(p_prime):
            self._pprime = p_prime
            self._pprime_data = p_prime(np.linspace(0, 1, 50))
        else:
            raise TypeError("p_prime must be a function")
        if callable(ff_prime):
            self._ffprime = ff_prime
            self._ffprime_data = ff_prime(np.linspace(0, 1, 50))
        else:
            raise TypeError("ff_prime must be a function")
        if I_p is not None:
            self._curr_target = I_p
        if B_0 is not None:
            self._B_0 = B_0
        if R_0 is not None:
            self._R_0 = R_0

    def _calculate_curr_tot(self) -> float:
        """Calculate the total current into the domain"""
        return dolfin.assemble(self.g * dolfin.Measure("dx", domain=self.mesh)())

    def _update_curr(self):
        self.k = 1

        self.define_g()

        if self._curr_target:
            self.k = self._curr_target / self._calculate_curr_tot()

    def _reset_psi_cache(self):
        """
        Reset cached psi-axis and psi-boundary properties.
        """
        self._psi_b = None
        self._psi_ax = None
        self._grad_psi = None

    def _check_all_inputs_ready_error(self):
        if self.mesh is None:
            raise EquilibriaError(
                "You cannot solve this problem yet! Please set the mesh first, using"
                " set_mesh(mesh)."
            )
        if self._pprime is None or self._ffprime is None:
            raise EquilibriaError(
                "You cannot solve this problem yet! Please set the profile functions"
                " first, using set_profiles(p_prime, ff_prime)."
            )

    def solve(
        self,
        plot: bool = False,
        debug: bool = False,
        gif: bool = False,
        figname: Optional[str] = None,
    ) -> FixedBoundaryEquilibrium:
        """
        Solve the G-S problem.

        Parameters
        ----------
        plot:
            Whether or not to plot
        debug:
            Whether or not to display debug information
        gif: bool
            Whether or not to produce a GIF
        figname:
            The name of the figure. If None, a suitable default is used.

        Returns
        -------
        FixedBoundaryEquilibrium object corresponding to the solve
        """
        self._check_all_inputs_ready_error()
        self.define_g()

        points = self.mesh.coordinates()
        plot = any((plot, debug, gif))
        folder = try_get_bluemira_path(
            "", subfolder="generated_data", allow_missing=False
        )
        if figname is None:
            figname = "Fixed boundary equilibrium iteration "

        super().solve()
        self._reset_psi_cache()
        self._update_curr()

        if plot:
            plot_defaults()
            f, ax, cax = self._setup_plot(debug)

        diff = np.zeros(len(points))
        for i in range(1, self.max_iter + 1):
            prev_psi = self.psi.vector()[:]
            prev = np.array([self.psi_norm_2d(p) for p in points])

            if plot:
                self._plot_current_iteration(ax, cax, i, points, prev, diff, debug)
                if debug or gif:
                    save_figure(
                        f,
                        figname + str(i),
                        save=True,
                        folder=folder,
                        dpi=DPI_GIF,
                    )

            super().solve()
            self._reset_psi_cache()

            new = np.array([self.psi_norm_2d(p) for p in points])
            diff = new - prev

            eps = np.linalg.norm(diff, ord=2) / np.linalg.norm(new, ord=2)

            bluemira_print_flush(
                f"iter = {i} eps = {eps:.3E} psi_ax : {self.psi_ax:.2f}"
            )

            # Update psi in-place (Fenics handles this with the below syntax)
            self.psi.vector()[:] = (1 - self.relaxation) * self.psi.vector()[
                :
            ] + self.relaxation * prev_psi

            self._update_curr()

            if eps < self.iter_err_max:
                break

        if plot:
            plt.close(f)
        if gif:
            make_gif(folder, figname, clean=not debug)

        return self._equilibrium()

    def _equilibrium(self):
        """Equilibrium data object"""
        return FixedBoundaryEquilibrium(
            self.mesh,
            self.psi,
            self._pprime_data,
            self._ffprime_data,
            self._R_0,
            self._B_0,
            self._calculate_curr_tot(),
        )

    @staticmethod
    def _setup_plot(debug: bool):
        n_col = 3 if debug else 2
        fig, ax = plt.subplots(1, n_col, figsize=(18, 10))
        plt.subplots_adjust(wspace=0.5)

        cax = []
        for axis in ax:
            divider = make_axes_locatable(axis)
            cax.append(divider.append_axes("right", size="10%", pad=0.1))

        return fig, ax, cax

    def _plot_current_iteration(
        self,
        ax,
        cax,
        i_iter: int,
        points: Iterable,
        prev: np.ndarray,
        diff: np.ndarray,
        debug: bool,
    ):
        for axis in ax:
            axis.clear()
            axis.set_xlabel("x")
            axis.set_ylabel("z")
            axis.set_aspect("equal")

        cm = self._plot_array(
            ax[0],
            points,
            np.array([self._g_func(p) for p in points]),
            f"({i_iter}) " + "$J_{tor}$",
            PLOT_DEFAULTS["current"]["cmap"],
        )
        self._add_colorbar(cm, cax[0], "A/m$^{2}$\n")

        levels = np.linspace(0, 1, 11)
        cm = self._plot_array(
            ax[1],
            points,
            prev,
            f"({i_iter}) " + "$\\Psi_{n}$",
            PLOT_DEFAULTS["psi"]["cmap"],
            levels,
        )
        self._add_colorbar(cm, cax[1], "")

        if debug:
            cm = self._plot_array(
                ax[2],
                points,
                100 * diff,
                f"({i_iter}) " + "$\\Psi_{n}$ error",
                "seismic",
            )
            self._add_colorbar(cm, cax[2], "%")

        plt.pause(PLT_PAUSE)

    @staticmethod
    def _plot_array(
        ax,
        points: np.ndarray,
        array: np.ndarray,
        title: str,
        cmap: str,
        levels: Optional[np.ndarray] = None,
    ):
        cm = ax.tricontourf(points[:, 0], points[:, 1], array, cmap=cmap, levels=levels)
        ax.tricontour(
            points[:, 0], points[:, 1], array, colors="k", linewidths=0.5, levels=levels
        )

        ax.set_title(title)
        return cm

    @staticmethod
    def _add_colorbar(cm, cax, title):
        last_axes = plt.gca()
        ax = cm.axes
        fig = ax.figure
        fig.colorbar(cm, cax=cax)
        cax.set_title(title)
        plt.sca(last_axes)
