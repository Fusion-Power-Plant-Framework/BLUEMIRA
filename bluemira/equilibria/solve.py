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
Picard iteration procedures for equilibria (and their infinite variations)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterator, Optional

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.file import try_get_bluemira_path
from bluemira.base.look_and_feel import (
    bluemira_print,
    bluemira_print_flush,
    bluemira_warn,
)
from bluemira.equilibria.constants import DPI_GIF, PLT_PAUSE, PSI_REL_TOL
from bluemira.utilities.error import ExternalOptError
from bluemira.utilities.plot_tools import make_gif, save_figure

if TYPE_CHECKING:
    from bluemira.equilibria.equilibrium import Equilibrium
    from bluemira.equilibria.optimisation.problem import CoilsetOptimisationProblem
    from bluemira.equilibria.optimisation.problem.base import CoilsetOptimiserResult

__all__ = [
    "DudsonConvergence",
    "CunninghamConvergence",
    "JsourceConvergence",
    "JeonConvergence",
    "JrelConvergence",
    "PicardIterator",
]


class ConvergenceCriterion(ABC):
    """
    Convergence criterion base class

    Parameters
    ----------
    limit:
        The limit at which the convergence criterion is met.
    """

    flag_psi = True

    def __init__(self, limit: float):
        self.limit = limit
        self.progress = []
        self.math_string = NotImplemented

    @abstractmethod
    def __call__(
        self, old_val: np.ndarray, new_val: np.ndarray, i: int, print_status: bool = True
    ) -> bool:
        """
        Carry out convergence check.

        Parameters
        ----------
        old_val:
            The value from the previous iteration.
        new_val:
            The value from the current iteration.
        i:
            The index of the iteration.
        print_status:
            If True then prints the status of the convergence, by default True.

        Returns
        -------
        True if the convergence criterion is met, else False.
        """

    def check_converged(self, value: float) -> bool:
        """
        Check for convergence.

        Parameters
        ----------
        value:
            The value of the convergence criterion

        Returns
        -------
        Whether or not convergence has been reached
        """
        self.progress.append(value)
        if value <= self.limit:
            return True
        return False

    def plot(self, ax: Optional[plt.Axes] = None):
        """
        Plot the convergence behaviour.

        Parameters
        ----------
        ax:
            The matplotlib axes onto which to plot
        """
        if ax is None:
            f, ax = plt.subplots()
        ax.semilogy(self.progress)
        ax.set_xlabel("Iterations [n]")
        ax.set_ylabel(self.math_string)


class DudsonConvergence(ConvergenceCriterion):
    """
    FreeGS convergence criterion

    Parameters
    ----------
    limit:
        The limit at which the convergence criterion is met.
    """

    def __init__(self, limit: float = PSI_REL_TOL):
        super().__init__(limit)
        self.math_string = "$\\dfrac{max|\\Delta\\psi|}{max(\\psi)-min(\\psi)}$"

    def __call__(
        self, psi_old: np.ndarray, psi: np.ndarray, i: int, print_status: bool = True
    ) -> bool:
        """
        Carry out convergence check.

        Parameters
        ----------
        psi_old:
            The value from the previous iteration.
        psi:
            The value from the current iteration.
        i:
            The index of the iteration.
        print_status:
            If True then prints the status of the convergence, by default True.

        Returns
        -------
        True if the convergence criterion is met, else False.
        """
        dpsi = psi_old - psi
        dpsi_max = np.amax(abs(dpsi))
        dpsi_rel = dpsi_max / (np.amax(psi) - np.amin(psi))
        if print_status:
            bluemira_print_flush(
                f"EQUILIBRIA G-S iter {i}: relative delta_psi: {100*dpsi_rel:.2f} %"
            )
        return self.check_converged(dpsi_rel)


class JrelConvergence(ConvergenceCriterion):
    """
    FreeGS convergence criterion

    Parameters
    ----------
    limit:
        The limit at which the convergence criterion is met.
    """

    flag_psi = False

    def __init__(self, limit: float = 1e-2):
        super().__init__(limit)
        self.math_string = "$\\dfrac{max|\\Delta J|}{max(J)-min(J)}$"

    def __call__(
        self, j_old: np.ndarray, j_new: np.ndarray, i: int, print_status: bool = True
    ) -> bool:
        """
        Carry out convergence check.

        Parameters
        ----------
        j_old:
            The value from the previous iteration.
        j_new:
            The value from the current iteration.
        i:
            The index of the iteration.
        print_status:
            If True then prints the status of the convergence, by default True.

        Returns
        -------
        True if the convergence criterion is met, else False.
        """
        d_j = j_old - j_new
        d_j_max = np.amax(abs(d_j))
        d_j_rel = d_j_max / (np.amax(j_new) - np.amin(j_new))
        if print_status:
            bluemira_print_flush(
                f"EQUILIBRIA G-S iter {i}: relative delta_J: {100*d_j_rel:.2f} %"
            )
        return self.check_converged(d_j_rel)


class LacknerConvergence(ConvergenceCriterion):
    """
    Karl Lackner's convergence criterion (< 10E-4)
    (Lackner, Computation of ideal MHD equilibria, 1976)

    Parameters
    ----------
    limit:
        The limit at which the convergence criterion is met.
    """

    def __init__(self, limit: float = 10e-4):
        super().__init__(limit)
        self.math_string = "$max\\dfrac{|\\Delta\\psi|}{\\psi}$"

    def __call__(
        self, psi_old: np.ndarray, psi: np.ndarray, i: int, print_status: bool = True
    ) -> bool:
        """
        Carry out convergence check.

        Parameters
        ----------
        psi_old:
            The value from the previous iteration.
        psi:
            The value from the current iteration.
        i:
            The index of the iteration.
        print_status:
            If True then prints the status of the convergence, by default True.

        Returns
        -------
        True if the convergence criterion is met, else False.
        """
        conv = np.amax(np.abs((psi - psi_old) / psi))
        if print_status:
            bluemira_print_flush(f"EQUILIBRIA G-S iter {i}: psi convergence: {conv:e}")
        return self.check_converged(conv)


class JeonConvergence(ConvergenceCriterion):
    """
    TES convergence criterion

    Parameters
    ----------
    limit:
        The limit at which the convergence criterion is met.
    """

    def __init__(self, limit: float = 1e-4):
        super().__init__(limit)
        self.math_string = "$||\\Delta\\psi||$"

    def __call__(
        self, psi_old: np.ndarray, psi: np.ndarray, i: int, print_status: bool = True
    ) -> bool:
        """
        Carry out convergence check.

        Parameters
        ----------
        psi_old:
            The value from the previous iteration.
        psi:
            The value from the current iteration.
        i:
            The index of the iteration.
        print_status:
            If True then prints the status of the convergence, by default True.

        Returns
        -------
        True if the convergence criterion is met, else False.
        """
        conv = np.linalg.norm(psi_old - psi)
        if print_status:
            bluemira_print_flush(
                f"EQUILIBRIA G-S iter {i}: psi norm convergence: {conv:e}"
            )
        return self.check_converged(conv)


class CunninghamConvergence(ConvergenceCriterion):
    """
    FIESTA convergence criterion

    Parameters
    ----------
    limit:
        The limit at which the convergence criterion is met.
    """

    flag_psi = False

    def __init__(self, limit: float = 1e-7):
        super().__init__(limit)
        self.math_string = "$\\dfrac{\\sum{\\Delta J_{n}^{2}}}{\\sum{J_{n+1}^{2}}}$"

    def __call__(
        self, j_old: np.ndarray, j_new: np.ndarray, i: int, print_status: bool = True
    ) -> bool:
        """
        Carry out convergence check.

        Parameters
        ----------
        j_old:
            The value from the previous iteration.
        j_new:
            The value from the current iteration.
        i:
            The index of the iteration.
        print_status:
            If True then prints the status of the convergence, by default True.

        Returns
        -------
        True if the convergence criterion is met, else False.
        """
        d_j = j_old - j_new
        conv = np.sum(d_j**2) / np.sum(j_new**2)
        if print_status:
            bluemira_print_flush(
                f"EQUILIBRIA G-S iter {i}: J_phi source convergence: {conv:e}"
            )
        self._conv = conv
        return self.check_converged(conv)


class JsourceConvergence(ConvergenceCriterion):
    """
    Plasma current source convergence criterion.

    Parameters
    ----------
    limit:
        The limit at which the convergence criterion is met.
    """

    flag_psi = False

    def __init__(self, limit: float = 1e-4):
        super().__init__(limit)
        self.math_string = "$||\\Delta J||$"

    def __call__(
        self, j_old: np.ndarray, j_new: np.ndarray, i: int, print_status: bool = True
    ) -> bool:
        """
        Carry out convergence check.

        Parameters
        ----------
        j_old:
            The value from the previous iteration.
        j_new:
            The value from the current iteration.
        i:
            The index of the iteration.
        print_status:
            If True then prints the status of the convergence, by default True.

        Returns
        -------
        True if the convergence criterion is met, else False.
        """
        conv = np.linalg.norm(j_old - j_new)
        if print_status:
            # Format convergence
            bluemira_print_flush(
                f"EQUILIBRIA G-S iter {i}: ||J_phi_old-J_phi|| convergence: {conv:e}"
            )
        return self.check_converged(conv)


class PicardIterator:
    """A Picard iterative solver.

    Child classes must provide a __call__ method which carries out the
    iteration process(es)

    Parameters
    ----------
    eq:
        The equilibrium to solve for
    optimisation_problem:
        The optimisation problem to use when iterating
    convergence:
        The convergence criterion to use (defaults to Dudson)
    fixed_coils:
        Whether or not the coil positions are fixed
    relaxation:
        The relaxation parameter to use between iterations
    maxiter:
        The maximum number of iterations
    plot:
        Whether or not to plot
    gif:
        Whether or not to make a GIF
    figure_folder:
        The path where figures will be saved. If the input value is None (e.g. default)
        then this will be reinterpreted as the path data/plots/equilibria under the
        bluemira root folder, if that path is available.
    plot_name:
        GIF plot file base-name
    """

    def __init__(
        self,
        eq: Equilibrium,
        optimisation_problem: CoilsetOptimisationProblem,
        convergence: Optional[ConvergenceCriterion] = None,
        fixed_coils: bool = False,
        relaxation: float = 0,
        maxiter: int = 30,
        plot: bool = True,
        gif: bool = False,
        figure_folder: Optional[str] = None,
        plot_name: str = "default_0",
    ):
        self.eq = eq
        self.coilset = self.eq.coilset
        self.opt_prob = optimisation_problem
        if isinstance(convergence, ConvergenceCriterion):
            self.convergence = convergence
        elif convergence is None:
            self.convergence = DudsonConvergence()
        else:
            raise ValueError(
                "Optimiser convergence specification must be a sub-class of"
                " ConvergenceCriterion."
            )
        self.fixed_coils = fixed_coils

        self.relaxation = relaxation
        self.maxiter = maxiter
        self.plot_flag = plot or (gif and not plot)
        self.gif_flag = gif
        if figure_folder is None:
            figure_folder = try_get_bluemira_path(
                "", subfolder="generated_data", allow_missing=not self.gif_flag
            )
        self.figure_folder = figure_folder
        self.store = []
        self.i = 0
        if self.plot_flag:
            self.pname = plot_name
            self.f, self.ax = plt.subplots()

    def _optimise_coilset(self):
        try:
            result = self.opt_prob.optimise(fixed_coils=self.fixed_coils)
            try:
                coilset = result.coilset
            except AttributeError:
                coilset = result
            self.store.append(coilset)
        except ExternalOptError:
            coilset = self.store[-1]
        self.result = result
        self.coilset = coilset

    @property
    def psi(self) -> np.ndarray:
        """
        Get the magnetic flux array.
        """
        return self._psi

    @property
    def j_tor(self) -> np.ndarray:
        """
        Get the toroidal current density array.
        """
        return self._j_tor

    def __call__(self) -> CoilsetOptimiserResult:
        """
        The iteration object call handle.
        """
        iterator = iter(self)
        while self.i < self.maxiter:
            try:
                next(iterator)
            except StopIteration:  # noqa: PERF203
                print()
                bluemira_print("EQUILIBRIA G-S converged value found.")
                break
        else:
            print()
            bluemira_warn(
                "EQUILIBRIA G-S unable to find converged value after"
                f" {self.i} iterations."
            )
        self._teardown()
        return self.result

    def __iter__(self) -> Iterator:
        """
        Make the class a Python iterator.

        Returns
        -------
        The current instance as an iterator.
        """
        self._setup()
        return self

    def __next__(self):
        """
        Perform an interation of the solver.
        """
        if not hasattr(self, "_psi"):
            self._setup()

        if self.i > 0 and self.check_converged(print_status=False):
            raise StopIteration

        self._psi_old = self.psi.copy()
        self._j_tor_old = self.j_tor.copy()
        self._solve()
        self._psi = self.eq.psi()
        self._j_tor = self.eq._jtor
        check = self.check_converged()
        if self.plot_flag:
            self.update_fig()
        if check:
            if self.gif_flag:
                make_gif(self.figure_folder, self.pname)
            raise StopIteration
        self._optimise_coilset()
        self._psi = (
            1 - self.relaxation
        ) * self.eq.psi() + self.relaxation * self._psi_old
        self.i += 1

    def iterate_once(self) -> CoilsetOptimiserResult:
        """
        Perform a single iteration and handle convergence.
        """
        try:
            next(self)
        except StopIteration:
            bluemira_print("EQUILIBRIA G-S converged value found, nothing to do.")
            self._teardown()
        return self.result

    def check_converged(self, print_status: bool = True) -> bool:
        """
        Check if the iterator has converged.

        Parameters
        ----------
        print_status:
            If True then prints the status of the convergence, by default True.

        Returns
        -------
        True if the convergence criterion is met, else False.
        """
        if self.convergence.flag_psi:
            return self.convergence(
                self._psi_old, self.psi, self.i, print_status=print_status
            )

        return self.convergence(
            self._j_tor_old, self.j_tor, self.i, print_status=print_status
        )

    def update_fig(self):
        """
        Updates the figure if plotting is used
        """
        self.ax.clear()
        self.eq.plot(ax=self.ax)
        plt.pause(PLT_PAUSE)
        save_figure(
            self.f,
            self.pname + str(self.i),
            save=self.gif_flag,
            folder=self.figure_folder,
            dpi=DPI_GIF,
        )

    def _solve(self):
        """
        Solve for this iteration.
        """
        if self.eq._li_flag and self.i > self.eq.profiles._l_i_min_iter:
            self.eq.solve_li(psi=self.psi)
        else:
            self.eq.solve(psi=self.psi)

    def _initial_optimise_coilset(self, **kwargs):
        self._optimise_coilset(**kwargs)

    def _setup(self):
        """
        Initialise psi and toroidal current values.
        """
        self._initial_optimise_coilset()
        self._psi = self.eq.psi()
        self._j_tor = self.eq._jtor
        if self._j_tor is None:
            self._j_tor = np.zeros((self.eq.grid.nx, self.eq.grid.nz))

    def _teardown(self):
        """Final clean-up for consistency between psi and jtor.

        In the case of converged equilibria, slight (artificial) improvement
        in consistency. In the case of unconverged equilibria, gives a more
        reasonable understanding of the final state.
        """
        o_points, x_points = self.eq.get_OX_points(force_update=True)
        self.eq._jtor = self.eq.profiles.jtor(
            self.eq.x, self.eq.z, self.eq.psi(), o_points, x_points
        )
