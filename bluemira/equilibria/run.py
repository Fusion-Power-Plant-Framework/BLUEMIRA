# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2022 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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
Interface for building and loading equilibria and coilset designs
"""

from copy import deepcopy
from typing import Dict, List, Optional, Type

import numpy as np

from bluemira.base.look_and_feel import bluemira_print, bluemira_warn
from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.eq_constraints import MagneticConstraintSet
from bluemira.equilibria.equilibrium import Breakdown, Equilibrium
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.opt_constraints import L2_norm_constraint
from bluemira.equilibria.opt_problems import (
    BreakdownCOP,
    BreakdownZoneStrategy,
    CoilsetOptimisationProblem,
    MinimalCurrentCOP,
    UnconstrainedCurrentCOP,
)
from bluemira.equilibria.physics import calc_psib
from bluemira.equilibria.profiles import Profile
from bluemira.equilibria.solve import PicardCoilsetIterator
from bluemira.utilities.opt_problems import OptimisationConstraint
from bluemira.utilities.optimiser import Optimiser


class Snapshot:
    """
    Abstract object for grouping of equilibria objects in a given state.

    Parameters
    ----------
    eq: Equilibrium object
        The equilibrium at the snapshot
    coilset: CoilSet
        The coilset at the snapshot
    constraints: Constraints object
        The constraints at the snapshot
    profiles: Profile object
        The profile at the snapshot
    optimiser: EquilibriumOptimiser object
        The optimiser for the snapshot
    limiter: Limiter object
        The limiter for the snapshot
    tfcoil: Loop object
        The PF coil placement boundary
    """

    def __init__(
        self,
        eq,
        coilset,
        constraints,
        profiles,
        optimiser=None,
        limiter=None,
        tfcoil=None,
    ):
        self.eq = deepcopy(eq)
        self.coilset = deepcopy(coilset)
        if constraints is not None:
            self.constraints = constraints
        else:
            self.constraints = None
        if profiles is not None:
            self.profiles = deepcopy(profiles)
        else:
            self.profiles = None
        if limiter is not None:
            self.limiter = deepcopy(limiter)
        else:
            self.limiter = None
        if optimiser is not None:
            self.optimiser = optimiser
        else:
            self.optimiser = None
        self.tf = tfcoil


class PulsedCoilsetProblem:
    """
    Abstract base class for the procedural design of a pulsed tokamak poloidal field
    coilset.
    """

    BREAKDOWN = "Breakdown"
    EQ_REF = "Reference"
    SOF = "SOF"
    EOF = "EOF"

    def __init__(self):
        self.snapshots = {}

    def take_snapshot(self, name, eq, coilset, optimiser, problem, profiles=None):
        """
        Take a snapshot of the pulse.
        """
        if name in self.snapshots:
            bluemira_warn(f"Over-writing snapshot {name}!")

        self.snapshots[name] = Snapshot(eq, coilset, problem, profiles, optimiser)

    def run_reference_equilibrium(self):
        """
        Run a reference equilibrium.
        """
        coilset = deepcopy(self.coilset)
        rb0 = [self.params.R_0.value, self.params.B_0.value]
        eq = Equilibrium(
            coilset,
            self.grid,
            Ip=self.params.I_p.value * 1e6,
            RB0=rb0,
            profiles=self.profiles,
        )
        optimiser = UnconstrainedCurrentCOP(
            coilset, eq, self.eq_targets, gamma=self._eq_settings["gamma"]
        )
        program = PicardCoilsetIterator(
            eq,
            self.profiles,
            self.eq_targets,
            optimiser,
            relaxation=0.1,
            plot=False,
        )
        program()
        self.take_snapshot(self.EQ_REF, eq, coilset, optimiser, self.eq_targets)

    def calculate_sof_eof_fluxes(self, psi_premag: Optional[float] = None):
        """
        Calculate the SOF and EOF plasma boundary fluxes.
        """
        if psi_premag is None:
            if self.BREAKDOWN not in self.snapshots:
                self.run_premagnetisation()
            psi_premag = self.snapshots[self.BREAKDOWN].eq.breakdown_psi
        psi_sof = calc_psib(
            2 * np.pi * psi_premag,
            self.params.R_0.value,
            1e6 * self.params.I_p.value,
            self.params.l_i.value,
            self.params.C_Ejima.value,
        )
        psi_eof = psi_sof - self.params.tau_flattop.value * self.params.v_burn.value
        return psi_sof, psi_eof

    def converge_equilibrium(self, eq, problem):
        """
        Converge an equilibrium problem from a 'frozen' plasma optimised state.
        """
        # TODO: Converge equilibria
        pass

    def plot(self):
        """
        Plot the pulsed equilibrium problem.
        """
        n_snapshots = len(self.snapshots)
        if n_snapshots == 0:
            return

        f, ax = plt.subplots(1, n_snapshots)
        for i, (k, snap) in enumerate(self.snapshots.items()):
            axi = ax[i]
            snap.eq.plot(ax=axi)
            snap.coilset.plot(ax=axi)
            axi.set_title(k)
        return f


class FixedPulsedCoilsetProblem(PulsedCoilsetProblem):
    """
    Procedural design for a pulsed tokamak with a known, fixed PF coilset.
    """

    def __init__(
        self,
        params,
        coilset: CoilSet,
        grid: Grid,
        coil_constraints: List[OptimisationConstraint],
        profiles: Profile,
        magnetic_targets: MagneticConstraintSet,
        breakdown_strategy_cls: Type[BreakdownZoneStrategy],
        breakdown_problem_cls: Type[BreakdownCOP],
        breakdown_optimiser: Optimiser = Optimiser(
            "COBYLA", opt_conditions={"max_eval": 5000, "ftol_rel": 1e-10}
        ),
        breakdown_settings: Dict = {"B_stray_con_tol": 1e-8, "n_B_stray_points": 20},
        equilibrium_problem_cls: Type[CoilsetOptimisationProblem] = MinimalCurrentCOP,
        equilibrium_optimiser: Optimiser = Optimiser(
            "SLSQP", opt_conditions={"max_eval": 1000, "ftol_rel": 1e-6}
        ),
        equilibrium_settings: Optional[Dict] = None,
    ):
        self.params = params
        self.coilset = coilset
        self.grid = grid
        self.profiles = profiles
        self.eq_targets = magnetic_targets

        self._bd_strat_cls = breakdown_strategy_cls
        self._bd_prob_cls = breakdown_problem_cls
        self._bd_settings = breakdown_settings
        self._bd_opt = breakdown_optimiser

        self._eq_prob_cls = equilibrium_problem_cls
        self._eq_opt = equilibrium_optimiser

        defaults = {
            "gamma": 1e-8,
            "target_rms_max": 0.5,
            "target_rms_con_tol": 1e-6,
        }
        self._eq_settings = {**defaults, **equilibrium_settings}
        self._coil_cons = coil_constraints

        super().__init__()

    def run_premagnetisation(self):
        """
        Run the breakdown optimisation problem
        """
        R_0 = self.params.R_0.value
        strategy = self._bd_strat_cls(
            R_0, self.params.A.value, self.params.tk_sol_ib.value
        )
        coilset = deepcopy(self.coilset)

        relaxed = all([c.flag_sizefix for c in coilset.coils.values()])
        i = 0
        i_max = 30
        while i == 0 or not relaxed:
            coilset.mesh_coils(0.1)
            breakdown = Breakdown(coilset, self.grid, R_0=R_0)

            constraints = deepcopy(self._coil_cons)
            for constraint in constraints:
                constraint._args["eq"] = breakdown

            # Coilset max currents known because the coilset geometry is fixed
            max_currents = self.coilset.get_max_currents(0)

            problem = self._bd_prob_cls(
                coilset,
                strategy,
                B_stray_max=self.params.B_premag_stray_max.value,
                B_stray_con_tol=self._bd_settings["B_stray_con_tol"],
                n_B_stray_points=self._bd_settings["n_B_stray_points"],
                optimiser=self._bd_opt,
                max_currents=max_currents,
                constraints=constraints,
            )
            coilset = problem.optimise(max_currents / 1e6)
            breakdown = Breakdown(coilset, self.grid, R_0=R_0)
            breakdown.set_breakdown_point(*strategy.breakdown_point)
            psi_premag = breakdown.breakdown_psi
            bluemira_print(f"Premagnetisation flux = {2*np.pi * psi_premag:.2f} V.s")

            if i == 0:
                psi_1 = psi_premag
            else:
                relaxed = np.isclose(psi_premag, psi_1, rtol=1e-2)
                psi_1 = psi_premag
            i += 1
            if i == i_max:
                raise EquilibriaError(
                    "Unable to relax the breakdown optimisation for coil sizes."
                )

        self.take_snapshot(self.BREAKDOWN, breakdown, coilset, self._bd_opt, problem)

    def optimise_currents(self):
        """
        Optimise the coil currents at the start and end of the current flat-top.
        """
        psi_sof, psi_eof = self.calculate_sof_eof_fluxes()
        if self.EQ_REF not in self.snapshots:
            self.run_reference_equilibrium()
        eq_ref = self.snapshots[self.EQ_REF].eq

        snapshots = [self.SOF, self.EOF]

        max_currents = self.coilset.get_max_currents(0)
        for snap, psi_boundary in zip(snapshots, [psi_sof, psi_eof]):
            eq = deepcopy(eq_ref)
            fixed_coils = all([c.flag_sizefix for c in eq.coilset.coils.values()])
            eq.coilset.mesh_coils(0.2)
            self.eq_targets(eq, I_not_dI=True, fixed_coils=fixed_coils)
            self.eq_targets.update_psi_boundary(psi_boundary / (2 * np.pi))
            _, A, b = self.eq_targets.get_weighted_arrays()

            L2_target_constraint = OptimisationConstraint(  # noqa: N806
                L2_norm_constraint,
                f_constraint_args={
                    "a_mat": A,
                    "b_vec": b,
                    "value": self._eq_settings["target_rms_max"],
                    "scale": 1e6,
                },
                tolerance=np.array([self._eq_settings["target_rms_con_tol"]]),
            )

            optimiser = deepcopy(self._eq_opt)
            constraints = deepcopy(self._coil_cons)
            for constraint in constraints:
                constraint._args["eq"] = eq
            constraints.append(L2_target_constraint)
            problem = self._eq_prob_cls(eq, max_currents, optimiser, constraints)
            coilset = problem.optimise()
            self.take_snapshot(
                snap, eq, coilset, optimiser, self.eq_targets, self.profiles
            )
