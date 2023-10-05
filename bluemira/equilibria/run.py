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
Interface for building and loading equilibria and coilset designs
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Type, Union

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.look_and_feel import bluemira_print, bluemira_warn
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.equilibria.equilibrium import Breakdown, Equilibrium, MHDState
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.optimisation.constraints import (
    MagneticConstraint,
    MagneticConstraintSet,
    PsiBoundaryConstraint,
    PsiConstraint,
    UpdateableConstraint,
)
from bluemira.equilibria.optimisation.problem import (
    BreakdownCOP,
    BreakdownZoneStrategy,
    CircularZoneStrategy,
    CoilsetOptimisationProblem,
    MinimalCurrentCOP,
    PulsedNestedPositionCOP,
    TikhonovCurrentCOP,
    UnconstrainedTikhonovCurrentGradientCOP,
)
from bluemira.equilibria.physics import calc_psib
from bluemira.equilibria.solve import (
    ConvergenceCriterion,
    DudsonConvergence,
    PicardIterator,
)
from bluemira.optimisation import Algorithm, AlgorithmType

if TYPE_CHECKING:
    from bluemira.equilibria.coils import CoilSet
    from bluemira.equilibria.grid import Grid
    from bluemira.equilibria.limiter import Limiter
    from bluemira.equilibria.optimisation.problem.base import CoilsetOptimiserResult
    from bluemira.equilibria.profiles import Profile
    from bluemira.geometry.coordinates import Coordinates
    from bluemira.utilities.optimiser import Optimiser
    from bluemira.utilities.positioning import PositionMapper


@dataclass
class Snapshot:
    """
    Abstract object for grouping of equilibria objects in a given state.

    Parameters
    ----------
    eq:
        The equilibrium at the snapshot
    coilset:
        The coilset at the snapshot
    opt_problem:
        The constraints at the snapshot
    profiles:
        The profile at the snapshot
    optimisation_result:
        The optimisation result
    limiter:
        The limiter for the snapshot
    tfcoil:
        The PF coil placement boundary
    """

    eq: MHDState
    coilset: CoilSet
    constraints: Optional[CoilsetOptimisationProblem] = None
    profiles: Optional[Profile] = None
    optimisation_result: Optional[CoilsetOptimiserResult] = None
    limiter: Optional[Limiter] = None
    tfcoil: Optional[Coordinates] = None

    def __post_init__(self):
        """Copy some variables on initialisation"""
        for fld in fields(type(self)):
            if (val := getattr(self, fld.name)) is not None and fld.name not in (
                "constraints",
                "tfcoil",
            ):
                setattr(self, fld.name, deepcopy(val))


@dataclass
class BreakdownCOPSettings:
    """Breakdown settings for PulsedCoilsetDesign"""

    problem: Type[BreakdownCOP] = BreakdownCOP
    strategy: Type[BreakdownZoneStrategy] = CircularZoneStrategy
    algorithm: AlgorithmType = Algorithm.COBYLA
    opt_conditions: Dict[str, Union[float, int]] = field(
        default_factory=lambda: {"max_eval": 5000, "ftol_rel": 1e-10}
    )
    B_stray_con_tol: float = 1e-8
    n_B_stray_points: int = 20


@dataclass
class EQSettings:
    """Equilibrium settings for PulsedCoilsetDesign"""

    problem: Type[CoilsetOptimisationProblem] = MinimalCurrentCOP
    convergence: ConvergenceCriterion = field(
        default_factory=lambda: DudsonConvergence(1e-2)
    )
    algorithm: AlgorithmType = Algorithm.SLSQP
    opt_conditions: Dict[str, Union[float, int]] = field(
        default_factory=lambda: {"max_eval": 1000, "ftol_rel": 1e-6}
    )
    opt_parameters: Dict[str, Any] = field(
        default_factory=lambda: {"initial_step": 0.03}
    )
    coil_mesh_size: float = 0.3
    gamma: float = 1e-8
    relaxation: float = 0.1
    peak_PF_current_factor: float = 1.5


@dataclass
class PositionSettings:
    """Position optimiser settings"""

    problem: Type[PulsedNestedPositionCOP] = PulsedNestedPositionCOP
    algorithm: AlgorithmType = Algorithm.COBYLA
    opt_conditions: Dict[str, Union[float, int]] = field(
        default_factory=lambda: {"max_eval": 100, "ftol_rel": 1e-4}
    )


@dataclass
class PulsedCoilsetDesignFrame(ParameterFrame):
    """PulsedCoilsetDesign Parameters"""

    A: Parameter[float]
    B_premag_stray_max: Parameter[float]
    C_Ejima: Parameter[float]
    I_p: Parameter[float]
    l_i: Parameter[float]
    R_0: Parameter[float]
    tau_flattop: Parameter[float]
    tk_sol_ib: Parameter[float]
    v_burn: Parameter[float]


class PulsedCoilsetDesign(ABC):
    """
    Abstract base class for the procedural design of a pulsed tokamak poloidal field
    coilset.

    Parameters
    ----------
    params:
        Parameter frame with which to perform the problem
    coilset:
        PF coilset to use in the equilibrium design
    grid:
        Grid to use in the equilibrium design
    equilibrium_constraints:
        List of magnetic constraints to use for equilibria. Depending on the optimisation
        problem, these may be used in the objective function or constraints
    profiles:
        Plasma profile object to use when solving equilibria
    breakdown_strategy_cls:
        BreakdownZoneStrategy class to use when determining breakdown constraints
    breakdown_problem_cls:
        Coilset optimisation problem class for the breakdown phase
    breakdown_optimiser:
        Optimiser for the breakdown,
        default is COBYLA with ftol_rel=1e-10 and max_eval=5000
    breakdown_settings:
        Breakdown optimiser settings
    equilibrium_problem_cls:
        Coilset optimisation problem class for the equilibria and current vector
    equilibrium_optimiser:
        Optimiser for the equilibria and current vector
        default is SLSQP with ftol_rel=1e-6 and max_eval=1000
    equilibrium_convergence:
        Convergence criteria to use when solving equilibria
        default is 1e-2 DudsonConvergence
    equilibrium_settings:
        Settings for the solution of equilibria
    current_opt_constraints:
        List of current optimisation constraints for equilibria
    coil_constraints:
        List of coil current optimisation constraints for all snapshots (including
        breakdown)
    limiter:
        Limiter to use when solving equilibria
    """

    BREAKDOWN = "Breakdown"
    EQ_REF = "Reference"
    SOF = "SOF"
    EOF = "EOF"

    def __init__(
        self,
        params: ParameterFrame,
        coilset: CoilSet,
        grid: Grid,
        equilibrium_constraints: List[MagneticConstraint],
        profiles: Profile,
        breakdown_settings: Optional[Union[Dict, BreakdownCOPSettings]] = None,
        equilibrium_settings: Optional[Union[Dict, EQSettings]] = None,
        # Remove in v2
        breakdown_strategy_cls: Optional[Type[BreakdownZoneStrategy]] = None,
        breakdown_problem_cls: Optional[Type[BreakdownCOP]] = None,
        breakdown_optimiser: Optional[Optimiser] = None,
        equilibrium_problem_cls: Optional[Type[CoilsetOptimisationProblem]] = None,
        equilibrium_optimiser: Optional[Optimiser] = None,
        equilibrium_convergence: Optional[ConvergenceCriterion] = None,
        # eos
        current_opt_constraints: Optional[List[UpdateableConstraint]] = None,
        coil_constraints: Optional[List[UpdateableConstraint]] = None,
        limiter: Optional[Limiter] = None,
    ):
        self.snapshots = {}
        self.params = PulsedCoilsetDesignFrame.from_frame(params)
        self.coilset = coilset
        self.grid = grid

        self._current_opt_cons = current_opt_constraints
        self.eq_constraints = equilibrium_constraints
        self.profiles = profiles
        self.bd_settings = breakdown_settings
        self.eq_settings = equilibrium_settings

        # Remove in v2
        warn_string = (
            f"Use of {type(self).__name__}'s '{{}}' argument is "
            "deprecated and it will be removed in version 2.0.0.\n"
            "See "
            "https://bluemira.readthedocs.io/en/latest/optimisation/"
            "optimisation.html "
            "for documentation of the new optimisation module."
        )
        if breakdown_optimiser is not None:
            warnings.warn(
                warn_string.format("breakdown_optimiser"),
                DeprecationWarning,
                stacklevel=2,
            )
            # warn
            self.bd_settings.opt_conditions = {
                **(
                    {}
                    if breakdown_optimiser.opt_conditions is None
                    else breakdown_optimiser.opt_conditions
                ),
                **self.bd_settings.opt_conditions,
            }

            self.bd_settings.algorithm = breakdown_optimiser.algorithm_name
        if breakdown_problem_cls is not None:
            warnings.warn(
                warn_string.format("breakdown_problem_cls"),
                DeprecationWarning,
                stacklevel=2,
            )
            self.bd_settings.problem = breakdown_problem_cls

        if breakdown_strategy_cls is not None:
            warnings.warn(
                warn_string.format("breakdown_strategy_cls"),
                DeprecationWarning,
                stacklevel=2,
            )
            self.bd_settings.strategy = breakdown_strategy_cls

        if equilibrium_optimiser is not None:
            warnings.warn(
                warn_string.format("equilibrium_optimiser"),
                DeprecationWarning,
                stacklevel=2,
            )
            self.eq_settings.opt_conditions = {
                **(
                    {}
                    if equilibrium_optimiser.opt_conditions is None
                    else equilibrium_optimiser.opt_conditions
                ),
                **self.eq_settings.opt_conditions,
            }

            self.eq_settings.algorithm = equilibrium_optimiser.algorithm_name
            self.eq_settings.opt_parameters = {
                **(
                    {}
                    if equilibrium_optimiser.opt_parameters is None
                    else equilibrium_optimiser.opt_parameters
                ),
                **self.eq_settings.opt_parameters,
            }

        if equilibrium_convergence is not None:
            warnings.warn(
                warn_string.format("equilibrium_convergence"),
                DeprecationWarning,
                stacklevel=2,
            )
            self.eq_settings.convergence = equilibrium_convergence
        if equilibrium_problem_cls is not None:
            warnings.warn(
                warn_string.format("equilibrium_problem_cls"),
                DeprecationWarning,
                stacklevel=2,
            )
            self.eq_settings.problem = equilibrium_problem_cls
        # eos

        self._coil_cons = [] if coil_constraints is None else coil_constraints
        self.limiter = limiter

    @abstractmethod
    def optimise(self, *args, **kwargs) -> CoilSet:
        """Run pulsed coilset design optimisation."""

    @property
    def bd_settings(self) -> BreakdownCOPSettings:
        """Breakdown COP settings."""
        return self.__bd_settings

    @property
    def _bd_settings(self) -> BreakdownCOPSettings:
        return self.__bd_settings

    @bd_settings.setter
    def bd_settings(self, value: Optional[Union[Dict, BreakdownCOPSettings]] = None):
        """Breakdown COP settings."""
        if value is None:
            self.__bd_settings = BreakdownCOPSettings()
        elif isinstance(value, BreakdownCOPSettings):
            self.__bd_settings = value
        else:
            self.__bd_settings = BreakdownCOPSettings(**value)

    @property
    def eq_settings(self) -> EQSettings:
        """Equilibrium COP settings."""
        return self.__eq_settings

    @property
    def _eq_settings(self) -> EQSettings:
        return self.__eq_settings

    @eq_settings.setter
    def eq_settings(self, value: Optional[Union[EQSettings, Dict]] = None):
        """Equilibrium COP settings."""
        if value is None:
            self.__eq_settings = EQSettings()
        elif isinstance(value, EQSettings):
            self.__eq_settings = value
        else:
            self.__eq_settings = EQSettings(**value)

    def take_snapshot(
        self,
        name: str,
        eq: MHDState,
        coilset: CoilSet,
        problem: CoilsetOptimisationProblem,
        profiles: Optional[Profile] = None,
    ):
        """Take a snapshot of the pulse."""
        if name in self.snapshots:
            bluemira_warn(f"Over-writing snapshot {name}!")

        self.snapshots[name] = Snapshot(
            eq, coilset, problem, profiles, limiter=self.limiter
        )

    def run_premagnetisation(self):
        """Run the breakdown optimisation problem."""
        R_0 = self.params.R_0.value
        strategy = self.bd_settings.strategy(
            R_0, self.params.A.value, self.params.tk_sol_ib.value
        )

        i_max = 30
        relaxed = all(self.coilset._flag_sizefix)
        for i in range(i_max):
            coilset = deepcopy(self.coilset)
            breakdown = Breakdown(coilset, self.grid)
            constraints = deepcopy(self._coil_cons)

            if relaxed:
                max_currents = self.coilset.get_max_current(0)
            else:
                max_currents = self.coilset.get_max_current(self.params.I_p.value)
                coilset.get_control_coils().current = max_currents
                coilset.discretisation = self.eq_settings.coil_mesh_size

            problem = self.bd_settings.problem(
                breakdown.coilset,
                breakdown,
                strategy,
                B_stray_max=self.params.B_premag_stray_max.value,
                B_stray_con_tol=self.bd_settings.B_stray_con_tol,
                n_B_stray_points=self.bd_settings.n_B_stray_points,
                max_currents=max_currents,
                opt_algorithm=self.bd_settings.algorithm,
                opt_conditions=self.bd_settings.opt_conditions,
                constraints=constraints,
            )
            result = problem.optimise(x0=max_currents, fixed_coils=False)
            breakdown.set_breakdown_point(*strategy.breakdown_point)
            psi_premag = breakdown.breakdown_psi

            if i == 0:
                psi_1 = psi_premag
            if relaxed or np.isclose(psi_premag, psi_1, rtol=1e-2):
                break

        else:
            raise EquilibriaError(
                "Unable to relax the breakdown optimisation for coil sizes."
            )

        bluemira_print(f"Premagnetisation flux = {2*np.pi * psi_premag:.2f} V.s")

        self._psi_premag = 2 * np.pi * psi_premag
        self.take_snapshot(self.BREAKDOWN, breakdown, result.coilset, problem)

    def run_reference_equilibrium(self):
        """Run a reference equilibrium."""
        coilset = deepcopy(self.coilset)
        eq = Equilibrium(
            coilset,
            self.grid,
            self.profiles,
        )
        opt_problem = UnconstrainedTikhonovCurrentGradientCOP(
            coilset,
            eq,
            MagneticConstraintSet(self.eq_constraints),
            gamma=self.eq_settings.gamma,
        )
        program = PicardIterator(
            eq,
            opt_problem,
            convergence=self.eq_settings.convergence,
            relaxation=self.eq_settings.relaxation,
            fixed_coils=True,
            plot=False,
        )
        program()

        opt_problem = self._make_opt_problem(
            eq,
            self._get_max_currents(eq.coilset),
            current_constraints=None,
            eq_constraints=[
                deepcopy(con)
                for con in self.eq_constraints
                if not isinstance(con, (PsiConstraint, PsiBoundaryConstraint))
            ],
        )

        program = PicardIterator(
            eq,
            opt_problem,
            convergence=self.eq_settings.convergence,
            relaxation=self.eq_settings.relaxation,
            fixed_coils=True,
            plot=False,
        )
        program()

        self.take_snapshot(self.EQ_REF, eq, coilset, opt_problem, self.profiles)

    def calculate_sof_eof_fluxes(self, psi_premag: Optional[float] = None):
        """Calculate the SOF and EOF plasma boundary fluxes."""
        if psi_premag is None:
            if self.BREAKDOWN not in self.snapshots:
                self.run_premagnetisation()
            psi_premag = self.snapshots[self.BREAKDOWN].eq.breakdown_psi

        psi_sof = calc_psib(
            2 * np.pi * psi_premag,
            self.params.R_0.value,
            self.params.I_p.value,
            self.params.l_i.value,
            self.params.C_Ejima.value,
        )
        psi_eof = psi_sof - self.params.tau_flattop.value * self.params.v_burn.value
        return psi_sof, psi_eof

    def _get_max_currents(self, coilset: CoilSet) -> np.ndarray:
        return coilset.get_max_current(
            self.eq_settings.peak_PF_current_factor * self.params.I_p.value
        )

    def get_sof_eof_opt_problems(
        self, psi_sof: float, psi_eof: float
    ) -> List[CoilsetOptimisationProblem]:
        """Get start of flat top and end of flat top optimisation problems."""
        eq_ref = self.snapshots[self.EQ_REF].eq
        max_currents_pf = self._get_max_currents(self.coilset.get_coiltype("PF"))
        max_currents = self._get_max_currents(self.coilset)

        opt_problems = []
        for psi_boundary in [psi_sof, psi_eof]:
            eq = deepcopy(eq_ref)
            eq.coilset.get_coiltype("PF").resize(max_currents_pf)

            current_constraints = []
            if self._current_opt_cons:
                current_constraints += deepcopy(self._current_opt_cons)
            if self._coil_cons:
                current_constraints += deepcopy(self._coil_cons)

            eq_constraints = deepcopy(self.eq_constraints)
            for constraints in (eq_constraints, current_constraints):
                for con in constraints:
                    if isinstance(con, (PsiBoundaryConstraint, PsiConstraint)):
                        con.target_value = psi_boundary / (2 * np.pi)

            opt_problems.append(
                self._make_opt_problem(
                    eq, max_currents, current_constraints, eq_constraints
                )
            )

        return opt_problems

    def _make_opt_problem(
        self,
        eq: Equilibrium,
        max_currents: np.ndarray,
        current_constraints: Optional[List[UpdateableConstraint]],
        eq_constraints: List[MagneticConstraint],
    ) -> CoilsetOptimisationProblem:
        if self.eq_settings.problem == MinimalCurrentCOP:
            constraints = eq_constraints
            if current_constraints:
                constraints += current_constraints

            problem = self.eq_settings.problem(
                eq.coilset,
                eq,
                max_currents=max_currents,
                opt_conditions=self.eq_settings.opt_conditions,
                opt_algorithm=self.eq_settings.algorithm,
                constraints=constraints,
            )
        elif self.eq_settings.problem == TikhonovCurrentCOP:
            problem = self.eq_settings.problem(
                eq.coilset,
                eq,
                MagneticConstraintSet(eq_constraints),
                gamma=self.eq_settings.gamma,
                opt_conditions=self.eq_settings.opt_conditions,
                opt_algorithm=self.eq_settings.algorithm,
                opt_parameters=self.eq_settings.opt_parameters,
                max_currents=max_currents,
                constraints=current_constraints,
            )
        else:
            raise EquilibriaError(
                "Only MinimalCurrentCOP and TikhonovCurrentCOP"
                " equilibrium problems supported"
            )
        return problem

    def converge_equilibrium(self, eq: Equilibrium, problem: CoilsetOptimisationProblem):
        """Converge an equilibrium problem from a 'frozen' plasma optimised state."""
        program = PicardIterator(
            eq,
            problem,
            fixed_coils=True,
            convergence=self._eq_settings.convergence,
            relaxation=self.eq_settings.relaxation,
            plot=False,
        )
        program()

    def converge_and_snapshot(
        self,
        sub_opt_problems: Iterable[CoilsetOptimisationProblem],
        problem_names: Iterable[str] = (SOF, EOF),
    ):
        """Converge equilibrium optimisation problems and take snapshots."""
        for snap, problem in zip(problem_names, sub_opt_problems):
            eq = problem.eq
            self.converge_equilibrium(eq, problem)
            self.take_snapshot(snap, eq, eq.coilset, problem, eq.profiles)

    def plot(self):
        """Plot the pulsed equilibrium problem."""
        n_snapshots = len(self.snapshots)
        if n_snapshots == 0:
            return None

        f, ax = plt.subplots(1, n_snapshots)
        for i, (k, snap) in enumerate(self.snapshots.items()):
            axi = ax[i]
            snap.eq.plot(ax=axi)
            snap.coilset.plot(ax=axi)
            if k == self.BREAKDOWN:
                psi_name, psi_val = r"$\Psi_{bd}$", snap.eq.breakdown_psi
            else:
                psi_name, psi_val = r"$\Psi_{b}$", snap.eq.get_OX_points()[1][0].psi

            axi.set_title(f"{k} {psi_name}: {2* np.pi * psi_val:.2f} V.s")
        return f


class FixedPulsedCoilsetDesign(PulsedCoilsetDesign):
    """Procedural design for a pulsed tokamak with a known, fixed PF coilset."""

    def optimise(self) -> CoilSet:
        """Run pulsed coilset design optimisation."""
        self.optimise_currents()
        return self.coilset

    def optimise_currents(self):
        """Optimise the coil currents at the start and end of the current flat-top."""
        psi_sof, psi_eof = self.calculate_sof_eof_fluxes()
        if self.EQ_REF not in self.snapshots:
            self.run_reference_equilibrium()

        self.converge_and_snapshot(self.get_sof_eof_opt_problems(psi_sof, psi_eof))


class OptimisedPulsedCoilsetDesign(PulsedCoilsetDesign):
    """
    Procedural design for a pulsed tokamak with no prescribed PF coil positions.

    Parameters
    ----------
    params:
        Parameter frame with which to perform the problem
    coilset:
        PF coilset to use in the equilibrium design
    position_mapper:
        Normalised coil position mapping
    grid:
        Grid to use in the equilibrium design
    equilibrium_constraints:
        List of magnetic constraints to use for equilibria. Depending on the optimisation
        problem, these may be used in the objective function or constraints
    profiles:
        Plasma profile object to use when solving equilibria
    breakdown_strategy_cls:
        BreakdownZoneStrategy class to use when determining breakdown constraints
    breakdown_problem_cls:
        Coilset optimisation problem class for the breakdown phase
    breakdown_optimiser:
        Optimiser for the breakdown,
        default is COBYLA with ftol_rel=1e-10 and max_eval=5000
    breakdown_settings:
        Breakdown optimiser settings
    equilibrium_problem_cls:
        Coilset optimisation problem class for the equilibria and current vector
    equilibrium_optimiser:
        Optimiser for the equilibria and current vector
        default is SLSQP with ftol_rel=1e-6 and max_eval=1000
    equilibrium_convergence:
        Convergence criteria to use when solving equilibria
        default is 1e-2 DudsonConvergence
    equilibrium_settings:
        Settings for the solution of equilibria
    current_opt_constraints:
        List of current optimisation constraints for equilibria
    coil_constraints:
        List of coil current optimisation constraints for all snapshots (including
        breakdown)
    limiter:
        Limiter to use when solving equilibria
    position_problem_cls:
        Coilset optimisation problem class for the coil positions
    position_optimiser:
        Optimiser for the coil positions
        default is COBYLA with ftol_rel=1e-4 and max_eval=100
    """

    def __init__(
        self,
        params: ParameterFrame,
        coilset: CoilSet,
        position_mapper: PositionMapper,
        grid: Grid,
        equilibrium_constraints: List[MagneticConstraint],
        profiles: Profile,
        breakdown_settings: Optional[Union[Dict, BreakdownCOPSettings]] = None,
        equilibrium_settings: Optional[Union[Dict, EQSettings]] = None,
        # Remove in v2
        breakdown_strategy_cls: Optional[Type[BreakdownZoneStrategy]] = None,
        breakdown_problem_cls: Optional[Type[BreakdownCOP]] = None,
        breakdown_optimiser: Optional[Optimiser] = None,
        equilibrium_problem_cls: Optional[Type[CoilsetOptimisationProblem]] = None,
        equilibrium_optimiser: Optional[Optimiser] = None,
        equilibrium_convergence: Optional[ConvergenceCriterion] = None,
        # eos
        current_opt_constraints: Optional[List[UpdateableConstraint]] = None,
        coil_constraints: Optional[List[UpdateableConstraint]] = None,
        limiter: Optional[Limiter] = None,
        position_settings: Optional[Union[Dict, PositionSettings]] = None,
        # Remove in v2
        position_problem_cls: Optional[Type[PulsedNestedPositionCOP]] = None,
        position_optimiser: Optional[Optimiser] = None,
    ):
        super().__init__(
            params,
            coilset,
            grid,
            equilibrium_constraints,
            profiles,
            breakdown_settings,
            equilibrium_settings,
            breakdown_strategy_cls,
            breakdown_problem_cls,
            breakdown_optimiser,
            equilibrium_problem_cls,
            equilibrium_optimiser,
            equilibrium_convergence,
            current_opt_constraints,
            coil_constraints,
            limiter,
        )
        self.coilset = self._prepare_coilset(self.coilset)
        self.position_mapper = position_mapper

        self.pos_settings = position_settings

        warn_string = (
            f"Use of {type(self).__name__}'s '{{}}' argument is "
            "deprecated and it will be removed in version 2.0.0.\n"
            "See "
            "https://bluemira.readthedocs.io/en/latest/optimisation/"
            "optimisation.html "
            "for documentation of the new optimisation module."
        )

        if position_problem_cls is not None:
            warnings.warn(
                warn_string.format("position_problem_cls"),
                DeprecationWarning,
                stacklevel=2,
            )
            self.pos_settings.problem = position_problem_cls
        if position_optimiser is not None:
            warnings.warn(
                warn_string.format("position_optimiser"),
                DeprecationWarning,
                stacklevel=2,
            )
            self.pos_settings.algorithm = position_optimiser.algorithm_name
            self.pos_settings.opt_conditions = {
                **(
                    {}
                    if position_optimiser.opt_conditions is None
                    else position_optimiser.opt_conditions
                ),
                **self.pos_settings.opt_conditions,
            }

    @property
    def pos_settings(self) -> PositionSettings:
        """Position COP settings."""
        return self._pos_settings

    @pos_settings.setter
    def pos_settings(self, value: Optional[Union[PositionSettings, Dict]] = None):
        """Position COP settings."""
        if value is None:
            self._pos_settings = PositionSettings()
        elif isinstance(value, PositionSettings):
            self._pos_settings = value
        else:
            self._pos_settings = PositionSettings(**value)

    def _prepare_coilset(self, coilset: CoilSet) -> CoilSet:
        coilset = deepcopy(coilset)
        coilset.discretisation = np.where(
            coilset._flag_sizefix,
            self.eq_settings.coil_mesh_size,
            coilset.discretisation,
        )
        return coilset

    def optimise(self, verbose: bool = False) -> CoilSet:
        """
        Optimise the coil positions for the start and end of the current flat-top.
        """
        psi_sof, psi_eof = self.calculate_sof_eof_fluxes()
        if self.EQ_REF not in self.snapshots:
            self.run_reference_equilibrium()

        sub_opt_problems = self.get_sof_eof_opt_problems(psi_sof, psi_eof)

        pos_opt_problem = self.pos_settings.problem(
            sub_opt_problems[0].eq.coilset,
            self.position_mapper,
            sub_opt_problems,
            self.pos_settings.algorithm,
            self.pos_settings.opt_conditions,
            constraints=None,
        )
        result = pos_opt_problem.optimise(verbose=verbose)
        optimised_coilset = self._consolidate_coilset(result.coilset, sub_opt_problems)

        self.converge_and_snapshot(sub_opt_problems)

        # Re-run breakdown
        psi_bd_orig = self._psi_premag
        self.coilset = optimised_coilset
        self.run_premagnetisation()
        if self._psi_premag < psi_bd_orig - 2.0:
            bluemira_warn(
                "Breakdown flux significantly lower with optimised coil positions: "
                f"{self._psi_premag:.2f} < {psi_bd_orig:.2f}"
            )
        return optimised_coilset

    def _consolidate_coilset(
        self, coilset: CoilSet, sub_opt_problems: Iterable[CoilsetOptimisationProblem]
    ) -> CoilSet:
        """
        Set the current bounds on the current optimisation problems, fix coil sizes, and
        mesh.
        """
        max_cs_currents = coilset.get_coiltype("CS").get_max_current(0.0)
        pf_currents = []
        for problem in sub_opt_problems:
            pf_coils = problem.eq.coilset.get_coiltype("PF").get_control_coils()
            pf_currents.append(np.abs(pf_coils.current))

        max_pf_currents = np.max(pf_currents, axis=0)
        # Relax the max currents a bit to avoid oscillation
        max_pf_current = self.eq_settings.peak_PF_current_factor * self.params.I_p.value
        max_pf_currents = np.clip(1.1 * max_pf_currents, 0, max_pf_current)

        for problem in sub_opt_problems:
            pf_coils = problem.eq.coilset.get_coiltype("PF").get_control_coils()
            pf_coils.resize(max_pf_currents)
            pf_coils.fix_sizes()
            pf_coils.discretisation = self.eq_settings.coil_mesh_size
            problem.set_current_bounds(
                np.concatenate([max_pf_currents, max_cs_currents])
            )

        consolidated_coilset = deepcopy(problem.eq.coilset)
        consolidated_coilset.fix_sizes()
        consolidated_coilset.get_control_coils().current = 0
        return consolidated_coilset
