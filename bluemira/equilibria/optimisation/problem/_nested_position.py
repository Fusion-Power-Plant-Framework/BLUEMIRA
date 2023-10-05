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
OptimisationProblems for coilset design.

New optimisation schemes for the coilset can be provided by subclassing
from CoilsetOP, which is an abstract base class for OptimisationProblems
that use a coilset as their parameterisation object.

Subclasses must provide an optimise() method that returns an optimised
coilset according to a given optimisation objective function.
As the exact form of the state vector that is optimised is often
specific to each objective function, each subclass of CoilsetOP is
generally also specific to a given objective function, since
the method used to map the coilset object to the state vector
(and additional required arguments) will generally differ in each case.

"""

from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt

from bluemira.base.look_and_feel import bluemira_print_flush
from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.constraints import (
    MagneticConstraintSet,
    UpdateableConstraint,
)
from bluemira.equilibria.optimisation.problem.base import (
    CoilsetOptimisationProblem,
    CoilsetOptimiserResult,
)
from bluemira.optimisation import Algorithm, AlgorithmType, optimise
from bluemira.utilities.positioning import PositionMapper


class NestedCoilsetPositionCOP(CoilsetOptimisationProblem):
    """
    Coilset OptimisationProblem for coil currents and positions
    subject to maximum current bounds and positions bounded within
    a provided region. Performs a nested optimisation for coil
    currents within each position optimisation function call.

    Parameters
    ----------
    sub_opt:
        Coilset OptimisationProblem to use for the optimisation of
        coil currents at each trial set of coil positions.
        sub_opt.coilset must exist, and will be modified
        during the optimisation.
    eq:
        Equilibrium object used to update magnetic field targets.
    targets:
        Set of magnetic field targets to use in objective function.
    position_mapper:
        position mapper object of the regions to optimise the coil positions within
    opt_algorithm:
        The optimisation algorithm to use (e.g. SLSQP)
    opt_conditions:
        The stopping conditions for the optimiser.
        for defaults see
        :class:`~bluemira.optimisation._algorithm.AlgorithDefaultTolerances`
        along with `max_eval=100`


    Notes
    -----
    Setting stopval and maxeval is the most reliable way to stop optimisation
    at the desired figure of merit and number of iterations respectively.
    Some NLOpt optimisers display unexpected behaviour when setting xtol and
    ftol, and may not terminate as expected when those criteria are reached.
    """

    def __init__(
        self,
        sub_opt: CoilsetOptimisationProblem,
        eq: Equilibrium,
        targets: MagneticConstraintSet,
        position_mapper: PositionMapper,
        opt_algorithm: AlgorithmType = Algorithm.SBPLX,
        opt_conditions: Optional[Dict[str, float]] = None,
        constraints: Optional[List[UpdateableConstraint]] = None,
    ):
        self.eq = eq
        self.targets = targets
        self.position_mapper = position_mapper

        opt_dimension = self.position_mapper.dimension
        self.bounds = (np.zeros(opt_dimension), np.ones(opt_dimension))
        self.coilset = sub_opt.coilset
        self.sub_opt = sub_opt
        self.opt_algorithm = opt_algorithm
        self.opt_conditions = opt_conditions or self._opt_condition_defaults(
            {"max_eval": 100}
        )
        self._constraints = [] if constraints is None else constraints

        self.initial_state, self.substates = self.read_coilset_state(
            self.coilset, self.scale
        )
        self.I0 = np.array_split(self.initial_state, self.substates)[2]

    def optimise(self, **_):
        """
        Run the optimisation.

        Returns
        -------
        Optimised CoilSet object.
        """
        # Get initial currents, and trim to within current bounds.
        initial_state, substates = self.read_coilset_state(self.coilset, self.scale)
        _, _, initial_currents = np.array_split(initial_state, substates)
        initial_mapped_positions = self.position_mapper.to_L(
            self.coilset.x, self.coilset.z
        )

        # TODO: find more explicit way of passing this to objective?
        self.I0 = initial_currents
        eq_constraints, ineq_constraints = self._make_numerical_constraints()
        opt_result = optimise(
            f_objective=self.objective,
            x0=initial_mapped_positions,
            algorithm=self.opt_algorithm,
            opt_conditions=self.opt_conditions,
            eq_constraints=eq_constraints,
            ineq_constraints=ineq_constraints,
        )
        self.set_coilset_state(self.coilset, opt_result.x, self.scale)
        return CoilsetOptimiserResult.from_opt_result(self.coilset, opt_result)

    def objective(self, vector: npt.NDArray) -> float:
        """Objective function to minimise."""
        coilset_state = np.concatenate((self.position_mapper.to_xz(vector), self.I0))
        self.set_coilset_state(self.coilset, coilset_state, self.scale)

        # Update targets
        self.eq._remap_greens()
        self.targets(self.eq, I_not_dI=True, fixed_coils=False)

        # Run the sub-optimisation
        sub_opt_result = self.sub_opt.optimise()
        return sub_opt_result.f_x


class PulsedNestedPositionCOP(CoilsetOptimisationProblem):
    """
    Coilset position optimisation problem for multiple sub-optimisation problems.

    Parameters
    ----------
    coilset:
        Coilset for which to optimise positions
    position_mapper:
        Position mapper tool to parameterise coil positions
    sub_opt_problems:
        The list of sub-optimisation problems to solve
    opt_algorithm:
        The optimisation algorithm to use (e.g. SLSQP)
    opt_conditions:
        The stopping conditions for the optimiser.
        for defaults see
        :class:`~bluemira.optimisation._algorithm.AlgorithDefaultTolerances`
        along with `max_eval=100`
    constraints:
        Constraints to use. Note these should be applicable to the parametric position
        vector
    initial_currents:
        Initial currents to use when solving the current sub-optimisation problems
    debug:
        Whether or not to run in debug mode (will affect run-time noticeably)
    """

    def __init__(
        self,
        coilset: CoilSet,
        position_mapper: PositionMapper,
        sub_opt_problems: List[CoilsetOptimisationProblem],
        opt_algorithm: AlgorithmType = Algorithm.COBYLA,
        opt_conditions: Optional[Dict[str, float]] = None,
        constraints=None,
        initial_currents=None,
        debug: bool = False,
    ):
        self.coilset = coilset
        self.position_mapper = position_mapper
        self.sub_opt_problems = sub_opt_problems
        self.opt_algorithm = opt_algorithm
        self.opt_conditions = opt_conditions or self._opt_condition_defaults(
            {"max_eval": 100}
        )
        self._constraints = constraints

        if initial_currents:
            self.initial_currents = initial_currents / self.sub_opt_problems[0].scale
        else:
            self.initial_currents = np.zeros(coilset.get_control_coils().n_coils())
        self.debug = {0: debug}
        self.iter = {0: 0.0}
        opt_dimension = self.position_mapper.dimension
        self.bounds = (np.zeros(opt_dimension), np.ones(opt_dimension))

    @staticmethod
    def _run_reporting(itern, max_fom, verbose):
        """
        Keep track of objective function value over iterations.
        """
        i = max(list(itern.keys())) + 1
        itern[i] = max_fom

        if verbose:
            bluemira_print_flush(f"Coil position iteration {i} FOM value: {max_fom:.6e}")

    @staticmethod
    def _run_diagnostics(
        debug,
        sub_opt_prob: CoilsetOptimisationProblem,
        opt_result: CoilsetOptimiserResult,
    ):
        """
        In debug mode, store the LCFS at each iteration for each of the sub-optimisation
        problems.

        Notes
        -----
        This can significantly impact run-time.
        """
        if debug[0]:
            entry = max(list(debug.keys()))
            value = opt_result.f_x
            sub_opt_prob.eq._remap_greens()
            sub_opt_prob.eq._clear_OX_points()
            lcfs = sub_opt_prob.eq.get_LCFS()
            debug[entry].append([lcfs, value])

    def sub_opt_objective(self, vector: npt.NDArray, verbose: bool = False) -> float:
        """Run the sub-optimisations and return the largest figure of merit."""
        positions = self.position_mapper.to_xz_dict(vector)

        if self.debug[0]:
            # Increment debug dictionary
            i = max(list(self.debug.keys())) + 1
            self.debug[i] = []

        fom_values = []
        for sub_opt_prob in self.sub_opt_problems:
            for coil, position in positions.items():
                sub_opt_prob.coilset[coil].position = position
            result = sub_opt_prob.optimise(x0=self.initial_currents, fixed_coils=False)
            self._run_diagnostics(self.debug, sub_opt_prob, result)
            fom_values.append(result.f_x)
        max_fom = max(fom_values)

        self._run_reporting(self.iter, max_fom, verbose)
        return max_fom

    def objective(self, vector: npt.NDArray, verbose: bool = False) -> float:
        """The objective function of the parent optimisation."""
        return self.sub_opt_objective(vector, verbose=verbose)

    def _get_initial_vector(self) -> npt.NDArray:
        """
        Get a vector representation of the initial coilset state from the PositionMapper.
        """
        x, z = [], []
        for name in self.position_mapper.interpolators:
            x.append(self.coilset[name].x)
            z.append(self.coilset[name].z)
        return self.position_mapper.to_L(x, z)

    def optimise(
        self, x0: Optional[npt.NDArray] = None, verbose: bool = False
    ) -> CoilsetOptimiserResult:
        """
        Run the PulsedNestedPositionCOP

        Parameters
        ----------
        x0:
            Initial solution vector (parameterised positions)
        verbose:
            Whether or not to print progress information

        Returns
        -------
        coilset:
            Optimised CoilSet
        """
        if x0 is None:
            x0 = self._get_initial_vector()

        eq_constraints, ineq_constraints = self._make_numerical_constraints()
        opt_result = optimise(
            f_objective=lambda x: self.objective(x, verbose=verbose),
            x0=x0,
            df_objective=None,  # use a numerical approximation if needed
            algorithm=self.opt_algorithm,
            opt_conditions=self.opt_conditions,
            bounds=self.bounds,
            eq_constraints=eq_constraints,
            ineq_constraints=ineq_constraints,
        )
        optimal_positions = opt_result.x
        # Call the objective one last time
        self.sub_opt_objective(optimal_positions)

        # Clean up state of Equilibrium objects
        for sub_opt in self.sub_opt_problems:
            sub_opt.eq._remap_greens()
            sub_opt.eq._clear_OX_points()
        return CoilsetOptimiserResult.from_opt_result(self.coilset, opt_result)
