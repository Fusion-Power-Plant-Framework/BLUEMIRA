# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import numpy.typing as npt

from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.constraints import (
    MagneticConstraintSet,
    UpdateableConstraint,
)
from bluemira.equilibria.optimisation.objectives import regularised_lsq_fom
from bluemira.equilibria.optimisation.problem.base import (
    CoilsetOptimisationProblem,
    CoilsetOptimiserResult,
)
from bluemira.optimisation import Algorithm, AlgorithmType, optimise
from bluemira.utilities.positioning import PositionMapper


class CoilsetPositionCOP(CoilsetOptimisationProblem):
    """
    Coilset OptimisationProblem for coil currents and positions
    subject to maximum current bounds and positions bounded within
    a provided region.

    Coil currents and positions are optimised simultaneously.

    Parameters
    ----------
    coilset:
        Coilset to optimise.
    eq:
        Equilibrium object used to update magnetic field targets.
    targets:
        Set of magnetic field targets to use in objective function.
    position_mapper:
        Position mappings of coil regions
    max_currents:
        Maximum allowed current for each independent coil current in coilset [A].
        If specified as a float, the float will set the maximum allowed current
        for all coils.
    gamma:
        Tikhonov regularisation parameter in units of [A⁻¹].
    opt_algorithm:
        The optimisation algorithm to use (e.g. SLSQP)
    opt_conditions:
        The stopping conditions for the optimiser.
        for defaults see
        :class:`~bluemira.optimisation._algorithm.AlgorithDefaultTolerances`
        along with `max_eval=100`
    opt_parameters:
        Optimiser specific parameters,
        see https://nlopt.readthedocs.io/en/latest/NLopt_Reference/#algorithm-specific-parameters
        Otherwise, the parameters can be founded by digging through the source code.
    constraints:
        List of optimisation constraints to apply to the optimisation problem

    Notes
    -----
    Setting stopval and maxeval is the most reliable way to stop optimisation
    at the desired figure of merit and number of iterations respectively.
    Some NLOpt optimisers display unexpected behaviour when setting xtol and
    ftol, and may not terminate as expected when those criteria are reached.
    """

    def __init__(
        self,
        coilset: CoilSet,
        eq: Equilibrium,
        targets: MagneticConstraintSet,
        position_mapper: PositionMapper,
        max_currents: npt.ArrayLike | None = None,
        gamma=1e-8,
        opt_algorithm: AlgorithmType = Algorithm.SBPLX,
        opt_conditions: dict[str, float] | None = None,
        opt_parameters: dict[str, float] | None = None,
        constraints: list[UpdateableConstraint] | None = None,
    ):
        self.coilset = coilset
        self.eq = eq
        self.targets = targets
        self.position_mapper = position_mapper
        self.bounds = self.get_mapped_state_bounds(max_currents)
        self.gamma = gamma
        self.opt_algorithm = opt_algorithm
        self.opt_conditions = opt_conditions or self._opt_condition_defaults({
            "max_eval": 100
        })
        self.opt_parameters = opt_parameters
        self._constraints = [] if constraints is None else constraints

    def optimise(self, **_) -> CoilsetOptimiserResult:
        """
        Run the optimisation.

        Returns
        -------
        The result of the optimisation.
        """
        # Get initial state and apply region mapping to coil positions.
        initial_state, _ = self.read_coilset_state(self.coilset, self.scale)
        initial_x, initial_z, initial_currents = np.array_split(initial_state, 3)
        initial_mapped_positions = self.position_mapper.to_L(initial_x, initial_z)
        eq_constraints, ineq_constraints = self._make_numerical_constraints()
        opt_result = optimise(
            f_objective=self.objective,
            x0=np.concatenate((initial_mapped_positions, initial_currents)),
            bounds=self.bounds,
            opt_conditions=self.opt_conditions,
            opt_parameters=self.opt_parameters,
            algorithm=self.opt_algorithm,
            eq_constraints=eq_constraints,
            ineq_constraints=ineq_constraints,
        )
        self.set_coilset_state(self.coilset, opt_result.x, self.scale)
        return CoilsetOptimiserResult.from_opt_result(self.coilset, opt_result)

    def objective(self, vector: npt.NDArray[np.float64]) -> float:
        """
        Least-squares objective with Tikhonov regularisation term.

        Parameters
        ----------
        vector:
            The new coilset state vector.

        Returns
        -------
        The figure of merit being minimised.
        """
        # Update the coilset with the new state vector
        mapped_x, mapped_z, currents = np.array_split(vector, 3)
        mapped_positions = np.concatenate((mapped_x, mapped_z))
        x_vals, z_vals = self.position_mapper.to_xz(mapped_positions)
        coilset_state = np.concatenate((x_vals, z_vals, currents))
        self.set_coilset_state(self.coilset, coilset_state, self.scale)

        # Update target
        self.eq._remap_greens()

        # Scale the control matrix and constraint vector by weights.
        self.targets(self.eq, I_not_dI=True, fixed_coils=False)
        _, a_mat, b_vec = self.targets.get_weighted_arrays()

        return regularised_lsq_fom(currents * self.scale, a_mat, b_vec, self.gamma)[0]

    def get_mapped_state_bounds(
        self, max_currents: npt.ArrayLike | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get mapped bounds on the coilset state vector from the coil regions and
        maximum coil currents.

        Parameters
        ----------
        region_mapper:
            RegionMapper mapping coil positions within the allowed optimisation
            regions.
        max_currents:
            Maximum allowed current for each independent coil current in coilset [A].
            If specified as a float, the float will set the maximum allowed current
            for all coils.

        Returns
        -------
        bounds: np.array
            Array containing state vectors representing lower and upper bounds
            for coilset state degrees of freedom.
        """
        # Get mapped position bounds from RegionMapper
        opt_dimension = self.position_mapper.dimension
        lower_pos_bounds, upper_pos_bounds = (
            np.zeros(opt_dimension),
            np.ones(opt_dimension),
        )
        current_bounds = self.get_current_bounds(self.coilset, max_currents, self.scale)

        return (
            np.concatenate((lower_pos_bounds, current_bounds[0])),
            np.concatenate((upper_pos_bounds, current_bounds[1])),
        )
