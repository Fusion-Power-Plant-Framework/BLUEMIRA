# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from copy import deepcopy

import numpy as np
import numpy.typing as npt

from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.diagnostics import EqDiagnosticOptions
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.constraints import UpdateableConstraint
from bluemira.equilibria.optimisation.objectives import CoilCurrentsObjective
from bluemira.equilibria.optimisation.problem.base import (
    CoilsetOptimisationProblem,
    CoilsetOptimiserResult,
)
from bluemira.equilibria.plotting import EquilibriumComparisonPlotter
from bluemira.optimisation import Algorithm, AlgorithmType, optimise


class MinimalCurrentCOP(CoilsetOptimisationProblem):
    """
    Bounded, constrained, minimal current optimisation problem.

    Parameters
    ----------
    coilset:
        Coilset to optimise
    eq:
        Equilibrium object to optimise the currents for
    max_currents:
        Current bounds vector [A]
    opt_conditions:
        Optimiser conditions
    opt_algorithm:
        optimiser algorithm
    opt_parameters:
        Optimiser specific parameters,
        see https://nlopt.readthedocs.io/en/latest/NLopt_Reference/#algorithm-specific-parameters
        Otherwise, the parameters can be founded by digging through the source code.
    constraints:
        List of optimisation constraints to apply to the optimisation problem
    plot:
        Whether or not to plot
    diag_ops:
        Diagnostic plotting options for Equilibrium
    """

    def __init__(
        self,
        coilset: CoilSet,
        eq: Equilibrium,
        max_currents: npt.ArrayLike | None = None,
        opt_algorithm: AlgorithmType = Algorithm.SLSQP,
        opt_conditions: dict[str, float | int] | None = None,
        opt_parameters: dict[str, float] | None = None,
        constraints: list[UpdateableConstraint] | None = None,
        *,
        plot: bool | None = False,
        diag_ops: EqDiagnosticOptions | None = None,
    ):
        self.coilset = coilset
        self.eq = eq
        self.bounds = self.get_current_bounds(self.coilset, max_currents, self.scale)
        self.opt_conditions = opt_conditions
        self.opt_algorithm = opt_algorithm
        self.opt_parameters = opt_parameters
        self._constraints = [] if constraints is None else constraints

        self.plotting_enabled = plot

        if self.plotting_enabled:
            self.comp_plot = EquilibriumComparisonPlotter(
                equilibrium=self.eq,
                diag_ops=EqDiagnosticOptions(reference_eq=deepcopy(self.eq))
                if diag_ops is None
                else diag_ops,
            )

    def optimise(
        self, x0: npt.NDArray | None = None, *, fixed_coils: bool = True
    ) -> CoilsetOptimiserResult:
        """
        Run the optimisation problem

        Parameters
        ----------
        fixed_coils:
            Whether or not to update to coilset response matrices

        Returns
        -------
        coilset: CoilSet
            Optimised CoilSet
        """
        self.update_magnetic_constraints(I_not_dI=True, fixed_coils=fixed_coils)

        if x0 is None:
            cs_opt_state = self.coilset.get_optimisation_state(current_scale=self.scale)
            x0 = np.clip(cs_opt_state.currents, *self.bounds)
        else:
            x0 = np.clip(x0 / self.scale, *self.bounds)

        objective = CoilCurrentsObjective()

        eq_constraints, ineq_constraints = self._make_numerical_constraints(self.coilset)
        opt_result = optimise(
            algorithm=self.opt_algorithm,
            f_objective=objective.f_objective,
            df_objective=getattr(objective, "df_objective", None),
            x0=x0,
            bounds=self.bounds,
            opt_conditions=self.opt_conditions,
            opt_parameters=self.opt_parameters,
            eq_constraints=eq_constraints,
            ineq_constraints=ineq_constraints,
        )
        opt_currents = opt_result.x
        self.coilset.set_optimisation_state(
            opt_currents=opt_currents,
            current_scale=self.scale,
        )

        if self.plotting_enabled:
            self.comp_plot.update_plot()

        return CoilsetOptimiserResult.from_opt_result(self.coilset, opt_result)
