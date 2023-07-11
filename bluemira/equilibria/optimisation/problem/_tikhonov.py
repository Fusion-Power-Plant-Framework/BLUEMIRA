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

from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt

from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.opt_constraints import MagneticConstraintSet
from bluemira.equilibria.optimisation.constraints import ConstraintFunction
from bluemira.equilibria.optimisation.objectives import RegularisedLsqObjective
from bluemira.equilibria.optimisation.problem.base import (
    CoilsetOptimisationProblem,
    CoilsetOptimiserResult,
)
from bluemira.optimisation import optimise


class TikhonovCurrentCOP(CoilsetOptimisationProblem):
    """
    Coilset OptimisationProblem for coil currents subject to maximum current bounds.

    Coilset currents optimised using objectives.regularised_lsq_objective as
    objective function.

    Parameters
    ----------
    coilset:
        Coilset to optimise.
    eq:
        Equilibrium object used to update magnetic field targets.
    targets:
        Set of magnetic field targets to use in objective function.
    gamma:
        Tikhonov regularisation parameter in units of [A⁻¹].
    max_currents:
        Maximum allowed current for each independent coil current in coilset [A].
        If specified as a float, the float will set the maximum allowed current
        for all coils.
    optimiser:
        Optimiser object to use for constrained optimisation.
    constraints:
        Optional list of OptimisationConstraint objects storing
        information about constraints that must be satisfied
        during the coilset optimisation, to be provided to the
        optimiser.
    """

    def __init__(
        self,
        coilset: CoilSet,
        eq: Equilibrium,
        targets: MagneticConstraintSet,
        gamma: float,
        opt_algorithm: str = "SLSQP",
        opt_conditions: Dict[str, Union[float, int]] = {
            "xtol_rel": 1e-4,
            "xtol_abs": 1e-4,
            "ftol_rel": 1e-4,
            "ftol_abs": 1e-4,
            "max_eval": 100,
        },
        opt_parameters: Dict[str, Any] = {"initial_step": 0.03},
        max_currents: Optional[npt.ArrayLike] = None,
        constraints: Optional[List[ConstraintFunction]] = None,
    ):
        self.coilset = coilset
        self.eq = eq
        self.targets = targets
        self.gamma = gamma
        self.bounds = self.get_current_bounds(self.coilset, max_currents, self.scale)
        self.opt_algorithm = opt_algorithm
        self.opt_conditions = opt_conditions
        self.opt_parameters = opt_parameters
        self._constraints = constraints

    def optimise(self, x0=None, fixed_coils=True) -> CoilsetOptimiserResult:
        """
        Solve the optimisation problem

        Parameters
        ----------
        fixed_coils: True
            Whether or not to update to coilset response matrices

        Returns
        -------
        coilset: CoilSet
            Optimised CoilSet
        """
        # Scale the control matrix and magnetic field targets vector by weights.
        self.targets(self.eq, I_not_dI=True, fixed_coils=fixed_coils)
        _, a_mat, b_vec = self.targets.get_weighted_arrays()
        self.update_magnetic_constraints(I_not_dI=True, fixed_coils=fixed_coils)

        if x0 is None:
            initial_state, n_states = self.read_coilset_state(self.coilset, self.scale)
            _, _, initial_currents = np.array_split(initial_state, n_states)
            x0 = np.clip(initial_currents, *self.bounds)

        objective = RegularisedLsqObjective(
            scale=self.scale,
            a_mat=a_mat,
            b_vec=b_vec,
            gamma=self.gamma,
        )
        opt_result = optimise(
            f_objective=objective.f_objective,
            df_objective=getattr(objective, "df_objective", None),
            x0=x0,
            bounds=self.bounds,
            opt_conditions=self.opt_conditions,
            algorithm=self.opt_algorithm,
            opt_parameters=self.opt_parameters,
        )
        currents = opt_result.x
        self.coilset.get_control_coils().current = currents * self.scale
        return CoilsetOptimiserResult.from_opt_result(self.coilset, opt_result)
