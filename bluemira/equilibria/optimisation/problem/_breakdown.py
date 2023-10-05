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
import abc
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.equilibrium import Breakdown
from bluemira.equilibria.optimisation.constraints import (
    FieldConstraints,
    UpdateableConstraint,
)
from bluemira.equilibria.optimisation.objectives import MaximiseFluxObjective
from bluemira.equilibria.optimisation.problem.base import (
    CoilsetOptimisationProblem,
    CoilsetOptimiserResult,
)
from bluemira.optimisation import Algorithm, AlgorithmType, optimise


class BreakdownZoneStrategy(abc.ABC):
    """
    Abstract base class for the definition of a breakdown zone strategy.

    Parameters
    ----------
    R_0:
        Major radius of the reference plasma
    A:
        Aspect ratio of the reference plasma
    tk_sol:
        Thickness of the scrape-off layer
    """

    def __init__(self, R_0, A, tk_sol, **kwargs):  # noqa: ARG002
        self.R_0 = R_0
        self.A = A
        self.tk_sol = tk_sol

    @abc.abstractproperty
    def breakdown_point(self) -> Tuple[float, float]:
        """
        The location of the breakdown point.

        Returns
        -------
        x_c:
            Radial coordinate of the breakdown point
        z_c:
            Vertical coordinate of the breakdown point
        """

    @abc.abstractproperty
    def breakdown_radius(self) -> float:
        """
        The radius of the breakdown zone.
        """

    @abc.abstractmethod
    def calculate_zone_points(self, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the discretised set of points representing the breakdown zone.
        """


class CircularZoneStrategy(BreakdownZoneStrategy):
    """
    Circular breakdown zone strategy.
    """

    def calculate_zone_points(self, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the discretised set of points representing the breakdown zone.
        """
        x_c, z_c = self.breakdown_point
        r_c = self.breakdown_radius
        theta = np.linspace(0, 2 * np.pi, n_points - 1, endpoint=False)
        x = x_c + r_c * np.cos(theta)
        z = z_c + r_c * np.sin(theta)
        x = np.append(x, x_c)
        z = np.append(z, z_c)
        return x, z


class InboardBreakdownZoneStrategy(CircularZoneStrategy):
    """
    Inboard breakdown zone strategy.
    """

    @property
    def breakdown_point(self) -> Tuple[float, float]:
        """
        The location of the breakdown point.

        Returns
        -------
        x_c:
            Radial coordinate of the breakdown point
        z_c:
            Vertical coordinate of the breakdown point
        """
        r_c = self.breakdown_radius
        x_c = self.R_0 - self.R_0 / self.A - self.tk_sol + r_c
        z_c = 0.0
        return x_c, z_c

    @property
    def breakdown_radius(self) -> float:
        """
        The radius of the breakdown zone.
        """
        return 0.5 * self.R_0 / self.A


class OutboardBreakdownZoneStrategy(CircularZoneStrategy):
    """
    Outboard breakdown zone strategy.
    """

    @property
    def breakdown_point(self) -> Tuple[float, float]:
        """
        The location of the breakdown point.

        Returns
        -------
        x_c:
            Radial coordinate of the breakdown point
        z_c:
            Vertical coordinate of the breakdown point
        """
        r_c = self.breakdown_radius
        x_c = self.R_0 + self.R_0 / self.A + self.tk_sol - r_c
        z_c = 0.0
        return x_c, z_c

    @property
    def breakdown_radius(self) -> float:
        """
        The radius of the breakdown zone.
        """
        return 0.7 * self.R_0 / self.A


class InputBreakdownZoneStrategy(CircularZoneStrategy):
    """
    User input breakdown zone strategy.
    """

    def __init__(self, x_c, z_c, r_c):
        self.x_c = x_c
        self.z_c = z_c
        self.r_c = r_c

    @property
    def breakdown_point(self) -> Tuple[float, float]:
        """
        The location of the breakdown point.

        Returns
        -------
        x_c:
            Radial coordinate of the breakdown point
        z_c:
            Vertical coordinate of the breakdown point
        """
        return self.x_c, self.z_c

    @property
    def breakdown_radius(self) -> float:
        """
        The radius of the breakdown zone.
        """
        return self.r_c


class BreakdownCOP(CoilsetOptimisationProblem):
    """
    Coilset optimisation problem for the premagnetisation / breakdown phase.
    """

    def __init__(
        self,
        coilset: CoilSet,
        breakdown: Breakdown,
        breakdown_strategy: BreakdownZoneStrategy,
        B_stray_max: float,
        B_stray_con_tol: float,
        n_B_stray_points: int,
        max_currents: npt.ArrayLike,
        opt_algorithm: AlgorithmType = Algorithm.SLSQP,
        opt_conditions: Optional[Dict[str, Union[float, int]]] = None,
        constraints: Optional[List[UpdateableConstraint]] = None,
    ):
        self.coilset = coilset
        self.eq = breakdown
        self.opt_algorithm = opt_algorithm
        self.opt_conditions = opt_conditions

        self._args = {
            "c_psi_mat": np.array(
                coilset.psi_response(*breakdown_strategy.breakdown_point)
            ),
            "scale": self.scale,
        }
        x_zone, z_zone = breakdown_strategy.calculate_zone_points(n_B_stray_points)
        stray_field_cons = FieldConstraints(
            x_zone, z_zone, B_max=B_stray_max, tolerance=B_stray_con_tol
        )

        self._constraints = constraints
        if self._constraints is not None:
            self._constraints.append(stray_field_cons)
        else:
            self._constraints = [stray_field_cons]

        max_currents = np.atleast_1d(max_currents)
        self.bounds = (-max_currents / self.scale, max_currents / self.scale)

    def optimise(self, x0=None, fixed_coils=True):  # noqa: ARG002
        """
        Solve the optimisation problem.
        """
        self.update_magnetic_constraints(I_not_dI=True, fixed_coils=fixed_coils)

        initial_state, n_states = self.read_coilset_state(self.coilset, self.scale)
        _, _, initial_currents = np.array_split(initial_state, n_states)
        initial_currents = np.clip(initial_currents, *self.bounds)

        objective = MaximiseFluxObjective(**self._args)
        eq_constraints, ineq_constraints = self._make_numerical_constraints()
        opt_result = optimise(
            f_objective=objective.f_objective,
            df_objective=objective.df_objective,
            x0=initial_currents,
            algorithm=self.opt_algorithm,
            opt_conditions=self.opt_conditions,
            bounds=self.bounds,
            eq_constraints=eq_constraints,
            ineq_constraints=ineq_constraints,
        )
        currents = opt_result.x
        self.coilset.get_control_coils().current = currents * self.scale
        return CoilsetOptimiserResult.from_opt_result(self.coilset, opt_result)
