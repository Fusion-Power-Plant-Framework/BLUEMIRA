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
from typing import List, Tuple

import numpy as np

from bluemira.optimisation import OptimisationProblem
from bluemira.optimisation.typing import ConstraintT


class SimpleOptProblem(OptimisationProblem):
    """
    Simple optimisation that expects solution (1/3, 8/27).

    Minimises sqrt(x2) subject to constraints:
        x2 > 0
        x2 >= (2*x1)^3
        x2 >= (-x1 + 1)^3
    """

    df_call_count: int
    """
    Number of times ``df_objective`` is called.

    Useful for testing we're actually calling the given gradient and not
    an approximation.
    """

    def __init__(self) -> None:
        self.df_call_count = 0

    def objective(self, x: np.ndarray) -> float:
        """Objective to minimise (x - 1)^2 + x + 3."""
        return np.sqrt(x[1])

    def df_objective(self, x: np.ndarray) -> np.ndarray:
        """Gradient of the objective function."""
        self.df_call_count += 1
        return np.array([0.0, 0.5 / np.sqrt(x[1])])

    def ineq_constraints(self) -> List[ConstraintT]:
        return [
            {
                "f_constraint": self.constraint_1,
                "df_constraint": self.df_constraint_1,
                "tolerance": np.full(1, 1e-8),
            },
            {
                "f_constraint": self.constraint_2,
                "df_constraint": self.df_constraint_2,
                "tolerance": np.full(1, 1e-8),
            },
        ]

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([-np.inf, 0]), np.array([np.inf, np.inf])

    def constraint_1(self, x: np.ndarray) -> np.ndarray:
        return (2 * x[0] + 0) ** 3 - x[1]

    def df_constraint_1(self, x: np.ndarray) -> np.ndarray:
        return np.array([3 * 2 * (2 * x[0]) * 2 * x[0], -1.0])

    def constraint_2(self, x: np.ndarray) -> np.ndarray:
        return (-1 * x[0] + 1) ** 3 - x[1]

    def df_constraint_2(self, x: np.ndarray) -> np.ndarray:
        return np.array([3 * -1 * (-1 * x[0] + 1) * (-1 * x[0] + 1), -1.0])


class OptProblemNoGrad(OptimisationProblem):
    def objective(self, x: np.ndarray) -> float:
        """Objective to minimise (x - 1)^2 + x + 3."""
        return np.sqrt(x[1])

    def ineq_constraints(self) -> List[ConstraintT]:
        return [
            {
                "f_constraint": self.constraint_1,
                "tolerance": np.full(1, 1e-8),
            },
            {
                "f_constraint": self.constraint_2,
                "tolerance": np.full(1, 1e-8),
            },
        ]

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([-np.inf, 0]), np.array([np.inf, np.inf])

    def constraint_1(self, x: np.ndarray) -> np.ndarray:
        return (2 * x[0] + 0) ** 3 - x[1]

    def constraint_2(self, x: np.ndarray) -> np.ndarray:
        return (-1 * x[0] + 1) ** 3 - x[1]


class OptProblemEqConstraint(OptimisationProblem):
    """
    Optimisation problem using an equality constraint.

    Maximise x*y*z such that:
        x^2 + y^2 = 4
        x + z = 2

    Example adapted from https://youtu.be/-t7EawoZPn8.
    """

    def objective(self, x: np.ndarray) -> float:
        return -np.prod(x)

    def eq_constraints(self) -> List[ConstraintT]:
        return [
            {
                "f_constraint": self.eq_constraint_1,
                "df_constraint": self.df_eq_constraint_1,
                "tolerance": np.array([1e-8]),
            },
            # no derivative for this constraint, to test approximation
            {"f_constraint": self.eq_constraint_2, "tolerance": np.array([1e-8])},
        ]

    def eq_constraint_1(self, x: np.ndarray) -> np.ndarray:
        """Equality constraint: x^2 + y^2 = 4."""
        return x[0] ** 2 + x[1] ** 2 - 4

    def df_eq_constraint_1(self, x: np.ndarray) -> np.ndarray:
        """Derivative of equality constraint: x^2 + y^2 = 4."""
        return np.array([2 * x[0], 2 * x[1], 0])

    def eq_constraint_2(self, x: np.ndarray) -> np.ndarray:
        """Equality constraint: x + z = 2."""
        return x[0] + x[2] - 2


class TestOptimisationProblem:
    def test_simple_optimisation_returns_correct_result(self):
        op = SimpleOptProblem()
        conditions = {"xtol_rel": 1e-6, "max_eval": 100}
        result = op.optimise(np.array([1, 1]), opt_conditions=conditions)

        np.testing.assert_allclose(result.x, [1 / 3, 8 / 27], rtol=1e-4)
        assert op.df_call_count > 0

    def test_check_constraints_prints_warnings_if_violated(self, caplog):
        op = SimpleOptProblem()

        constraints_ok = op.check_constraints(np.array([20, 30]))

        assert not constraints_ok
        messages = "\n".join(caplog.messages)
        assert all(m in messages for m in ["constraints", "not", "satisfied"])

    def test_check_constraints_no_warnings_given_warn_false(self, caplog):
        op = SimpleOptProblem()

        constraints_ok = op.check_constraints(np.array([20, 30]), warn=False)

        assert not constraints_ok
        assert not caplog.messages

    def test_check_constraints_no_warnings_given_no_violation(self, caplog):
        op = SimpleOptProblem()

        constraints_ok = op.check_constraints(np.array([1 / 3, 8 / 27]))

        assert constraints_ok
        assert not caplog.messages

    def test_opt_problem_with_no_gradient_defined(self):
        # We should still get a good solution, as we should be
        # approximating the gradient automatically.
        op = OptProblemNoGrad()
        conditions = {"xtol_rel": 1e-6, "max_eval": 100}
        result = op.optimise(
            np.array([1, 1]), algorithm="SLSQP", opt_conditions=conditions
        )

        np.testing.assert_allclose(result.x, [1 / 3, 8 / 27], rtol=1e-4)

    def test_opt_problem_with_eq_constraint(self):
        op = OptProblemEqConstraint()
        conditions = {"xtol_rel": 1e-6, "max_eval": 200}
        result = op.optimise(
            np.array([1, 1, 1]), algorithm="SLSQP", opt_conditions=conditions
        )

        x = (np.sqrt(13) - 1) / 3
        y = np.sqrt(4 - x**2)
        z = 2 - x
        np.testing.assert_allclose(result.x, [x, y, z])
