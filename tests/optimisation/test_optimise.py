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
from unittest import mock

import numpy as np
import pytest

from bluemira.optimisation import Algorithm, optimise
from tests.optimisation.tools import d_rosenbrock, rosenbrock


class TestOptimise:
    @pytest.mark.parametrize(
        "algorithm", ["SLSQP", Algorithm.SBPLX, "MMA", Algorithm.BFGS]
    )
    def test_minimum_found_on_rosenbrock_no_bounds(self, algorithm):
        result = optimise(
            rosenbrock,
            algorithm=algorithm,
            df_objective=d_rosenbrock,
            dimensions=2,
            opt_conditions={"ftol_rel": 1e-6},
        )

        np.testing.assert_allclose(result.x, [1, 1])

    def test_history_recorded_given_keep_history_is_True(self):
        result = optimise(
            rosenbrock,
            algorithm="SLSQP",
            df_objective=d_rosenbrock,
            dimensions=2,
            opt_conditions={"ftol_rel": 1e-6},
            keep_history=True,
        )

        assert len(result.history) == result.n_evals
        # Just check that the first and second x values are not equal.
        # We just want to verify we're not just adding the same array
        # every time
        assert not np.allclose(result.history[0][0], result.history[1][0])
        assert not np.allclose(result.history[0][1], result.history[1][1])

    @pytest.mark.parametrize(
        "algorithm",
        [
            Algorithm.SLSQP,
            "SBPLX",
            Algorithm.MMA,
            "BFGS",
            "DIRECT",
            Algorithm.DIRECT_L,
            "ISRES",
        ],
    )
    def test_grad_based_algs_find_min_on_rosenbrock_within_bounds(self, algorithm):
        result = optimise(
            rosenbrock,
            algorithm=algorithm,
            df_objective=d_rosenbrock,
            dimensions=2,
            opt_conditions={"ftol_rel": 1e-8, "max_eval": 5000},
            bounds=(np.array([0.5, 1.05]), np.array([0.95, 2])),
        )

        np.testing.assert_allclose(
            result.x,
            [0.95, 1.05],
            atol=0,
            rtol=1e-3,
            err_msg=f"after {result.n_evals} evals",
        )

    def test_ValueError_given_invalid_algorithm(self):
        with pytest.raises(ValueError):  # noqa: PT011
            optimise(lambda: None, algorithm="NOT_AN_ALG")

    def test_ValueError_given_x0_and_dimensions_not_consistent(self):
        with pytest.raises(ValueError):  # noqa: PT011
            optimise(
                rosenbrock,
                dimensions=3,
                x0=np.array([1, 1]),
                df_objective=d_rosenbrock,
                algorithm="SLSQP",
                opt_conditions={"ftol_rel": 1e-8, "max_eval": 5000},
                bounds=(np.array([0.5, 1.05]), np.array([0.95, 2])),
            )

    @pytest.mark.parametrize("alg", ["SLSQP", "COBYLA"])
    def test_nonlinear_constrain_optimisation(self, alg):
        """
        Test an optimisation with nonlinear constraints.

        See 'examples/nonlinearly_constrained_problem.py' for details of
        the problem.
        """
        result = optimise(
            NonLinearExample.f_objective,
            x0=np.array([1, 1]),
            algorithm=alg,
            df_objective=NonLinearExample.df_objective,
            opt_conditions={"xtol_rel": 1e-8, "max_eval": 1000},
            bounds=(np.array([-np.inf, 0]), np.array([np.inf, np.inf])),
            ineq_constraints=[
                {
                    "f_constraint": lambda x: NonLinearExample.f_constraint(x, 2, 0),
                    # Exclude df_constraint to test approx_derivative is doing work
                    "tolerance": np.array([5e-6]),
                },
                {
                    "f_constraint": lambda x: NonLinearExample.f_constraint(x, -1, 1),
                    "df_constraint": lambda x: NonLinearExample.df_constraint(x, -1, 1),
                    "tolerance": np.array([5e-6]),
                },
            ],
        )

        np.testing.assert_allclose(result.x, NonLinearExample.EXPECTED_RESULT, atol=1e-4)
        assert result.constraints_satisfied is True

    def test_scalar_lower_bounds_are_expanded(self):
        result = optimise(
            np.sum,
            x0=np.array([5, 5]),
            algorithm="COBYLA",
            bounds=(1, np.array([10, 20])),
            opt_conditions={"ftol_rel": 1e-6, "max_eval": 100},
        )

        np.testing.assert_allclose(result.x, [1, 1])

    def test_scalar_upper_bounds_are_expanded(self):
        result = optimise(
            lambda x: -np.sum(x),
            x0=np.array([5, 5]),
            algorithm="COBYLA",
            bounds=(1, 10),
            opt_conditions={"ftol_rel": 1e-6, "max_eval": 100},
        )

        np.testing.assert_allclose(result.x, [10, 10])

    @pytest.mark.parametrize("bad_bounds", [(None, 1), (1, 2, 3), (), 10])
    def test_error_given_invalid_bounds(self, bad_bounds):
        with pytest.raises((ValueError, TypeError)):
            optimise(
                np.sum,
                x0=np.array([5, 5]),
                algorithm="COBYLA",
                bounds=bad_bounds,
                opt_conditions={"ftol_rel": 1e-6, "max_eval": 1},
            )

    @pytest.mark.parametrize("constraint_type", ["ineq", "eq"])
    @mock.patch("bluemira.optimisation._optimise.bluemira_warn")
    def test_warning_given_constraints_not_satisfied(
        self, bm_warn_mock, constraint_type
    ):
        constraint = {
            "f_constraint": lambda x: NonLinearExample.f_constraint(x, 2, 0),
            "tolerance": np.array([1e-8]),
        }

        result = optimise(
            NonLinearExample.f_objective,
            x0=np.array([1, 1]),
            opt_conditions={"max_eval": 1},
            **{f"{constraint_type}_constraints": [constraint]},
        )

        bm_warn_mock.assert_called_once()
        message = bm_warn_mock.call_args[0][0].lower()
        comp_str = "!<" if constraint_type == "ineq" else "!="
        msg_strs = ["constraints", "not", "satisfied", constraint_type, comp_str]
        assert all(m in message for m in msg_strs)
        assert result.constraints_satisfied is False

    @pytest.mark.parametrize("constraint_type", ["ineq", "eq"])
    @mock.patch("bluemira.optimisation._optimise.bluemira_warn")
    def test_no_warning_given_constraints_not_satisfied_and_warn_False(
        self, bm_warn_mock, constraint_type
    ):
        constraint = {
            "f_constraint": lambda x: NonLinearExample.f_constraint(x, 2, 0),
            "tolerance": np.array([1e-8]),
        }

        result = optimise(
            NonLinearExample.f_objective,
            x0=np.array([1, 1]),
            opt_conditions={"max_eval": 1},
            **{f"{constraint_type}_constraints": [constraint]},
            check_constraints_warn=False,
        )

        assert bm_warn_mock.call_count == 0
        assert result.constraints_satisfied is False

    @mock.patch("bluemira.optimisation._optimise.bluemira_warn")
    def test_no_constraint_warning_given_check_constraints_False(self, bm_warn_mock):
        result = optimise(
            NonLinearExample.f_objective,
            x0=np.array([1, 1]),
            opt_conditions={"max_eval": 1},
            ineq_constraints=[
                {
                    "f_constraint": lambda x: NonLinearExample.f_constraint(x, 2, 0),
                    "tolerance": np.array([1e-8]),
                },
                {
                    "f_constraint": lambda x: NonLinearExample.f_constraint(x, -1, 1),
                    "tolerance": np.array([1e-8]),
                },
            ],
            check_constraints=False,
        )

        assert result.constraints_satisfied is None
        assert bm_warn_mock.call_count == 0

    @mock.patch("bluemira.optimisation._optimise.bluemira_warn")
    def test_no_warnings_given_constraints_satisfied(self, bm_warn_mock):
        def f_objective(x):
            return np.sqrt(x[1])

        def f_ineq_constraint(x):
            return np.sum(x) - 5

        def f_eq_constraint(x):
            return abs(x[0] - x[1])

        result = optimise(
            f_objective,
            x0=np.array([1, 1]),
            opt_conditions={"max_eval": 1},
            eq_constraints=[
                {"f_constraint": f_eq_constraint, "tolerance": np.array([1e-8])}
            ],
            ineq_constraints=[
                {"f_constraint": f_ineq_constraint, "tolerance": np.array([1e-8])}
            ],
            check_constraints=True,
        )

        assert result.constraints_satisfied is True
        assert bm_warn_mock.call_count == 0


class NonLinearExample:
    """
    A basic optimisation example with non-linear inequality constraints.

    See 'examples/nonlinearly_constrained_problem.py' for details of
    the problem.
    """

    EXPECTED_RESULT = np.array([1 / 3, 8 / 27])

    @staticmethod
    def f_objective(x):
        return np.sqrt(x[1])

    @staticmethod
    def df_objective(x):
        return np.array([0.0, 0.5 / np.sqrt(x[1])])

    @staticmethod
    def f_constraint(x, a, b):
        return (a * x[0] + b) ** 3 - x[1]

    @staticmethod
    def df_constraint(x, a, b):
        return np.array([3 * a * (a * x[0] + b) * (a * x[0] + b), -1.0])
