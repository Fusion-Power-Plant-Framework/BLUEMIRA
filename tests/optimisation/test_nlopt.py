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

import nlopt
import numpy as np
import pytest

from bluemira.base.constants import EPS
from bluemira.optimisation._algorithm import Algorithm
from bluemira.optimisation._nlopt import NloptOptimiser
from bluemira.optimisation.error import (
    OptimisationConditionsError,
    OptimisationError,
    OptimisationParametersError,
)
from bluemira.utilities.error import OptVariablesError
from tests.optimisation.tools import rosenbrock

NLOPT_OPT_REF = "bluemira.optimisation._nlopt.optimiser.nlopt.opt"


def no_op(x):  # noqa: ARG001
    # No op for use as a dummy function within tests
    pass


class TestNloptOptimiser:
    @pytest.mark.parametrize("alg", [1, 1.1, [1, 2]])
    def test_TypeError_given_invalid_algorithm_type(self, alg):
        with pytest.raises(TypeError):
            NloptOptimiser(alg, 1)

    @mock.patch(NLOPT_OPT_REF)
    def test_opt_conditions_set_on_nlopt_optimiser(self, nlopt_mock):
        opt_conditions = {
            "ftol_abs": 1,
            "ftol_rel": 2,
            "xtol_abs": 3,
            "xtol_rel": 4,
            "max_eval": 5,
            "max_time": 6,
            "stop_val": 7,
        }

        NloptOptimiser("SLSQP", 5, no_op, opt_conditions=opt_conditions)

        opt_mock = nlopt_mock.return_value
        opt_mock.set_ftol_abs.assert_called_once_with(1)
        opt_mock.set_ftol_rel.assert_called_once_with(2)
        opt_mock.set_xtol_abs.assert_called_once_with(3)
        opt_mock.set_xtol_rel.assert_called_once_with(4)
        opt_mock.set_maxeval.assert_called_once_with(5)
        opt_mock.set_maxtime.assert_called_once_with(6)
        opt_mock.set_stopval.assert_called_once_with(7)

    @pytest.mark.parametrize(
        ("stop_condition", "setter"),
        [
            ("ftol_abs", "set_ftol_abs"),
            ("ftol_rel", "set_ftol_rel"),
            ("xtol_abs", "set_xtol_abs"),
            ("xtol_rel", "set_xtol_rel"),
            ("max_eval", "set_maxeval"),
            ("max_time", "set_maxtime"),
            ("stop_val", "set_stopval"),
        ],
    )
    def test_opt_condition_not_set_if_not_given(self, stop_condition, setter):
        opt_conditions = {
            "ftol_abs": 1,
            "ftol_rel": 2,
            "xtol_abs": 3,
            "xtol_rel": 4,
            "max_eval": 5,
            "max_time": 6,
            "stop_val": 7,
        }
        opt_conditions.pop(stop_condition)

        with mock.patch(NLOPT_OPT_REF) as nlopt_mock:
            NloptOptimiser("SLSQP", 5, no_op, opt_conditions=opt_conditions)

        getattr(nlopt_mock.return_value, setter).assert_not_called()

    def test_OptimisationConditionsError_given_no_stopping_condition_set(self):
        with pytest.raises(OptimisationConditionsError):
            NloptOptimiser("SLSQP", 5, no_op)

    def test_warning_raised_if_max_eval_is_float(self, caplog):
        NloptOptimiser("SLSQP", 5, no_op, opt_conditions={"ftol_abs": 1, "max_eval": 5})
        assert len(caplog.records) == 0
        NloptOptimiser(
            "SLSQP", 5, no_op, opt_conditions={"ftol_abs": 1, "max_eval": 1e6}
        )
        assert len(caplog.records) == 1

    @pytest.mark.parametrize(
        ("string", "enum_value", "nlopt_enum"),
        [
            ("SLSQP", Algorithm.SLSQP, nlopt.LD_SLSQP),
            ("COBYLA", Algorithm.COBYLA, nlopt.LN_COBYLA),
            ("SBPLX", Algorithm.SBPLX, nlopt.LN_SBPLX),
        ],
    )
    def test_algorithm_converted_from_str_to_enum(self, string, enum_value, nlopt_enum):
        with mock.patch(NLOPT_OPT_REF) as nlopt_mock:
            opt = NloptOptimiser(string, 5, no_op, opt_conditions={"max_eval": 200})

        assert opt.algorithm == enum_value
        nlopt_mock.assert_called_once_with(nlopt_enum, 5)

    def test_direct_l_algorithm_can_be_selected_with_hyphened_string(self):
        with mock.patch(NLOPT_OPT_REF) as nlopt_mock:
            opt = NloptOptimiser("DIRECT-L", 5, no_op, opt_conditions={"max_eval": 200})

        assert opt.algorithm == Algorithm.DIRECT_L
        nlopt_mock.assert_called_once_with(nlopt.GN_DIRECT_L, 5)

    @mock.patch(NLOPT_OPT_REF)
    def test_algorithm_parameters_set_if_they_exist(self, nlopt_mock):
        opt_mock = nlopt_mock.return_value
        opt_mock.has_param.return_value = True

        NloptOptimiser(
            "SLSQP",
            5,
            no_op,
            opt_conditions={"max_eval": 200},
            opt_parameters={"param1": 1, "param2": 2},
        )

        opt_mock.set_param.call_args_list = [
            mock.call("param1", 1),
            mock.call("param2", 2),
        ]

    @mock.patch(NLOPT_OPT_REF)
    def test_initial_step_parameter_set(self, nlopt_mock):
        opt_mock = nlopt_mock.return_value
        opt_mock.has_param.return_value = False

        NloptOptimiser(
            "SLSQP",
            2,
            no_op,
            opt_conditions={"max_eval": 200},
            opt_parameters={"initial_step": [1, 2]},
        )

        opt_mock.set_param.assert_not_called()
        opt_mock.set_initial_step.assert_called_once_with([1, 2])

    @mock.patch(NLOPT_OPT_REF)
    def test_OptimisationParametersError_given_unrecognised_param(self, nlopt_mock):
        nlopt_mock.return_value.has_param.return_value = False

        with pytest.raises(OptimisationParametersError):
            NloptOptimiser(
                "SLSQP",
                2,
                no_op,
                opt_conditions={"max_eval": 200},
                opt_parameters={"not_a_param": 1},
            )

    @mock.patch(NLOPT_OPT_REF)
    def test_minimising_objective_function_set_on_init(self, nlopt_mock):
        NloptOptimiser(
            "SLSQP",
            2,
            lambda _: 5,
            df_objective=lambda _: np.array([1]),
            opt_conditions={"max_eval": 200},
            opt_parameters={"not_a_param": 1},
        )

        assert nlopt_mock.return_value.set_min_objective.call_count == 1
        # Retrieve the callable from the mock as we wrap the objective function in a
        # new callable, which means we can't directly assert the function that's set.
        # The output of the function should not change though, so we can assert on that.
        set_func = nlopt_mock.return_value.set_min_objective.call_args[0][0]
        assert set_func(1, np.array([2])) == 5

    def test_minimum_found_on_rosenbrock_no_bounds(self):
        opt = NloptOptimiser("SLSQP", 2, rosenbrock, opt_conditions={"ftol_rel": 1e-6})

        result = opt.optimise()

        np.testing.assert_allclose(result.x, [1, 1])

    def test_minimum_found_on_rosenbrock_within_bounds(self):
        opt = NloptOptimiser(
            "SLSQP",
            2,
            rosenbrock,
            opt_conditions={"ftol_rel": 1e-6, "max_eval": 200},
        )
        opt.set_lower_bounds(np.array([1.1, 0.5]))
        opt.set_upper_bounds(np.array([2, 0.9]))

        result = opt.optimise()

        np.testing.assert_allclose(result.x, [1.1, 0.9])

    @pytest.mark.parametrize(
        "alg",
        [
            Algorithm.SBPLX,
            Algorithm.MMA,
            Algorithm.BFGS,
            Algorithm.DIRECT,
            Algorithm.DIRECT_L,
            Algorithm.CRS,
        ],
    )
    def test_OptimisationError_adding_eq_constraint_on_unsupported_algorithm(self, alg):
        with mock.patch(NLOPT_OPT_REF) as nlopt_mock:
            opt = NloptOptimiser(alg, 2, no_op, opt_conditions={"max_eval": 200})

        with pytest.raises(OptimisationError):
            opt.add_eq_constraint(no_op, np.zeros(2))
        nlopt_mock.return_value.add_equality_mconstraint.assert_not_called()

    @pytest.mark.parametrize("alg", [Algorithm.SLSQP, Algorithm.COBYLA, Algorithm.ISRES])
    def test_add_eq_constraint_sets_constraint(self, alg):
        opt = NloptOptimiser(alg, 1, no_op, opt_conditions={"max_eval": 200})

        with mock.patch(f"{NLOPT_OPT_REF}.add_equality_mconstraint") as add_eq_mock:
            opt.add_eq_constraint(lambda _: -1, 1, lambda _: 2)

        assert add_eq_mock.call_count == 1
        # Retrieve the constraint from the mock, and call it to check it
        # returns what we've told it to
        result = np.zeros(1)
        grad = np.zeros(1)
        constraint_func = add_eq_mock.call_args[0][0]
        constraint_func(result, np.zeros(1), grad)
        np.testing.assert_equal(result, [-1])
        np.testing.assert_equal(grad, [2])
        np.testing.assert_allclose(add_eq_mock.call_args[0][1], np.array([1, 1]))

    @pytest.mark.parametrize(
        "alg",
        [
            Algorithm.SBPLX,
            Algorithm.MMA,
            Algorithm.BFGS,
            Algorithm.DIRECT,
            Algorithm.DIRECT_L,
            Algorithm.CRS,
        ],
    )
    def test_OptimisationError_adding_ineq_constraint_on_unsupported_algorithm(
        self, alg
    ):
        with mock.patch(NLOPT_OPT_REF) as nlopt_mock:
            opt = NloptOptimiser(alg, 2, no_op, opt_conditions={"max_eval": 200})

        with pytest.raises(OptimisationError):
            opt.add_eq_constraint(no_op, np.zeros(2))
        nlopt_mock.return_value.add_inequality_mconstraint.assert_not_called()

    @pytest.mark.parametrize("alg", [Algorithm.SLSQP, Algorithm.COBYLA, Algorithm.ISRES])
    def test_add_ineq_constraint_sets_constraint(self, alg):
        opt = NloptOptimiser(alg, 1, lambda _: 1, opt_conditions={"max_eval": 200})

        with mock.patch(f"{NLOPT_OPT_REF}.add_inequality_mconstraint") as add_ineq_mock:
            opt.add_ineq_constraint(lambda _: -1, 1, lambda _: 2)

        assert add_ineq_mock.call_count == 1
        # Retrieve the constraint from the mock, and call it to check it
        # returns what we've told it to
        result = np.zeros(1)
        grad = np.zeros(1)
        constraint_func = add_ineq_mock.call_args[0][0]
        constraint_func(result, np.zeros(1), grad)
        np.testing.assert_equal(result, [-1])
        np.testing.assert_equal(grad, [2])
        np.testing.assert_allclose(add_ineq_mock.call_args[0][1], np.array([1, 1]))

    @pytest.mark.parametrize(
        "cond",
        ["ftol_abs", "ftol_rel", "xtol_abs", "xtol_rel"],
    )
    @mock.patch("bluemira.optimisation._nlopt.conditions.bluemira_warn")
    def test_warning_given_stopping_condition_lt_eps(self, bm_warn, cond):
        NloptOptimiser("SLSQP", 1, no_op, opt_conditions={cond: EPS / 1.1})
        bm_warn.assert_called_once()

    @pytest.mark.parametrize(
        "cond",
        ["ftol_abs", "ftol_rel", "xtol_abs", "xtol_rel"],
    )
    @mock.patch("bluemira.optimisation._nlopt.conditions.bluemira_warn")
    def test_no_warning_given_stopping_condition_gt_eps(self, bm_warn, cond):
        NloptOptimiser("SLSQP", 1, no_op, opt_conditions={cond: EPS * 1.1})
        assert bm_warn.call_count == 0

    @pytest.mark.parametrize("t", ["eq", "ineq"])
    def test_OptimisationError_adding_constraint_to_unsupported_algorithm(self, t):
        opt = NloptOptimiser("BFGS", 2, no_op, opt_conditions={"max_eval": 200})

        with pytest.raises(OptimisationError):
            getattr(opt, f"add_{t}_constraint")(no_op, np.array([1e-4, 1e-4]))

    @pytest.mark.parametrize("err", [nlopt.RoundoffLimited, OptVariablesError])
    def test_warning_and_prev_iter_result_given_recoverable_error(self, caplog, err):
        # This is a bit of tricky one to test, so this is also a little
        # bit hacky, sorry!
        # Run a deterministic optimisation once, keeping the history.
        # Then run that optimisation again, but throw a round-off error
        # a set no. of iterations in. Then check, in the history, that
        # we return the parameterisation from the iteration previous to
        # the one we threw the round-off error in.
        def objective(x):
            return -np.sum(x)

        class ErroringObjective:
            def __init__(self, error_on_iter: int) -> None:
                self.iter_num = 0
                self.error_on_iter = error_on_iter

            def __call__(self, x):
                self.iter_num += 1
                if self.iter_num == self.error_on_iter:
                    raise err
                return objective(x)

        # Run the first optimisation
        opt = NloptOptimiser(
            "COBYLA", 2, objective, opt_conditions={"max_eval": 5}, keep_history=True
        )
        hist = opt.optimise(np.array([0, 0])).history

        # Now run it again, but throw an error on iteration 4 of the
        # optimisation loop
        error_on = 4
        erroring_objective = ErroringObjective(error_on)
        err_opt = NloptOptimiser(
            "COBYLA", 2, erroring_objective, opt_conditions={"max_eval": 5}
        )
        err_res = err_opt.optimise(np.array([0, 0]))

        assert len(caplog.messages) >= 1
        np.testing.assert_allclose(err_res.x, hist[error_on - 1][0])

    @pytest.mark.parametrize("bad_alg", [0, ["SLSQP"]])
    def test_TypeError_setting_alg_with_invalid_type(self, bad_alg):
        with pytest.raises(TypeError):
            NloptOptimiser(bad_alg, 2, no_op)

    def test_ValueError_setting_lower_bounds_with_wrong_dims(self):
        opt = NloptOptimiser("SLSQP", 2, no_op, opt_conditions={"max_eval": 200})

        with pytest.raises(ValueError):  # noqa: PT011
            opt.set_lower_bounds(np.array([1, 2, 3]))

    def test_ValueError_setting_upper_bounds_with_wrong_dims(self):
        opt = NloptOptimiser("SLSQP", 4, no_op, opt_conditions={"max_eval": 200})

        with pytest.raises(ValueError):  # noqa: PT011
            opt.set_upper_bounds(np.array([[1, 2], [3, 4]]))

    @pytest.mark.parametrize(
        "fixture",
        [
            ((np.full(4, -np.inf), np.full(4, np.inf)), [0, 0, 0, 0]),
            (
                ([-np.inf, -50, 0, 50], [0, 0, 50, np.inf]),
                [
                    -np.finfo(np.float64).max / 2,
                    -25,
                    25,
                    np.finfo(np.float64).max / 2 + 25,
                ],
            ),
        ],
    )
    def test_initial_guess_derived_from_bounds_if_not_given(self, fixture):
        bounds, x0 = fixture
        opt = NloptOptimiser(
            "SLSQP", 4, no_op, df_objective=no_op, opt_conditions={"max_eval": 200}
        )
        opt.set_lower_bounds(bounds[0])
        opt.set_upper_bounds(bounds[1])

        with mock.patch(f"{NLOPT_OPT_REF}.optimize") as optimize_mock:
            opt.optimise()

        assert optimize_mock.call_count == 1
        call_args = optimize_mock.call_args[0]
        assert len(call_args) == 1
        np.testing.assert_allclose(call_args[0], x0)

    def test_set_lower_bounds_sets_bounds(self):
        bounds = np.array([0, 1, 2, 3])
        opt = NloptOptimiser(
            "SLSQP", 4, no_op, df_objective=no_op, opt_conditions={"max_eval": 200}
        )

        opt.set_lower_bounds(bounds)

        np.testing.assert_allclose(opt.lower_bounds, bounds)

    def test_set_upper_bounds_sets_bounds(self):
        bounds = np.array([0, 1, 2, 3])
        opt = NloptOptimiser(
            "SLSQP", 4, no_op, df_objective=no_op, opt_conditions={"max_eval": 200}
        )

        opt.set_upper_bounds(bounds)

        np.testing.assert_allclose(opt.upper_bounds, bounds)

    @pytest.mark.parametrize(
        ("nlopt_err", "msg"),
        [
            (nlopt.MAXEVAL_REACHED, ["succeeded", "maximum", "evaluations"]),
            (nlopt.MAXTIME_REACHED, ["succeeded", "maximum", "duration"]),
            (nlopt.ROUNDOFF_LIMITED, ["round-off", "last", "parameterisation"]),
            (nlopt.FAILURE, ["failed"]),
            (nlopt.INVALID_ARGS, ["invalid", "arguments"]),
            (nlopt.OUT_OF_MEMORY, ["out", "of", "memory"]),
            (nlopt.FORCED_STOP, ["forced", "stop"]),
        ],
    )
    @pytest.mark.parametrize(
        ("opt_err", "rethrow_err"),
        [(RuntimeError, OptimisationError), (KeyboardInterrupt, KeyboardInterrupt)],
    )
    @mock.patch("bluemira.optimisation._nlopt.optimiser.nlopt.opt.last_optimize_result")
    def test_warning_given_optimise_exception(
        self, lor_mock, caplog, opt_err, rethrow_err, nlopt_err, msg
    ):
        def objective(_):
            raise opt_err

        lor_mock.return_value = nlopt_err
        opt = NloptOptimiser("COBYLA", 2, objective, opt_conditions={"max_eval": 5})

        with pytest.raises(rethrow_err):
            opt.optimise(np.zeros(2))

        warning_msgs = caplog.messages
        assert len(warning_msgs) == 1
        assert all(m in warning_msgs[0] for m in msg)
