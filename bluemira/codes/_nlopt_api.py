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
Thin wrapper API interface to optimisation library (NLOpt)
"""

import functools
from typing import Any, Callable, Dict, Optional

import nlopt
import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.utilities.error import (
    ExternalOptError,
    OptUtilitiesError,
    OptVariablesError,
)

EPS = np.finfo(np.float64).eps

NLOPT_ALG_MAPPING = {
    "SLSQP": nlopt.LD_SLSQP,
    "COBYLA": nlopt.LN_COBYLA,
    "SBPLX": nlopt.LN_SBPLX,
    "MMA": nlopt.LD_MMA,
    "BFGS": nlopt.LD_LBFGS,
    "DIRECT": nlopt.GN_DIRECT,
    "DIRECT-L": nlopt.GN_DIRECT_L,
    "CRS": nlopt.GN_CRS2_LM,
    "ISRES": nlopt.GN_ISRES,
}

GRAD_ALGS = ["SLSQP", "MMA", "BGFS"]

INEQ_CON_ALGS = ["SLSQP", "COBYLA", "ISRES"]

EQ_CON_ALGS = ["SLSQP", "COBYLA", "ISRES"]

TERMINATION_KEYS = [
    "ftol_abs",
    "ftol_rel",
    "xtol_abs",
    "xtol_rel",
    "max_eval",
    "max_time",
    "stop_val",
]


def process_NLOPT_conditions(  # noqa: N802
    opt_conditions: Dict[str, float]
) -> Dict[str, float]:
    """
    Process NLopt termination conditions. Checks for negative or 0 values on some
    conditions (which mean they are inactive), and warns if you are doing weird stuff.

    Parameters
    ----------
    opt_conditions:
        Dictionary of termination condition keys and values

    Returns
    -------
    Dictionary of processed termination condition keys and values

    Raises
    ------
    OptUtilitiesError
        If no valid termination conditions are specified
    """
    conditions = {}
    for k, v in opt_conditions.items():
        if k not in TERMINATION_KEYS:
            bluemira_warn(f"Unrecognised termination condition: {k}")

        # Negative or 0 conditions result in inactive NLopt termination conditions, for
        # the most part.
        if k == "stop_val":
            conditions[k] = v

        elif v > 0:
            if k in ["ftol_abs", "ftol_res", "xtol_abs", "xtol_res"] and v < EPS:
                bluemira_warn(
                    "You are setting an optimisation termination condition to below"
                    " machine precision. Don't.."
                )

            conditions[k] = v

    if not conditions:
        raise OptUtilitiesError(
            "You must specify at least one termination criterion for the optimisation"
            " algorithm."
        )
    return conditions


def process_NLOPT_result(opt: nlopt.opt):  # noqa: N802
    """
    Handle a NLopt optimiser and check results.

    Parameters
    ----------
    opt:
        The optimiser to check
    """
    result = opt.last_optimize_result()

    if result == nlopt.MAXEVAL_REACHED:
        bluemira_warn(
            "\nNLopt Optimiser succeeded but stopped at the maximum number of"
            " evaulations.\n"
        )
    elif result == nlopt.MAXTIME_REACHED:
        bluemira_warn(
            "\nNLopt Optimiser succeeded but stopped at the maximum duration.\n"
        )
    elif result == nlopt.ROUNDOFF_LIMITED:
        bluemira_warn(
            "\nNLopt Optimiser was halted due to round-off errors. A useful result was"
            " probably found...\n"
        )
    elif result == nlopt.FAILURE:
        bluemira_warn("\nNLopt Optimiser failed real hard...\n")
    elif result == nlopt.INVALID_ARGS:
        bluemira_warn("\nNLopt Optimiser failed because of invalid arguments.\n")
    elif result == nlopt.OUT_OF_MEMORY:
        bluemira_warn("\nNLopt Optimiser failed because it ran out of memory.\n")
    elif result == nlopt.FORCED_STOP:
        bluemira_warn("\nNLopt Optimiser failed because of a forced stop.\n")


class _NLOPTObjectiveFunction:
    """
    Wrapper to store x-vector in case of RoundOffLimited errors.
    """

    def __init__(self, func: Callable[[Any], float]):
        self.func = func
        self.last_x = None

    def __call__(self, x, grad, *args):
        self.store_x(x)
        return self.func(x, grad, *args)

    def store_x(self, x):
        if not np.isnan(np.sum(x)):
            self.last_x = x


class NLOPTOptimiser:
    """
    NLOpt optimiser API class.

    Parameters
    ----------
    algorithm_name:
        Optimisation algorithm to use
    n_variables:
        Size of the variable vector
    opt_conditions:
        Dictionary of algorithm termination criteria
    opt_parameters:
        Dictionary of algorithm parameters
    """

    def __init__(
        self,
        algorithm_name: str,
        n_variables: Optional[int] = None,
        opt_conditions: Optional[Dict[str, float]] = None,
        opt_parameters: Optional[Dict[str, float]] = None,
    ):
        self.opt_conditions = opt_conditions
        self.opt_parameters = opt_parameters
        self.algorithm_name = algorithm_name
        self._flag_built = False
        self.build_optimiser(n_variables)

    def _opt_inputs_ready(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self._flag_built:
                raise OptUtilitiesError(
                    "You must specify the dimensionality of the optimisation problem"
                    f" before using {func.__name__}."
                )
            func(self, *args, **kwargs)

        return wrapper

    def _setup_teardown(self):
        """
        Setup / teardown the wrapper (phoenix design pattern).
        """
        self.n_evals = 0
        self.optimum_value = None
        self._f_objective = None
        self.lower_bounds = None
        self.upper_bounds = None
        self.constraints = []
        self.constraint_tols = []

    @property
    def algorithm_name(self) -> str:
        """
        Name of the optimisation algorithm.
        """
        return self._algorithm_name

    @algorithm_name.setter
    def algorithm_name(self, name: str):
        """
        Setter for the name of the optimisation algorithm.

        Parameters
        ----------
        name:
            Name of the optimisation algorithm

        Raises
        ------
        OptUtilitiesError:
            If the algorithm is not recognised
        """
        if name not in NLOPT_ALG_MAPPING:
            raise OptUtilitiesError(f"Unknown or unmapped algorithm: {name}")
        self._algorithm_name = name

    def build_optimiser(self, n_variables: Optional[int]):
        """
        Build the underlying optimisation algorithm.

        Parameters
        ----------
        n_variables:
            Dimension of the optimisation problem. If None, the algorithm is not built.
        """
        self.n_variables = n_variables
        if n_variables is None:
            return

        self._set_algorithm(n_variables)
        self._flag_built = True
        self.set_termination_conditions(self.opt_conditions)
        self.set_algorithm_parameters(self.opt_parameters)
        self._setup_teardown()

    def _append_constraint_tols(self, constraint, tolerance):
        """
        Append constraint function and tolerances.
        """
        self.constraints.append(constraint)
        self.constraint_tols.append(tolerance)

    def _set_algorithm(self, n_variables):
        """
        Initialise the underlying NLOPT algorithm.
        """
        algorithm = NLOPT_ALG_MAPPING[self.algorithm_name]
        self._opt = nlopt.opt(algorithm, n_variables)

    @_opt_inputs_ready
    def set_algorithm_parameters(self, opt_parameters: Dict[str, float]):
        """
        Set the optimisation algorithm parameters to use.

        Parameters
        ----------
        opt_parameters:
            Optimisation algorithm parameters to use
        """
        if opt_parameters is None:
            return

        for k, v in opt_parameters.items():
            if self._opt.has_param(k):
                self._opt.set_param(k, v)
            elif k == "initial_step":
                self._opt.set_initial_step(v)
            else:
                bluemira_warn(f"Unrecognised algorithm parameter: {k}")

    @_opt_inputs_ready
    def set_termination_conditions(self, opt_conditions: Dict[str, float]):
        """
        Set the optimisation algorithm termination condition(s) to use.

        Parameters
        ----------
        opt_conditions:
            Termination conditions for the optimisation algorithm
        """
        if opt_conditions is None:
            return

        conditions = process_NLOPT_conditions(opt_conditions)

        if "ftol_abs" in conditions:
            self._opt.set_ftol_abs(conditions["ftol_abs"])
        if "ftol_rel" in opt_conditions:
            self._opt.set_ftol_rel(conditions["ftol_rel"])
        if "xtol_abs" in opt_conditions:
            self._opt.set_xtol_abs(conditions["xtol_abs"])
        if "xtol_rel" in opt_conditions:
            self._opt.set_xtol_rel(conditions["xtol_rel"])
        if "max_time" in opt_conditions:
            self._opt.set_maxtime(conditions["max_time"])
        if "max_eval" in opt_conditions:
            self._opt.set_maxeval(conditions["max_eval"])
        if "stop_val" in opt_conditions:
            self._opt.set_stopval(conditions["stop_val"])

    @_opt_inputs_ready
    def set_objective_function(self, f_objective: Callable[[Any], float]):
        """
        Set the objective function (minimisation).

        Parameters
        ----------
        f_objective:
            Objective function to minimise
        """
        f_objective = _NLOPTObjectiveFunction(f_objective)
        self._f_objective = f_objective
        self._opt.set_min_objective(f_objective)

    @_opt_inputs_ready
    def set_lower_bounds(self, lower_bounds: np.ndarray):
        """
        Set the lower bounds.

        Parameters
        ----------
        lower_bounds:
            Lower bound vector
        """
        self.lower_bounds = lower_bounds
        self._opt.set_lower_bounds(lower_bounds)

    @_opt_inputs_ready
    def set_upper_bounds(self, upper_bounds: np.ndarray):
        """
        Set the upper bounds.

        Parameters
        ----------
        upper_bounds:
            Upper bound vector
        """
        self.upper_bounds = upper_bounds
        self._opt.set_upper_bounds(upper_bounds)

    @_opt_inputs_ready
    def add_eq_constraints(
        self, f_constraint: Callable[[Any], np.ndarray], tolerance: float
    ):
        """
        Add a vector-valued equality constraint.

        Parameters
        ----------
        f_constraint:
            Constraint function
        tolerance:
            Tolerance with which to enforce the constraint
        """
        if self.algorithm_name not in EQ_CON_ALGS:
            raise OptUtilitiesError(
                f"{self.algorithm_name} does not support equality constraints."
            )

        self._opt.add_equality_mconstraint(f_constraint, tolerance)
        self._append_constraint_tols(f_constraint, tolerance)

    @_opt_inputs_ready
    def add_ineq_constraints(
        self, f_constraint: Callable[[Any], np.ndarray], tolerance: float
    ):
        """
        Add a vector-valued inequality constraint.

        Parameters
        ----------
        f_constraint:
            Constraint function
        tolerance:
            Tolerance array with which to enforce the constraint
        """
        if self.algorithm_name not in INEQ_CON_ALGS:
            raise OptUtilitiesError(
                f"{self.algorithm_name} does not support inequality constraints."
            )

        self._opt.add_inequality_mconstraint(f_constraint, tolerance)
        self._append_constraint_tols(f_constraint, tolerance)

    def optimise(self, x0: np.ndarray) -> np.ndarray:
        """
        Run the optimiser.

        Parameters
        ----------
        x0:
            Starting solution vector

        Returns
        -------
        Optimal solution vector
        """
        if self._f_objective is None:
            raise OptUtilitiesError(
                "You must first specify an objective function before performing the"
                " optimisation."
            )

        if x0 is None:
            x0 = np.zeros(self.n_variables)

        try:
            x_star = self._opt.optimize(x0)

        except nlopt.RoundoffLimited:
            # It's likely that the last call was still a reasonably good solution.
            self.optimum_value = self._opt.last_optimum_value()
            x_star = self._f_objective.last_x

        except OptVariablesError:
            # Probably still some rounding errors due to numerical gradients
            # It's likely that the last call was still a reasonably good solution.
            bluemira_warn("Badly behaved numerical gradients are causing trouble...")
            self.optimum_value = self._opt.last_optimum_value()
            x_star = np.round(self._f_objective.last_x, 6)

        except RuntimeError as error:
            # Usually "more than iter SQP iterations"
            self.optimum_value = self._opt.last_optimum_value()
            self.n_evals = self._opt.get_numevals()
            process_NLOPT_result(self._opt)
            raise ExternalOptError(str(error))

        except KeyboardInterrupt:
            self.optimum_value = self._opt.last_optimum_value()
            self.n_evals = self._opt.get_numevals()
            process_NLOPT_result(self._opt)
            raise KeyboardInterrupt(
                "The optimisation was halted by the user. Please check "
                "your optimisation problem and termination conditions."
            )

        self.optimum_value = self._opt.last_optimum_value()
        self.n_evals = self._opt.get_numevals()
        process_NLOPT_result(self._opt)
        return x_star

    _opt_inputs_ready = staticmethod(_opt_inputs_ready)
