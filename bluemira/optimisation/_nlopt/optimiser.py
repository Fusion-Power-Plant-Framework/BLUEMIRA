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
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union

import nlopt
import numpy as np
import numpy.typing as npt

from bluemira.base.look_and_feel import bluemira_debug, bluemira_warn
from bluemira.optimisation._algorithm import Algorithm, AlgorithmType
from bluemira.optimisation._nlopt.conditions import NLOptConditions
from bluemira.optimisation._nlopt.functions import (
    Constraint,
    ConstraintType,
    ObjectiveFunction,
)
from bluemira.optimisation._optimiser import Optimiser, OptimiserResult
from bluemira.optimisation.error import OptimisationError, OptimisationParametersError
from bluemira.utilities.error import OptVariablesError

if TYPE_CHECKING:
    from bluemira.optimisation.typing import ObjectiveCallable, OptimiserCallable

_NLOPT_ALG_MAPPING = {
    Algorithm.SLSQP: nlopt.LD_SLSQP,
    Algorithm.COBYLA: nlopt.LN_COBYLA,
    Algorithm.SBPLX: nlopt.LN_SBPLX,
    Algorithm.MMA: nlopt.LD_MMA,
    Algorithm.BFGS: nlopt.LD_LBFGS,
    Algorithm.DIRECT: nlopt.GN_DIRECT,
    Algorithm.DIRECT_L: nlopt.GN_DIRECT_L,
    Algorithm.CRS: nlopt.GN_CRS2_LM,
    Algorithm.ISRES: nlopt.GN_ISRES,
}


class NloptOptimiser(Optimiser):
    r"""
    Optimiser implementation using NLOpt as the backend.

    Parameters
    ----------
    algorithm:
        The optimisation algorithm to use. Available algorithms are:

            * SLSQP
            * COBYLA
            * SBPLX
            * MMA
            * BFGS
            * DIRECT
            * DIRECT_L
            * CRS
            * ISRES

    n_variables:
        The number of optimisation parameters.
    f_objective:
        The objective function to minimise. This function must take one
        argument (a numpy array), and return a numpy array or float.
    df_objective:
        The derivative of the objective function. This must take the
        form: :math:`f(x) \rightarrow y` where :math:`x` is a numpy
        array containing the optimization parameters, and :math:`y` is a
        numpy array where each element :math:`i` is the partial
        derivative :math:`\frac{\partial f(x)}{\partial x_{i}}`.
        If not given, and a gradient based algorithm is used, a
        numerical approximation of the gradient will be made.
    opt_conditions:
        The stopping conditions for the optimiser. At least one stopping
        condition is required. Supported conditions are:

            * ftol_abs: float
            * ftol_rel: float
            * xtol_abs: float
            * xtol_rel: float
            * max_eval: int
            * max_time: float
            * stop_val: float

    opt_parameters:
        Parameters specific to the algorithm being used. Consult the
        NLopt documentation for these.
    keep_history:
        Whether to record the history of each step of the optimisation.
        (default: ``False``)
    """

    def __init__(
        self,
        algorithm: AlgorithmType,
        n_variables: int,
        f_objective: ObjectiveCallable,
        df_objective: Optional[OptimiserCallable] = None,
        opt_conditions: Optional[Mapping[str, Union[int, float]]] = None,
        opt_parameters: Optional[Mapping[str, Any]] = None,
        keep_history: bool = False,
    ):
        opt_conditions = {} if opt_conditions is None else opt_conditions
        opt_parameters = {} if opt_parameters is None else opt_parameters
        self._keep_history = keep_history

        self._set_algorithm(algorithm)
        self._opt = nlopt.opt(_NLOPT_ALG_MAPPING[self.algorithm], n_variables)
        self._set_objective_function(f_objective, df_objective, n_variables)
        self._set_termination_conditions(opt_conditions)
        self._set_algorithm_parameters(opt_parameters)
        self._eq_constraints: List[Constraint] = []
        self._ineq_constraints: List[Constraint] = []

    @property
    def algorithm(self) -> Algorithm:
        """Return the optimiser's algorithm."""
        return self._algorithm

    @property
    def opt_conditions(self) -> Dict[str, float]:
        """Return the optimiser's stopping conditions."""
        return self._opt_conditions.to_dict()

    @property
    def opt_parameters(self) -> Mapping[str, Union[int, float]]:
        """Return the optimiser algorithms's parameters."""
        return self._opt_parameters

    @property
    def lower_bounds(self) -> np.ndarray:
        """Return the lower bounds for the optimisation parameters."""
        return self._opt.get_lower_bounds().copy()

    @property
    def upper_bounds(self) -> np.ndarray:
        """Return the upper bounds for the optimisation parameters."""
        return self._opt.get_upper_bounds().copy()

    def add_eq_constraint(
        self,
        f_constraint: OptimiserCallable,
        tolerance: np.ndarray,
        df_constraint: Optional[OptimiserCallable] = None,
    ) -> None:
        """
        Add an equality constraint.

        See :meth:`~bluemira.optimisation._optimiser.Optimiser.add_eq_constraint`.
        """
        if self.algorithm not in [Algorithm.SLSQP, Algorithm.COBYLA, Algorithm.ISRES]:
            raise OptimisationError(
                f"Algorithm '{self.algorithm.name}' does not support equality "
                "constraints."
            )
        constraint = Constraint(
            ConstraintType.EQUALITY,
            f_constraint,
            tolerance,
            df_constraint,
            bounds=(self.lower_bounds, self.upper_bounds),
        )
        self._opt.add_equality_mconstraint(constraint.call, constraint.tolerance)
        self._eq_constraints.append(constraint)

    def add_ineq_constraint(
        self,
        f_constraint: OptimiserCallable,
        tolerance: np.ndarray,
        df_constraint: Optional[OptimiserCallable] = None,
    ) -> None:
        """
        Add an inequality constraint.

        See :meth:`~bluemira.optimisation._optimiser.Optimiser.add_ineq_constraint`.
        """
        if self.algorithm not in [Algorithm.SLSQP, Algorithm.COBYLA, Algorithm.ISRES]:
            raise OptimisationError(
                f"Algorithm '{self.algorithm.name}' does not support inequality "
                "constraints."
            )
        constraint = Constraint(
            ConstraintType.INEQUALITY,
            f_constraint,
            tolerance,
            df_constraint,
            bounds=(self.lower_bounds, self.upper_bounds),
        )
        self._opt.add_inequality_mconstraint(constraint.call, constraint.tolerance)
        self._ineq_constraints.append(constraint)

    def optimise(self, x0: Optional[np.ndarray] = None) -> OptimiserResult:
        """
        Run the optimisation.

        See :meth:`~bluemira.optimisation._optimiser.Optimiser.optimise`.
        """
        if x0 is None:
            x0 = _initial_guess_from_bounds(self.lower_bounds, self.upper_bounds)

        try:
            x_star = self._opt.optimize(x0)
            f_x = self._objective.f(x_star)
        except nlopt.RoundoffLimited:
            # It's likely that the last call was still a reasonably good solution.
            x_star, f_x = self._get_previous_iter_result()
        except OptVariablesError:
            # Probably still some rounding errors due to numerical gradients
            # It's likely that the last call was still a reasonably good solution.
            bluemira_warn("Badly behaved numerical gradients are causing trouble...")
            x_star, f_x = self._get_previous_iter_result()
        except RuntimeError as error:
            # Usually "more than iter SQP iterations"
            _process_nlopt_result(self._opt, self.algorithm)
            raise OptimisationError(str(error)) from None
        except KeyboardInterrupt:
            _process_nlopt_result(self._opt, self.algorithm)
            raise KeyboardInterrupt(
                "The optimisation was halted by the user. Please check "
                "your optimisation problem and termination conditions."
            ) from None

        _process_nlopt_result(self._opt, self.algorithm)
        return OptimiserResult(
            f_x=f_x,
            x=x_star,
            n_evals=self._opt.get_numevals(),
            history=self._objective.history,
        )

    def set_lower_bounds(self, bounds: npt.ArrayLike) -> None:
        """
        Set the lower bound for each optimisation parameter.

        See :meth:`~bluemira.optimisation._optimiser.Optimiser.set_lower_bounds`.
        """
        bounds = np.array(bounds)
        _check_bounds(self._opt.get_dimension(), bounds)
        self._opt.set_lower_bounds(bounds)
        # As we use the optimisation variable bounds when calculating an
        # approximate derivative, we must set the new bounds on the
        # objective function and constraints.
        self._objective.set_approx_derivative_lower_bound(bounds)
        for constraint in self._eq_constraints + self._ineq_constraints:
            constraint.set_approx_derivative_lower_bound(bounds)

    def set_upper_bounds(self, bounds: npt.ArrayLike) -> None:
        """
        Set the upper bound for each optimisation parameter.

        See :meth:`~bluemira.optimisation._optimiser.Optimiser.set_upper_bounds`.
        """
        bounds = np.array(bounds)
        _check_bounds(self._opt.get_dimension(), bounds)
        self._opt.set_upper_bounds(bounds)
        # As we use the optimisation variable bounds when calculating an
        # approximate derivative, we must set the new bounds on the
        # objective function and constraints.
        self._objective.set_approx_derivative_upper_bound(bounds)
        for constraint in self._eq_constraints + self._ineq_constraints:
            constraint.set_approx_derivative_upper_bound(bounds)

    def _get_previous_iter_result(self) -> Tuple[np.ndarray, float]:
        """Get the parameterisation and result from the previous iteration."""
        x_star = self._objective.prev_iter
        f_x = self._objective.f(x_star) if x_star.size else np.inf
        return x_star, f_x

    def _handle_round_off_error(self) -> Tuple[np.ndarray, float]:
        """
        Handle a round-off error occurring in an optimisation.

        It's likely the last call was a decent solution, so return that
        (with a warning).
        """
        bluemira_warn(
            "optimisation: round-off error occurred. "
            "Returning last optimisation parameterisation."
        )
        x_star = self._objective.prev_iter
        f_x = self._objective.f(x_star) if x_star.size else np.inf
        return x_star, f_x

    def _set_algorithm(self, alg: AlgorithmType) -> None:
        """Set the optimiser's algorithm."""
        self._algorithm = _check_algorithm(alg)

    def _set_objective_function(
        self,
        func: ObjectiveCallable,
        df: Union[None, OptimiserCallable],
        n_variables: int,
    ) -> None:
        """Wrap and set the objective function."""
        self._objective = ObjectiveFunction(
            func, df, n_variables, bounds=(self.lower_bounds, self.upper_bounds)
        )
        if self._keep_history:
            self._opt.set_min_objective(self._objective.call_with_history)
        else:
            self._opt.set_min_objective(self._objective.call)

    def _set_termination_conditions(
        self, opt_conditions: Mapping[str, Union[int, float]]
    ) -> None:
        """Validate and set the termination conditions."""
        self._opt_conditions = NLOptConditions(**opt_conditions)
        if self._opt_conditions.ftol_abs:
            self._opt.set_ftol_abs(self._opt_conditions.ftol_abs)
        if self._opt_conditions.ftol_rel:
            self._opt.set_ftol_rel(self._opt_conditions.ftol_rel)
        if self._opt_conditions.xtol_abs:
            self._opt.set_xtol_abs(self._opt_conditions.xtol_abs)
        if self._opt_conditions.xtol_rel:
            self._opt.set_xtol_rel(self._opt_conditions.xtol_rel)
        if self._opt_conditions.max_time:
            self._opt.set_maxtime(self._opt_conditions.max_time)
        if self._opt_conditions.max_eval:
            self._opt.set_maxeval(self._opt_conditions.max_eval)
        if self._opt_conditions.stop_val:
            self._opt.set_stopval(self._opt_conditions.stop_val)

    def _set_algorithm_parameters(self, opt_parameters: Mapping) -> None:
        self._opt_parameters = opt_parameters
        unrecognised = []
        for k, v in self.opt_parameters.items():
            if self._opt.has_param(k):
                self._opt.set_param(k, v)
            elif k == "initial_step":
                self._opt.set_initial_step(v)
            else:
                unrecognised.append(k)

        if unrecognised:
            raise OptimisationParametersError(
                f"Unrecognised algorithm parameter(s): {str(unrecognised)[1:-1]}"
            )


def _check_algorithm(algorithm: AlgorithmType) -> Algorithm:
    """Validate, and convert, the given algorithm."""
    if isinstance(algorithm, str):
        return Algorithm[algorithm]
    if isinstance(algorithm, Algorithm):
        return algorithm
    raise TypeError(f"Cannot set algorithm with object of type '{type(algorithm)}'.")


def _check_bounds(n_dims: int, new_bounds: np.ndarray) -> None:
    """Validate that the bounds have the correct dimensions."""
    if new_bounds.ndim != 1 or new_bounds.size != n_dims:
        raise ValueError(
            f"Cannot set bounds with shape '{new_bounds.shape}', "
            f"array must be one dimensional and have '{n_dims}' elements."
        )


def _initial_guess_from_bounds(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Derive an initial guess for the optimiser.

    Takes the center of the bounds for each parameter.
    """
    bounds = np.array([lower, upper])
    # bounds are +/- inf by default, change to real numbers so
    # we can take an average
    np.nan_to_num(
        bounds,
        posinf=np.finfo(np.float64).max,
        neginf=np.finfo(np.float64).min,
        copy=False,
    )
    return np.mean(bounds, axis=0)


def _process_nlopt_result(opt: nlopt.opt, algorithm: Algorithm) -> None:
    """
    Communicate to the user the NLopt optimisation result.

    Usually this would be called when dealing with an error in an
    optimisation loop.

    Parameters
    ----------
    opt:
        The optimiser to check
    algorithm:
        The optimisation algorithm
    """
    result = opt.last_optimize_result()
    message = None
    log_func = bluemira_warn
    if result == nlopt.MAXEVAL_REACHED:
        message = "optimiser succeeded but stopped at the maximum number of evaluations."
        if algorithm is Algorithm.ISRES:
            log_func = bluemira_debug
            message += f"\nThis is expected for the {Algorithm.ISRES.name} algorithm."
    elif result == nlopt.MAXTIME_REACHED:
        message = "optimiser succeeded but stopped at the maximum duration."
    elif result == nlopt.ROUNDOFF_LIMITED:
        message = (
            "optimiser was halted due to round-off errors.\n"
            "Returning last optimisation parameterisation."
        )
    elif result == nlopt.FAILURE:
        message = "optimiser failed real hard..."
    elif result == nlopt.INVALID_ARGS:
        message = "optimiser failed because of invalid arguments."
    elif result == nlopt.OUT_OF_MEMORY:
        message = "optimiser failed because it ran out of memory."
    elif result == nlopt.FORCED_STOP:
        message = "optimiser failed because of a forced stop."
    if message:
        log_func(f"\n{message}\n")
