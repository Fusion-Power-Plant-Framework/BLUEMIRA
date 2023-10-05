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
"""Definition of the generic `optimise` function."""

from pprint import pformat
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.optimisation._algorithm import (
    Algorithm,
    AlgorithmDefaultConditions,
    AlgorithmType,
)
from bluemira.optimisation._nlopt import NloptOptimiser
from bluemira.optimisation._optimiser import Optimiser, OptimiserResult
from bluemira.optimisation.typing import (
    ConstraintT,
    ObjectiveCallable,
    OptimiserCallable,
)


def optimise(
    f_objective: ObjectiveCallable,
    df_objective: Optional[OptimiserCallable] = None,
    *,
    x0: Optional[np.ndarray] = None,
    dimensions: Optional[int] = None,
    algorithm: AlgorithmType = Algorithm.SLSQP,
    opt_conditions: Optional[Mapping[str, Union[int, float]]] = None,
    opt_parameters: Optional[Mapping[str, Any]] = None,
    bounds: Optional[Tuple[npt.ArrayLike, npt.ArrayLike]] = None,
    eq_constraints: Iterable[ConstraintT] = (),
    ineq_constraints: Iterable[ConstraintT] = (),
    keep_history: bool = False,
    check_constraints: bool = True,
    check_constraints_warn: bool = True,
) -> OptimiserResult:
    r"""
    Find the parameters that minimise the given objective function.

    Parameters
    ----------
    f_objective:
        The objective function to minimise.
    dimensions:
        The dimensionality of the problem. This or ``x0`` must be given.
    x0:
        The initial guess for the optimisation parameters. This or
        `dimensions` must be given, if both are given, ``x0.size`` must
        be equal to ``dimensions``.
    df_objective:
        The derivative of the objective function.
    algorithm:
        The optimisation algorithm to use. See enum
        :class:`.Algorithm` for supported algorithms.
        (default: ``"SLSQP"``)
    opt_conditions:
        The stopping conditions for the optimiser. Supported conditions
        are:

            * ftol_abs: float
            * ftol_rel: float
            * xtol_abs: float
            * xtol_rel: float
            * max_eval: int
            * max_time: float
            * stop_val: float

        for defaults see
        :class:`~bluemira.optimisation._algorithm.AlgorithmDefaultConditions`.

    opt_parameters:
        The algorithm-specific optimisation parameters.
    bounds:
        The upper and lower bounds for the optimisation parameters.
        The first array being the lower bounds, the second the upper.
        This can also be ``None``, in which case the bounds are
        ``(-inf, inf)`` for each optimisation parameter. You can also
        specify scalars for either the upper or lower bounds or both,
        and those bound will be applied to every optimisation parameter.
    eq_constraints:
        The equality constraints for the optimiser.
        A dict with keys:

            * f_constraint: the constraint function.
            * tolerance: the tolerances in each constraint dimension.
            * df_constraint (optional): the derivative of the constraint
              function. If not given, a numerical approximation of the
              gradient is made (if a gradient is required).

        A constraint is a vector-valued, non-linear, equality
        constraint of the form :math:`f_{c}(x) = 0`.

        The constraint function should have the form
        :math:`f(x) \rightarrow y`, where:

            * :math:`x` is a numpy array of the optimisation parameters.
            * :math:`y` is a numpy array containing the values of the
              constraint at :math:`x`, with size :math:`m`, where
              :math:`m` is the dimensionality of the constraint.

        The tolerance array must have the same dimensionality as the
        constraint.

        The gradient function should have the same form as the
        constraint function, however its output should have size
        :math:`n \times m` where :math:`m` is the dimensionality of the
        constraint and :math:`n` is the number of optimisation
        parameters.

        Equality constraints are only supported by algorithms:

            * SLSQP
            * COBYLA
            * ISRES

    ineq_constraints:
        The inequality constraints for the optimiser.
        This argument has the same form as the ``eq_constraint`` argument,
        but each constraint is in the form :math:`f_{c}(x) \le 0`.

        Inequality constraints are only supported by algorithms:

            * SLSQP
            * COBYLA
            * ISRES

    keep_history:
        Whether to record the history of each step of the optimisation.
        (default: False)
    check_constraints:
        Whether to check all constraints have been satisfied at the end
        of the optimisation, and warn if they have not. Note that, if
        this is set to False, the result's ``constraints_satisfied``
        attribute will be set to ``None``.
    check_constraints_warn:
        Whether to print a warning that constraints have not been
        satisfied at the end of an optimisation. This argument has no
        effect if ``check_constraints`` is ``False``.

    Returns
    -------
    The result of the optimisation; including the optimised parameters
    and the number of iterations.
    """
    if dimensions is None:
        if x0 is not None:
            dimensions = x0.size
        else:
            raise ValueError("Must give argument 'dimension' or 'x0'.")
    elif x0 is not None and x0.size != dimensions:
        raise ValueError("Size of 'x0' and 'dimensions' must agree.")

    opt_conditions = _set_default_termination_conditions(algorithm, opt_conditions)

    bounds = _process_bounds(bounds, dimensions)
    # Convert to lists, as these could be generators, and we may need to
    # consume them more than once.
    eq_constraints = list(eq_constraints)
    ineq_constraints = list(ineq_constraints)

    optimiser = _make_optimiser(
        f_objective,
        dimensions,
        df_objective,
        algorithm,
        opt_conditions,
        opt_parameters,
        bounds,
        eq_constraints,
        ineq_constraints,
        keep_history,
    )
    result = optimiser.optimise(x0)
    if check_constraints:
        result.constraints_satisfied = validate_constraints(
            result.x, eq_constraints, ineq_constraints, warn=check_constraints_warn
        )
    return result


def validate_constraints(
    x_star: np.ndarray,
    eq_constraints: List[ConstraintT],
    ineq_constraints: List[ConstraintT],
    warn: bool = True,
) -> bool:
    """
    Check the given parametrisation satisfies the given constraints.

    Additionally, print warnings listing constraints that are not
    satisfied.

    Parameters
    ----------
    x_star:
        The parameterisation to check the constraints against.
    eq_constraints:
        The list of equality constraints to check.
    ineq_constraints:
        The list of inequality constraints to check.
    warn:
        Whether to print warnings if constraints are violated.
        Default is true.

    Returns
    -------
    True if no constraints are violated by the parameterisation.
    """
    eq_warnings = _check_constraints(x_star, eq_constraints, "equality")
    ineq_warnings = _check_constraints(x_star, ineq_constraints, "inequality")
    all_warnings = eq_warnings + ineq_warnings
    if all_warnings:
        if warn:
            message = "\n".join(all_warnings)
            bluemira_warn(
                f"Some constraints have not been adequately satisfied.\n{message}"
            )
        return False
    return True


def _make_optimiser(
    f_objective: ObjectiveCallable,
    dimensions: int,
    df_objective: Optional[OptimiserCallable] = None,
    algorithm: AlgorithmType = Algorithm.SLSQP,
    opt_conditions: Optional[Mapping[str, Union[int, float]]] = None,
    opt_parameters: Optional[Mapping[str, Any]] = None,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    eq_constraints: Iterable[ConstraintT] = (),
    ineq_constraints: Iterable[ConstraintT] = (),
    keep_history: bool = False,
) -> Optimiser:
    """Make a new optimiser object."""
    opt = NloptOptimiser(
        algorithm,
        dimensions,
        f_objective=f_objective,
        df_objective=df_objective,
        opt_conditions=opt_conditions,
        opt_parameters=opt_parameters,
        keep_history=keep_history,
    )
    for constraint in eq_constraints:
        opt.add_eq_constraint(**constraint)
    for constraint in ineq_constraints:
        opt.add_ineq_constraint(**constraint)
    if bounds:
        opt.set_lower_bounds(bounds[0])
        opt.set_upper_bounds(bounds[1])
    return opt


def _process_bounds(
    bounds: Union[Tuple[npt.ArrayLike, npt.ArrayLike], None], dims: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Handle bounds converting ``None`` to +/-inf and expanding scalar bounds."""
    if bounds is None:
        return (np.full(dims, -np.inf), np.full(dims, np.inf))
    bounds = tuple(bounds)
    if len(bounds) != 2:  # noqa: PLR2004
        raise ValueError(f"Bounds must have exactly 2 elements, found '{len(bounds)}'")
    new_bounds = [np.full(dims, b) if np.isscalar(b) else np.array(b) for b in bounds]
    return new_bounds[0], new_bounds[1]


def _check_constraints(
    x_star: np.ndarray,
    constraints: List[ConstraintT],
    constraint_type: Literal["inequality", "equality"],
) -> List[str]:
    """
    Check if any of the given constraints are violated by the parameterisation.

    Returns a list of formatted warnings. If there are no warnings, there
    are no violations.
    """

    def _check_constraint(
        x_star: np.ndarray,
        constraint: ConstraintT,
        condition: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], None]:
        """Return the items in the constraint vector that violate the condition."""
        c_value = constraint["f_constraint"](x_star)
        # Deal with scalar constraints
        c_value = np.array([c_value]) if np.isscalar(c_value) else c_value
        tols = np.array(constraint["tolerance"])
        indices = np.nonzero(condition(c_value, tols))[0]
        if indices.size > 0:
            return (indices, c_value, tols)
        return None

    condition, comp_str = (
        (_eq_constraint_condition, "!=")
        if constraint_type == "equality"
        else (_ineq_constraint_condition, "!<")
    )

    warnings = []
    for i, constraint in enumerate(constraints):
        if diff := _check_constraint(x_star, constraint, condition):
            indices, c_value, tols = diff
            warnings.append(
                "\n".join(
                    [
                        f"{constraint_type} constraint {i} [{j}]: "
                        f"{pformat(c_value[j])} {comp_str} {pformat(tols[j])}"
                        for j in indices
                    ]
                )
            )
    return warnings


def _eq_constraint_condition(c_value: np.ndarray, tols: np.ndarray) -> np.ndarray:
    """Condition under which an equality constraint is violated."""
    return ~np.isclose(c_value, 0, atol=tols)


def _ineq_constraint_condition(c_value: np.ndarray, tols: np.ndarray) -> np.ndarray:
    """Condition under which an inequality constraint is violated."""
    return c_value > tols


def _set_default_termination_conditions(
    algorithm: AlgorithmType,
    opt_conditions: Optional[Mapping[str, Union[int, float]]] = None,
) -> Optional[Mapping[str, Union[int, float]]]:
    if opt_conditions is None:
        if isinstance(algorithm, str):
            algorithm = Algorithm[algorithm]

        if not isinstance(algorithm, Algorithm):
            return opt_conditions

        return getattr(AlgorithmDefaultConditions(), algorithm.name).to_dict()
    return opt_conditions
