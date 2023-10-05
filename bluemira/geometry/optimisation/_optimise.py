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
import copy
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    TypedDict,
    Union,
)

import numpy as np
from typing_extensions import NotRequired

from bluemira.geometry.optimisation import _tools
from bluemira.geometry.optimisation._tools import KeepOutZone
from bluemira.geometry.optimisation.typing import (
    GeomConstraintT,
    GeomOptimiserCallable,
    GeomOptimiserObjective,
)
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.geometry.wire import BluemiraWire
from bluemira.optimisation._algorithm import Algorithm, AlgorithmType
from bluemira.optimisation._optimise import optimise

_GeomT = TypeVar("_GeomT", bound=GeometryParameterisation)

__all__ = ["GeomOptimiserResult", "optimise_geometry", "KeepOutZone"]


@dataclass
class GeomOptimiserResult(Generic[_GeomT]):
    """Container for the result of a geometry optimisation."""

    # Note that the attributes here are duplicates of those in
    # 'OptimiserResult`. This is because, until Python 3.10, you cannot
    # extend dataclasses with default values through inheritance:
    # https://stackoverflow.com/a/53085935.
    # Once we're on Python 3.10, we can use `the `kw_only` argument of
    # `dataclass` to tidy this up.
    geom: _GeomT
    """The geometry parameterisation with optimised parameters."""
    f_x: float
    """The evaluation of the optimised parameterisation."""
    n_evals: int
    """The number of evaluations of the objective function in the optimisation."""
    history: List[Tuple[np.ndarray, float]] = field(repr=False)
    """The history of the parametrisation at each iteration."""
    constraints_satisfied: Union[bool, None] = None
    """
    Whether all constraints have been satisfied to within the required tolerance.

    Is ``None`` if constraints have not been checked.
    """


class KeepOutZoneDict(TypedDict):
    """Typing for a dict representing a keep-out zone for a geometry optimisation."""

    wire: BluemiraWire
    """Closed wire defining the keep-out zone."""
    byedges: NotRequired[bool]
    """Whether to discretize the keep-out zone by edges or not."""
    dl: NotRequired[Optional[float]]
    """
    The discretization length for the keep-out zone.

    This overrides ``n_discr`` if given.
    """
    n_discr: NotRequired[int]
    """The number of points to discretise the wire into."""
    shape_n_discr: NotRequired[int]
    """The number of points to discretise the keep-out zone into."""
    tol: NotRequired[float]
    """The number of points to discretise the geometry being optimised into."""


def optimise_geometry(
    geom: _GeomT,
    f_objective: GeomOptimiserObjective,
    df_objective: Optional[GeomOptimiserCallable] = None,
    *,
    keep_out_zones: Iterable[Union[BluemiraWire, KeepOutZoneDict, KeepOutZone]] = (),
    algorithm: AlgorithmType = Algorithm.SLSQP,
    opt_conditions: Optional[Mapping[str, Union[int, float]]] = None,
    opt_parameters: Optional[Mapping[str, Any]] = None,
    eq_constraints: Iterable[GeomConstraintT] = (),
    ineq_constraints: Iterable[GeomConstraintT] = (),
    keep_history: bool = False,
    check_constraints: bool = True,
    check_constraints_warn: bool = True,
) -> GeomOptimiserResult[_GeomT]:
    r"""
    Minimise the given objective function for a geometry parameterisation.

    Parameters
    ----------
    geom:
        The geometry to optimise the parameters of. The existing
        parameterisation is used as the initial guess in the
        optimisation.
    f_objective:
        The objective function to minimise. Must take as an argument a
        ``GeometryParameterisation`` and return a ``float``.
    df_objective:
        The derivative of the objective function, by default ``None``.
        If not given, an approximation of the derivative is made using
        the 'central differences' method.
        This argument is ignored if a non-gradient based algorithm is
        used.
    keep_out_zones:
        An iterable of keep-out zones: closed wires that the geometry
        must not intersect.
        Each item can be given as a :class:`.KeepOutZone`, or a
        dictionary with keys the same as the properties of the
        `:class:`.KeepOutZone` class, or just a :class:`.BluemiraWire`.
    algorithm:
        The optimisation algorithm to use, by default :obj:`.Algorithm.SLSQP`.
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
    eq_constraints:
        The equality constraints for the optimiser.
        A dict with keys:

        * f_constraint: the constraint function.
        * tolerance: the tolerances in each constraint dimension.
        * df_constraint (optional): the derivative of the constraint
            function. If not given, a numerical approximation of the
            gradient is made (if a gradient is required).

        The constraint is a vector-valued, non-linear, equality
        constraint of the form :math:`f_{c}(g) = 0`.

        The constraint function should have the form
        :math:`f(g) \rightarrow y`, where:

            * :math:`g` is a geometry parameterisation.
            * :math:`y` is a numpy array containing the values of the
              constraint at :math:`g`, with size :math:`m`, where
              :math:`m` is the dimensionality of the constraint.


        The tolerance array must have the same dimensionality as the
        constraint.

        The gradient function should have the same form as the
        constraint function, however its output should have size
        :math:`n \times m` where, again, :math:`m` is the dimensionality
        of the constraint and :math:`n` is the number of parameters in
        the geometry parameterisation.

        Equality constraints are only supported by algorithms:

            * SLSQP
            * COBYLA
            * ISRES

    ineq_constraints:
        The geometric inequality constraints for the optimiser.
        This argument has the same form as the ``eq_constraint``
        argument, but each constraint is of the form
        :math:`f_{c}(g) \le 0`.

        Inequality constraints are only supported by algorithms:

            * SLSQP
            * COBYLA
            * ISRES

    keep_history:
        Whether or not to record the history of the optimisation
        parameters at each iteration. Note that this can significantly
        impact the performance of the optimisation.
        (default: ``False``)
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
    The result of the optimisation.
    """
    geom = copy.deepcopy(geom)
    dimensions = geom.variables.n_free_variables
    f_obj = _tools.to_objective(f_objective, geom)
    if df_objective is not None:
        df_obj = _tools.to_optimiser_callable(df_objective, geom)
    else:
        df_obj = None
    ineq_constraints_list = list(ineq_constraints)
    ineq_constraints_list.extend(
        [_tools.make_keep_out_zone_constraint(_to_koz(zone)) for zone in keep_out_zones]
    )

    ineq_constraints = _tools.get_shape_ineq_constraint(geom) + [
        _tools.to_constraint(c, geom) for c in ineq_constraints_list
    ]
    result = optimise(
        f_obj,
        dimensions=dimensions,
        x0=geom.variables.get_normalised_values(),
        df_objective=df_obj,
        algorithm=algorithm,
        opt_conditions=opt_conditions,
        opt_parameters=opt_parameters,
        bounds=(np.zeros(dimensions), np.ones(dimensions)),
        eq_constraints=[_tools.to_constraint(c, geom) for c in eq_constraints],
        ineq_constraints=ineq_constraints,
        keep_history=keep_history,
        check_constraints=check_constraints,
        check_constraints_warn=check_constraints_warn,
    )
    # Make sure we update the geometry with the result, the last
    # geometry update may not have been with the optimum result
    geom.variables.set_values_from_norm(result.x)

    result_dict = asdict(result)
    result_dict.pop("x")
    return GeomOptimiserResult(**result_dict, geom=geom)


def _to_koz(koz: Union[BluemiraWire, KeepOutZoneDict, KeepOutZone]) -> KeepOutZone:
    """Convert ``koz`` to a ``KeepOutZone``."""
    if isinstance(koz, BluemiraWire):
        return KeepOutZone(koz)
    if isinstance(koz, Mapping):
        return KeepOutZone(**koz)
    if isinstance(koz, KeepOutZone):
        return koz
    raise TypeError(f"Type '{type(koz).__name__}' is not a valid keep-out zone.")
