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
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from bluemira.geometry.optimisation.typing import (
    GeomClsOptimiserCallable,
    GeomConstraintT,
    GeomOptimiserCallable,
    GeomOptimiserObjective,
)
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.geometry.tools import signed_distance_2D_polygon
from bluemira.geometry.wire import BluemiraWire
from bluemira.optimisation.error import GeometryOptimisationError
from bluemira.optimisation.typing import (
    ConstraintT,
    ObjectiveCallable,
    OptimiserCallable,
)


@dataclass
class KeepOutZone:
    """Definition of a keep-out zone for a geometry optimisation."""

    wire: BluemiraWire
    """Closed wire defining the keep-out zone."""
    byedges: bool = True
    """Whether to discretize the keep-out zone by edges or not."""
    dl: Optional[float] = None
    """
    The discretization length for the keep-out zone.

    This overrides ``n_discr`` if given.
    """
    n_discr: int = 100
    """The number of points to discretise the keep-out zone into."""
    shape_n_discr: int = 100
    """The number of points to discretise the geometry being optimised into."""
    tol: float = 1e-8
    """The tolerance for the keep-out zone constraint."""


def to_objective(
    geom_objective: GeomOptimiserObjective, geom: GeometryParameterisation
) -> ObjectiveCallable:
    """Convert a geometry objective function to a normal objective function."""

    def f(x):
        geom.variables.set_values_from_norm(x)
        return geom_objective(geom)

    return f


def to_optimiser_callable(
    geom_callable: GeomOptimiserCallable,
    geom: GeometryParameterisation,
) -> OptimiserCallable:
    """
    Convert a geometry optimiser function to a normal optimiser function.

    For example, a gradient or constraint.
    """

    def f(x):
        geom.variables.set_values_from_norm(x)
        return geom_callable(geom)

    return f


def to_optimiser_callable_from_cls(
    geom_callable: GeomClsOptimiserCallable,
    geom: GeometryParameterisation,
) -> OptimiserCallable:
    """
    Convert a geometry optimiser function to a normal optimiser function.

    For example, a gradient or constraint.
    """

    def f(x):
        geom.variables.set_values_from_norm(x)
        return geom_callable()

    return f


def to_constraint(
    geom_constraint: GeomConstraintT, geom: GeometryParameterisation
) -> ConstraintT:
    """Convert a geometry constraint to a normal one."""
    constraint: ConstraintT = {
        "f_constraint": to_optimiser_callable(geom_constraint["f_constraint"], geom),
        "df_constraint": None,
        "tolerance": geom_constraint["tolerance"],
    }
    if df_constraint := geom_constraint.get("df_constraint", None):
        constraint["df_constraint"] = to_optimiser_callable(df_constraint, geom)
    return constraint


def calculate_signed_distance(
    parameterisation: GeometryParameterisation,
    n_shape_discr: int,
    zone_points: np.ndarray,
):
    """
    Signed distance from the parameterised shape to the keep-out/in zone.
    """
    shape = parameterisation.create_shape()
    # Note that we do not discretize by edges here, as the number of
    # points must remain constant so the size of constraint vectors
    # remain constant.
    s = shape.discretize(n_shape_discr, byedges=False).xz
    return signed_distance_2D_polygon(s.T, zone_points.T).T


def make_keep_out_zone_constraint(koz: KeepOutZone) -> GeomConstraintT:
    """Make a keep-out zone inequality constraint from a wire."""
    if not koz.wire.is_closed():
        raise GeometryOptimisationError(
            f"Keep-out zone with label '{koz.wire.label}' is not closed."
        )
    koz_points = koz.wire.discretize(koz.n_discr, byedges=koz.byedges, dl=koz.dl).xz
    # Note that we do not allow discretization using 'dl' or 'byedges'
    # for the shape being optimised. The size of the constraint cannot
    # change within an optimisation loop (NLOpt will error) and these
    # options do not guarantee a constant number of discretized points.
    shape_n_discr = koz.shape_n_discr

    def _f_constraint(geom: GeometryParameterisation) -> np.ndarray:
        return calculate_signed_distance(
            geom,
            n_shape_discr=shape_n_discr,
            zone_points=koz_points,
        )

    return {"f_constraint": _f_constraint, "tolerance": np.full(shape_n_discr, koz.tol)}


def get_shape_ineq_constraint(geom: GeometryParameterisation) -> List[ConstraintT]:
    """
    Retrieve the inequality constraints registered for the given parameterisation.

    If no constraints are registered, return an empty list.
    """
    if geom.n_ineq_constraints < 1:
        return []
    if df_constraint := getattr(geom, "df_ineq_constraint", None):
        df_constraint = to_optimiser_callable_from_cls(df_constraint, geom)
    return [
        {
            "f_constraint": to_optimiser_callable_from_cls(geom.f_ineq_constraint, geom),
            "df_constraint": df_constraint,
            "tolerance": geom.tolerance,
        }
    ]
