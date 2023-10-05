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
Geometry optimisation classes and tools
"""

import warnings
from typing import List, Optional

import numpy as np

from bluemira.base.constants import EPS
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.geometry.tools import signed_distance_2D_polygon
from bluemira.geometry.wire import BluemiraWire
from bluemira.utilities.opt_problems import (
    OptimisationConstraint,
    OptimisationObjective,
    OptimisationProblem,
)
from bluemira.utilities.optimiser import Optimiser, approx_derivative

__all__ = ["GeometryOptimisationProblem", "minimise_length", "MinimiseLengthGOP"]

warnings.warn(
    f"The module '{__name__}' is deprecated and will be removed in v2.0.0.\n"
    "See "
    "https://bluemira.readthedocs.io/en/latest/optimisation/optimisation.html "
    "for documentation of the new optimisation module.",
    DeprecationWarning,
    stacklevel=2,
)


def calculate_length(vector, parameterisation):
    """
    Calculate the length of the parameterised shape for a given state vector.
    """
    parameterisation.variables.set_values_from_norm(vector)
    return parameterisation.create_shape().length


def minimise_length(vector, grad, parameterisation, ad_args=None):
    """
    Objective function for nlopt optimisation (minimisation) of length.

    Parameters
    ----------
    vector: np.ndarray
        State vector of the array of coil currents.
    grad: np.ndarray
        Local gradient of objective function used by LD NLOPT algorithms.
        Updated in-place.
    ad_args: Dict
        Additional arguments to pass to the `approx_derivative` function.

    Returns
    -------
    fom: Value of objective function (figure of merit).
    """
    ad_args = ad_args if ad_args is not None else {}

    length = calculate_length(vector, parameterisation)
    if grad.size > 0:
        grad[:] = approx_derivative(
            calculate_length, vector, f0=length, args=(parameterisation,), **ad_args
        )

    return length


def calculate_signed_distance(vector, parameterisation, n_shape_discr, koz_points):
    """
    Calculate the signed distances from the parameterised shape to
    the keep-out zone.
    """
    parameterisation.variables.set_values_from_norm(vector)

    shape = parameterisation.create_shape()
    s = shape.discretize(ndiscr=n_shape_discr).xz
    return signed_distance_2D_polygon(s.T, koz_points.T).T


def constrain_koz(
    constraint, vector, grad, parameterisation, n_shape_discr, koz_points, ad_args=None
):
    """
    Geometry constraint function to the keep-out-zone.
    """
    constraint[:] = calculate_signed_distance(
        vector, parameterisation, n_shape_discr, koz_points
    )
    if grad.size > 0:
        grad[:] = approx_derivative(
            calculate_signed_distance,
            vector,
            f0=constraint,
            args=(parameterisation, n_shape_discr, koz_points),
            **ad_args,
        )
    return constraint


class GeometryOptimisationProblem(OptimisationProblem):
    """
    Geometry optimisation problem class.

    Parameters
    ----------
    parameterisation: GeometryParameterisation
        Geometry parameterisation instance to use in the optimisation problem
    optimiser: bluemira.utilities.optimiser.Optimiser
        Optimiser instance to use in the optimisation problem
    """

    def __init__(
        self,
        geometry_parameterisation: GeometryParameterisation,
        optimiser: Optimiser = None,
        objective: OptimisationObjective = None,
        constraints: Optional[List[OptimisationConstraint]] = None,
    ):
        super().__init__(geometry_parameterisation, optimiser, objective, constraints)

        dimension = geometry_parameterisation.variables.n_free_variables
        bounds = (np.zeros(dimension), np.ones(dimension))
        self.set_up_optimiser(dimension, bounds)
        self._objective._args["ad_args"] = {"bounds": bounds}
        if constraints:
            for constraint in self._constraints:
                constraint._args["ad_args"] = {"bounds": bounds}

    def apply_shape_constraints(self):
        """
        Add shape constraints to the geometry parameterisation, if they exist.
        """
        n_shape_ineq_cons = self._parameterisation.n_ineq_constraints
        if n_shape_ineq_cons > 0:
            self.opt.add_ineq_constraints(
                self._parameterisation.shape_ineq_constraints,
                EPS * np.ones(n_shape_ineq_cons),
            )
        else:
            bluemira_warn(
                "GeometryParameterisation"
                f" {self._parameterisation.__class.__name__} doesnot have any shape"
                " constraints."
            )

    def update_parameterisation(self, x):
        """
        Update the GeometryParameterisation.
        """
        self._parameterisation.variables.set_values_from_norm(x)
        return self._parameterisation

    def optimise(self, x0=None):
        """
        Solve the GeometryOptimisationProblem.
        """
        self._objective._args["parameterisation"] = self._parameterisation
        if x0 is None:
            x0 = self._parameterisation.variables.get_normalised_values()
        x_star = self.opt.optimise(x0)
        return self.update_parameterisation(x_star)


class MinimiseLengthGOP(GeometryOptimisationProblem):
    """
    Optimiser to minimise the length of a geometry 2-D parameterisation
    in the xz-plane, with optional constraints in the form of a
    "keep-out zone".
    """

    def __init__(
        self,
        parameterisation: GeometryParameterisation,
        optimiser: Optimiser,
        keep_out_zone: BluemiraWire = None,
        n_koz_points: int = 100,
        koz_con_tol: float = 1e-3,
    ):
        objective = OptimisationObjective(
            minimise_length, {"parameterisation": parameterisation}
        )
        constraints = []
        if keep_out_zone is not None:
            koz_points = keep_out_zone.discretize(n_koz_points, byedges=True).xz
            koz_constraint = OptimisationConstraint(
                constrain_koz,
                f_constraint_args={
                    "parameterisation": parameterisation,
                    "n_shape_discr": n_koz_points,
                    "koz_points": koz_points,
                },
                tolerance=koz_con_tol * np.ones(n_koz_points),
            )
            constraints.append(koz_constraint)

        super().__init__(parameterisation, optimiser, objective, constraints=constraints)
