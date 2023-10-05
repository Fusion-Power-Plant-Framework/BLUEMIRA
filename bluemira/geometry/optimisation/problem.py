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
"""Interface for defining a geometry-based optimisation problem."""

import abc
from typing import Any, List, Mapping, Optional, TypeVar, Union

import numpy as np

from bluemira.geometry.optimisation._optimise import (
    GeomOptimiserResult,
    KeepOutZone,
    optimise_geometry,
)
from bluemira.geometry.optimisation.typing import GeomConstraintT
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.optimisation._algorithm import Algorithm, AlgorithmType
from bluemira.optimisation.problem import OptimisationProblemBase

_GeomT = TypeVar("_GeomT", bound=GeometryParameterisation)


class GeomOptimisationProblem(abc.ABC, OptimisationProblemBase):
    """
    Interface for a geometry optimisation problem.

    This is an alternative to running a geometry optimisation using the
    :func:`.optimise_geometry` function.
    """

    @abc.abstractmethod
    def objective(self, geom: _GeomT) -> float:
        """The objective function to minimise."""

    def df_objective(self, geom: _GeomT) -> np.ndarray:
        """
        The derivative of the objective function.

        If not overridden, an approximation of the derivative is made
        using the 'central differences' method.
        This method is ignored if a non-gradient based algorithm is
        used when calling :meth:`.GeomOptimisationProblem.optimise`.
        """
        raise NotImplementedError

    def eq_constraints(self) -> List[GeomConstraintT]:  # noqa: PLR6301
        """
        List of equality constraints for the optimisation.

        See :func:`.optimise_geometry` for a description of the form
        these constraints should take.
        """
        return []

    def ineq_constraints(self) -> List[GeomConstraintT]:  # noqa: PLR6301
        """
        List of inequality constraints for the optimisation.

        See :func:`.optimise_geometry` for a description of the form
        these constraints should take.
        """
        return []

    def keep_out_zones(self) -> List[KeepOutZone]:  # noqa: PLR6301
        """
        List of geometric keep-out zones.

        An iterable of keep-out zones: closed wires that the geometry
        must not intersect.
        """
        return []

    def optimise(
        self,
        geom: _GeomT,
        *,
        algorithm: AlgorithmType = Algorithm.SLSQP,
        opt_conditions: Optional[Mapping[str, Union[int, float]]] = None,
        opt_parameters: Optional[Mapping[str, Any]] = None,
        keep_history: bool = False,
        check_constraints: bool = True,
        check_constraints_warn: bool = True,
    ) -> GeomOptimiserResult[_GeomT]:
        """
        Run the geometry optimisation.

        See :func:`.optimise_geometry` for a description of the
        parameters.

        Returns
        -------
        The result of the optimisation.
        """
        df_objective = self._overridden_or_default(
            self.df_objective, GeomOptimisationProblem, None
        )
        return optimise_geometry(
            geom,
            f_objective=self.objective,
            df_objective=df_objective,
            keep_out_zones=self.keep_out_zones(),
            algorithm=algorithm,
            opt_conditions=opt_conditions,
            opt_parameters=opt_parameters,
            eq_constraints=self.eq_constraints(),
            ineq_constraints=self.ineq_constraints(),
            keep_history=keep_history,
            check_constraints=check_constraints,
            check_constraints_warn=check_constraints_warn,
        )
