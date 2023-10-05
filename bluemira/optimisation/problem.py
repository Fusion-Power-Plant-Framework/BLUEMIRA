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
"""Interface for defining an optimisation problem."""
import abc
from typing import Any, Callable, List, Mapping, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import numpy.typing as npt

from bluemira.optimisation._algorithm import Algorithm, AlgorithmType
from bluemira.optimisation._optimise import (
    OptimiserResult,
    optimise,
    validate_constraints,
)
from bluemira.optimisation.typing import ConstraintT


class OptimisationProblemBase:
    """Common base class for OptimisationProblem classes."""

    __MethodT = TypeVar("__MethodT", bound=Callable[..., Any])
    __AnyT = TypeVar("__AnyT")

    def _overridden_or_default(
        self, f: __MethodT, cls: Type[Any], default: __AnyT
    ) -> Union[__MethodT, __AnyT]:
        """
        If the given object is not a member of this class return a default.

        This can be used to decide whether a function has been overridden or not.
        Which is useful in this class for the ``df_objective`` case, where overriding
        the method is possible, but not necessary. We want it to appear in the class
        interface, but we want to be able to tell if it's been overridden so we can
        use an approximate gradient if it has not been.
        """
        if self.__is_method(f, cls):
            return f
        return default

    @staticmethod
    def __is_method(f: __MethodT, cls: Type[Any]) -> bool:
        """
        Determine if the given method is a member of this base class or not.

        Note that ``f`` must be a bound method, i.e., it needs the
        ``__func__`` dunder method.
        """
        try:
            this_f = getattr(cls, f.__name__)
        except AttributeError:
            return False
        return f.__func__ is not this_f


class OptimisationProblem(abc.ABC, OptimisationProblemBase):
    """
    Interface for an optimisation problem.

    This is an alternative to running an optimisation using the
    :func:`.optimise` function.

    Using this interface to define an optimisation can provide a few
    benefits, including:

        * Shared state between optimisation functions and constraints.
          This can enable things like shared parameters and dynamic
          constraints.
        * Switch out optimisation problems using Liskov Substitution.
        * Logical grouping of related functions.
    """

    @abc.abstractmethod
    def objective(self, x: np.ndarray) -> float:
        """The objective function to minimise."""

    def df_objective(self, x: np.ndarray) -> np.ndarray:
        """The gradient of the objective function at ``x``."""

    def eq_constraints(self) -> List[ConstraintT]:  # noqa: PLR6301
        """The equality constraints on the optimisation."""
        return []

    def ineq_constraints(self) -> List[ConstraintT]:  # noqa: PLR6301
        """The inequality constraints on the optimisation."""
        return []

    def bounds(self) -> Tuple[npt.ArrayLike, npt.ArrayLike]:  # noqa: PLR6301
        """
        The lower and upper bounds of the optimisation parameters.

        Each set of bounds must be convertible to a numpy array of
        floats. If the lower or upper bound is a scalar value, that
        value is set as the bound for each of the optimisation
        parameters.
        """
        return -np.inf, np.inf

    def optimise(
        self,
        x0: np.ndarray,
        *,
        algorithm: AlgorithmType = Algorithm.SLSQP,
        opt_conditions: Optional[Mapping[str, Union[int, float]]] = None,
        opt_parameters: Optional[Mapping[str, Any]] = None,
        keep_history: bool = False,
        check_constraints: bool = True,
        check_constraints_warn: bool = True,
    ) -> OptimiserResult:
        """
        Perform the optimisation.

        See :func:`.optimise` for more function parameter details.
        """
        df_objective = self._overridden_or_default(
            self.df_objective, OptimisationProblem, None
        )
        return optimise(
            self.objective,
            df_objective=df_objective,
            x0=x0,
            algorithm=algorithm,
            opt_conditions=opt_conditions,
            opt_parameters=opt_parameters,
            bounds=self.bounds(),
            eq_constraints=self.eq_constraints(),
            ineq_constraints=self.ineq_constraints(),
            keep_history=keep_history,
            check_constraints=check_constraints,
            check_constraints_warn=check_constraints_warn,
        )

    def check_constraints(self, x: np.ndarray, warn: bool = True) -> bool:
        """
        Check if the given parameterisation violates this optimiser's constraints.

        Parameters
        ----------
        x:
            The parametrisation to check the constraints against.
        warn:
            If ``True`` print a warning that lists the violated
            constraints.

        Returns
        -------
        True if any constraints are violated by the parameterisation.
        """
        return validate_constraints(
            x, self.eq_constraints(), self.ineq_constraints(), warn=warn
        )
