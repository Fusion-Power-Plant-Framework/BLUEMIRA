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
import enum
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from bluemira.optimisation._tools import approx_derivative
from bluemira.optimisation.typing import ObjectiveCallable, OptimiserCallable

_FloatOrArrayT = Union[np.ndarray, float]


class ConstraintType(enum.Enum):
    """Enumeration of constraint types."""

    EQUALITY = enum.auto()
    INEQUALITY = enum.auto()


class _NloptFunction:
    """
    Base class for NLOpt objective/constraint functions.

    Implements the ability to calculate a numerical gradient, if an
    analytical one is not provided.
    """

    def __init__(
        self,
        f: Callable[[np.ndarray], _FloatOrArrayT],
        bounds: Tuple[_FloatOrArrayT, _FloatOrArrayT],
    ):
        self.f = f
        self.f0: Union[float, np.ndarray] = 0
        self.bounds = bounds

    def _approx_derivative(self, x: np.ndarray) -> np.ndarray:
        return approx_derivative(self.f, x, bounds=self.bounds, f0=self.f0)

    def set_approx_derivative_lower_bound(self, lower_bound: np.ndarray) -> None:
        """Set the lower bounds for use in derivative approximation."""
        self.bounds = (lower_bound, self.bounds[1])

    def set_approx_derivative_upper_bound(self, upper_bound: np.ndarray) -> None:
        """Set the upper bounds for use in derivative approximation."""
        self.bounds = (self.bounds[0], upper_bound)


class ObjectiveFunction(_NloptFunction):
    """
    Holds an objective function for an NLOpt optimiser.

    Adapts the given objective function, and optional derivative, to a
    form understood by NLOpt.

    If no optimiser derivative is given, and the algorithm is gradient
    based, a numerical approximation of the gradient is calculated.
    """

    f: ObjectiveCallable
    f0: float

    def __init__(
        self,
        f: ObjectiveCallable,
        df: Optional[OptimiserCallable],
        n_variables: int,
        bounds: Tuple[_FloatOrArrayT, _FloatOrArrayT] = (-np.inf, np.inf),
    ):
        super().__init__(f, bounds)
        self.df = df if df is not None else self._approx_derivative
        self.history: List[Tuple[np.ndarray, float]] = []
        self.prev_iter = np.zeros(n_variables, dtype=float)

    def call(self, x: np.ndarray, grad: np.ndarray) -> float:
        """Execute the NLOpt objective function."""
        if not np.any(np.isnan(x)):
            self._store_x(x)
        return self._call_inner(x, grad)

    def call_with_history(self, x: np.ndarray, grad: np.ndarray) -> float:
        """Execute the NLOpt objective function, recording the iteration history."""
        f_x = self._call_inner(x, grad)
        self.history.append((np.copy(x), f_x))
        self.prev_iter = self.history[-1][0]
        return f_x

    def _call_inner(self, x: np.ndarray, grad: np.ndarray) -> float:
        """Execute the objective function in the form required by NLOpt."""
        # Cache f(x) so we do not need to recalculate it if we're using
        # an approximate gradient
        self.f0 = self.f(x)
        if grad.size > 0:
            grad[:] = self.df(x)
        return self.f0

    def _store_x(self, x: np.ndarray) -> None:
        """Store ``x`` in ``self.prev_iter``."""
        # Assign in place to avoid lots of allocations.
        # Not benchmarked, but may be more efficient...
        self.prev_iter[:] = x


class Constraint(_NloptFunction):
    """Holder for NLOpt constraint functions."""

    f: OptimiserCallable
    f0: np.ndarray

    def __init__(
        self,
        constraint_type: ConstraintType,
        f: OptimiserCallable,
        tolerance: np.ndarray,
        df: Optional[OptimiserCallable] = None,
        bounds: Tuple[_FloatOrArrayT, _FloatOrArrayT] = (-np.inf, np.inf),
    ):
        super().__init__(f, bounds)
        self.constraint_type = constraint_type
        self.tolerance = tolerance
        self.df = df if df is not None else self._approx_derivative

    def call(self, result: np.ndarray, x: np.ndarray, grad: np.ndarray) -> None:
        """
        Execute the constraint function in the form required by NLOpt.

        https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/#vector-valued-constraints
        """
        # Cache f(x) so we do not need to recalculate it if we're using
        # an approximate gradient
        result[:] = self.f(x)
        self.f0 = result
        if grad.size > 0:
            grad[:] = self.df(x)
