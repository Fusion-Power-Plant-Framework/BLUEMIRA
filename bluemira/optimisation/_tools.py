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
"""Collection of utility functions for the module."""

from typing import Any, Callable, Iterable, Optional, Tuple, Union

import numpy as np
from scipy.optimize._numdiff import approx_derivative as _approx_derivative

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.optimisation.error import OptimisationError

_FloatOrArray = Union[float, np.ndarray]


def approx_derivative(
    func: Callable[[np.ndarray], _FloatOrArray],
    x0: np.ndarray,
    method: str = "3-point",
    rel_step: Optional[_FloatOrArray] = None,
    f0: Optional[_FloatOrArray] = None,
    bounds: Optional[Iterable[_FloatOrArray]] = (-np.inf, np.inf),
    args: Optional[Tuple[Any, ...]] = (),
) -> np.ndarray:
    """
    Approximate the gradient of a function about a point.

    Parameters
    ----------
    func:
        Function for which to calculate the gradient.
    x0:
        Point about which to calculate the gradient.
    method:
        Finite difference method to use.
    rel_step:
        Relative step size to use.
    f0:
        Result of func(x0). If None, this is recomputed.
    bounds:
        Lower and upper bounds on individual variables.
    args:
        Additional positional arguments to ``func``.
    """
    return _approx_derivative(
        func, x0, method=method, rel_step=rel_step, f0=f0, bounds=bounds, args=args
    )


def process_scipy_result(res):
    """
    Handle a scipy.minimize OptimizeResult object. Process error codes, if any.

    Parameters
    ----------
    res:
        Scipy optimise result

    Returns
    -------
    x: np.array
        The optimal set of parameters (result of the optimisation)

    Raises
    ------
    InternalOptError if an error code returned without a usable result.
    """
    if res.success:
        return res.x

    if not hasattr(res, "status"):
        bluemira_warn("Scipy optimisation was not succesful. Failed without status.")
        raise OptimisationError(f"{res.message}\n{res.__str__()}")

    if res.status == 8:  # noqa: PLR2004
        # This can happen when scipy is not convinced that it has found a minimum.
        bluemira_warn(
            "\nOptimiser (scipy) found a positive directional derivative,\n"
            f"returning suboptimal result. \n\n{res.message}{res.__str__()}"
        )
        return res.x

    if res.status == 9:  # noqa: PLR2004
        bluemira_warn(
            "\nOptimiser (scipy) exceeded number of iterations, returning"
            f"suboptimal result. \n\n{res.message}{res.__str__()}"
        )
        return res.x

    raise OptimisationError(f"{res.message}\n{res.__str__()}")
