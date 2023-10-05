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
Just-in-time compilation and LowLevelCallable speed-up tools.
"""
import warnings
from typing import Callable, Iterable, List, Union

import numba as nb
import numpy as np
from numba.types import CPointer, float64, intc
from scipy import LowLevelCallable
from scipy.integrate import IntegrationWarning, nquad, quad

from bluemira.geometry.coordinates import Coordinates
from bluemira.magnetostatics.error import (
    MagnetostaticsError,
    MagnetostaticsIntegrationError,
)

__all__ = [
    "jit_llc3",
    "jit_llc4",
    "jit_llc5",
    "jit_llc7",
    "integrate",
    "n_integrate",
    "process_coords_array",
    "process_xyz_array",
]


# Integration decorators and utilities
# Significant speed gains by using LowLevelCallable (> x2)


def process_xyz_array(func):
    """
    Decorator for coordinate input handling in array-return functions and methods.
    """

    def wrapper(cls, x, y, z):
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)

        if not len(x) == len(y) == len(z):
            raise MagnetostaticsError("Coordinate vector lengths must be equal.")

        if len(x) == 1:
            # Float handling
            return func(cls, x[0], y[0], z[0])
        if len(x.shape) == 1:
            # 1-D array handling
            return np.array([func(cls, xi, yi, zi) for xi, yi, zi in zip(x, y, z)]).T
        if len(x.shape) == 2:  # noqa: PLR2004
            # 2-D array handling
            m, n = x.shape
            result = np.zeros((3, m, n))
            for i in range(m):
                for j in range(n):
                    result[:, i, j] = np.array([func(cls, x[i, j], y[i, j], z[i, j])])
            return result

        raise MagnetostaticsError(
            "This operation only supports floats and 1-D and 2-D arrays."
        )

    return wrapper


def process_coords_array(shape: Union[np.ndarray, Coordinates]) -> np.ndarray:
    """
    Parse Coordinates or array to an array.

    Parameters
    ----------
    shape:
        The Coordinates or array to make into a coordinate array

    Returns
    -------
    Array in proper dimensions
    """
    if isinstance(shape, np.ndarray):
        pass

    elif isinstance(shape, Coordinates):
        shape = shape.T

    else:
        raise MagnetostaticsError(
            f"Cannot make a CurrentSource from an object of type: {type(shape)}."
        )

    return shape


def process_to_coordinates(shape: Union[np.ndarray, dict, Coordinates]) -> Coordinates:
    """
    Parse input to Coordinates

    Raises
    ------
    CoordinatesError: if the type could not be parsed, or if the input was bad.
    """
    if isinstance(shape, Coordinates):
        return shape
    return Coordinates(shape)


def jit_llc7(f_integrand: Callable) -> LowLevelCallable:
    """
    Decorator for 6-argument integrand function to a low-level callable.

    Parameters
    ----------
    f_integrand:
        The integrand function

    Returns
    -------
    The decorated integrand function as a LowLevelCallable
    """
    f_jitted = nb.jit(f_integrand, nopython=True, cache=True)

    @nb.cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):  # noqa: ARG001
        return f_jitted(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6])

    return LowLevelCallable(wrapped.ctypes)


def jit_llc5(f_integrand: Callable) -> LowLevelCallable:
    """
    Decorator for 4-argument integrand function to a low-level callable.

    Parameters
    ----------
    f_integrand:
        The integrand function

    Returns
    -------
    The decorated integrand function as a LowLevelCallable
    """
    f_jitted = nb.jit(f_integrand, nopython=True, cache=True)

    @nb.cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):  # noqa: ARG001
        return f_jitted(xx[0], xx[1], xx[2], xx[3], xx[4])

    return LowLevelCallable(wrapped.ctypes)


def jit_llc4(f_integrand: Callable) -> LowLevelCallable:
    """
    Decorator for 3-argument integrand function to a low-level callable.

    Parameters
    ----------
    f_integrand:
        The integrand function

    Returns
    -------
    The decorated integrand function as a LowLevelCallable
    """
    f_jitted = nb.jit(f_integrand, nopython=True, cache=True)

    @nb.cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):  # noqa: ARG001
        return f_jitted(xx[0], xx[1], xx[2], xx[3])

    return LowLevelCallable(wrapped.ctypes)


def jit_llc3(f_integrand: Callable) -> LowLevelCallable:
    """
    Decorator for 2-argument integrand function to a low-level callable.

    Parameters
    ----------
    f_integrand:
        The integrand function

    Returns
    -------
    The decorated integrand function as a LowLevelCallable
    """
    f_jitted = nb.jit(f_integrand, nopython=True, cache=True)

    @nb.cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):  # noqa: ARG001
        return f_jitted(xx[0], xx[1], xx[2])

    return LowLevelCallable(wrapped.ctypes)


def integrate(func: Callable, args: Iterable, bound1: float, bound2: float) -> float:
    """
    Utility for integration of a function between bounds. Easier to refactor
    integration methods.

    Parameters
    ----------
    func:
        The function to integrate. The integration variable should be the last
        argument of this function.
    args:
        The iterable of static arguments to the function.
    bound1:
        The lower integration bound
    bound2:
        The upper integration bound

    Returns
    -------
    The value of the integral of the function between the bounds
    """
    warnings.filterwarnings("error", category=IntegrationWarning)
    try:
        result = quad(func, bound1, bound2, args=args)[0]
    except IntegrationWarning:
        # First attempt at fixing the integration problem
        points = [
            0.25 * (bound2 - bound1),
            0.5 * (bound2 - bound1),
            0.75 * (bound2 - bound1),
        ]
        try:
            result = quad(func, bound1, bound2, args=args, points=points, limit=200)[0]
        except IntegrationWarning as error:
            raise MagnetostaticsIntegrationError from error

    warnings.filterwarnings("default", category=IntegrationWarning)
    return result


def n_integrate(
    func: Callable, args: Iterable, bounds: List[Iterable[Union[int, float]]]
) -> float:
    """
    Utility for n-dimensional integration of a function between bounds. Easier
    to refactor integration methods.

    Parameters
    ----------
    func:
        The function to integrate. The integration variable should be the last
        argument of this function.
    args:
        The iterable of static arguments to the function.
    bounds:
        The list of lower and upper integration bounds applied to x[0], x[1], ..

    Returns
    -------
    The value of the integral of the function between the bounds
    """
    return nquad(func, bounds, args=args)[0]
