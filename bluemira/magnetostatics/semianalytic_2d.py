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
Semi-analytic methods for Bx, Bz, and psi for circular coils of rectangular
cross-section.
"""
from typing import Tuple, Union

import numba as nb
import numpy as np

from bluemira.base.constants import EPS
from bluemira.magnetostatics.error import MagnetostaticsIntegrationError
from bluemira.magnetostatics.tools import (
    integrate,
    jit_llc3,
    jit_llc5,
    jit_llc7,
    n_integrate,
)
from bluemira.utilities.tools import is_num

__all__ = ["semianalytic_Bx", "semianalytic_Bz", "semianalytic_psi"]


@nb.jit(nopython=True, cache=True)
def _partial_x_integrand(phi: float, rr: float, zz: float) -> float:
    """
    Integrand edge cases derived to constant integrals. Much faster than
    splitting up the integrands.
    """
    cos_phi = np.cos(phi)
    r0 = np.sqrt(rr**2 + 1 - 2 * rr * cos_phi + zz**2)

    if abs(zz) < EPS:
        if abs(rr - 1.0) < EPS:
            return -1.042258937608 / np.pi
        if rr < 1.0:  # noqa: PLR2004
            return cos_phi * (
                r0 + cos_phi * np.log((r0 + 1 + rr) / (r0 + 1 - rr))
            ) - 0.25 * (1 + np.log(4))

    return (r0 + cos_phi * np.log(r0 + rr - cos_phi)) * cos_phi


@jit_llc5
def _full_x_integrand(phi: float, r1: float, r2: float, z1: float, z2: float) -> float:
    """
    Calculate the P_x primitive integral.

    \t:math:`P_{x}(R, Z) = \\int_{0}^{\\pi}[R_{0}+cos(\\phi)ln(R_{0}+R`
    \t:math:`-cos(\\phi))]cos(\\phi)d\\phi`
    """
    return (
        _partial_x_integrand(phi, r1, z1)
        - _partial_x_integrand(phi, r1, z2)
        - _partial_x_integrand(phi, r2, z1)
        + _partial_x_integrand(phi, r2, z2)
    )


def _partial_z_integrand_nojit(phi: float, rr: float, zz: float) -> float:
    """
    Integrand edge cases derived to constant integrals. Much faster than
    splitting up the integrands.
    """
    if abs(zz) < EPS:
        return 0.0

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    r0 = np.sqrt(rr**2 + 1 - 2 * rr * cos_phi + zz**2)

    # F1
    result = zz * np.log(r0 + rr - cos_phi) - cos_phi * np.log(r0 + zz)

    # F2
    result = result - 0.5 * rr if rr - 1 < EPS else result - 0.5 / rr
    # F3
    if 0.5 * np.pi * sin_phi > 1e-9:  # noqa: PLR2004
        result = result - sin_phi * np.arctan(zz * (rr - cos_phi) / (r0 * sin_phi))
    return result


_partial_z_integrand = nb.jit(_partial_z_integrand_nojit, nopython=True, cache=True)
_partial_z_integrand_llc = jit_llc3(_partial_z_integrand_nojit)


@jit_llc5
def _full_z_integrand(phi: float, r1: float, r2: float, z1: float, z2: float) -> float:
    """
    Calculate the P_z primitive integral at all 4 corner combinations

    \t:math:`P_{z}(R, Z) = \\int_{0}^{\\pi} [Zln(R_{0}+R-cos(\\phi)`
    \t:math:`+\\dfrac{1}{2}cos(\\phi)ln(\\dfrac{R_{0}-Z}{R_{0}+Z})`
    \t:math:`-sin(\\phi)arctan(\\dfrac{Z[R-cos(\\phi)]}{R_{0}sin(\\phi)})]d\\phi`
    """
    return (
        _partial_z_integrand(phi, r1, z1)
        - _partial_z_integrand(phi, r1, z2)
        - _partial_z_integrand(phi, r2, z1)
        + _partial_z_integrand(phi, r2, z2)
    )


def _integrate_z_by_parts(r1: float, r2: float, z1: float, z2: float) -> float:
    """
    Integrate the Bz integrand by parts.

    This can be used as a fallback if the full integration fails.
    """
    return (
        integrate(_partial_z_integrand_llc, (r1, z1), 0, np.pi)
        - integrate(_partial_z_integrand_llc, (r1, z2), 0, np.pi)
        - integrate(_partial_z_integrand_llc, (r2, z1), 0, np.pi)
        + integrate(_partial_z_integrand_llc, (r2, z2), 0, np.pi)
    )


@nb.jit(nopython=True, cache=True)
def _get_working_coords(
    xc: float, zc: float, x: float, z: float, d_xc: float, d_zc: float
) -> Tuple[float, float, float, float, float]:
    """
    Convert coil and global coordinates to working coordinates.
    """
    z = z - zc
    r1, r2 = (xc - d_xc) / x, (xc + d_xc) / x
    z1, z2 = (-d_zc - z) / x, (d_zc - z) / x
    j_tor = 1 / (4 * d_xc * d_zc)  # Keep current out of the equation
    return r1, r2, z1, z2, j_tor


def _array_dispatcher(func):
    """
    Decorator for float and array handling.
    """

    def wrapper(xc, zc, x, z, d_xc, d_zc):
        # Handle floats
        if is_num(x):
            return func(xc, zc, x, z, d_xc, d_zc)

        # Handle arrays
        if len(x.shape) == 1:
            if not isinstance(xc, np.ndarray) or len(xc.shape) == 1:
                result = np.zeros(len(x))
                for i in range(len(x)):
                    result[i] = func(xc, zc, x[i], z[i], d_xc, d_zc)
            else:
                result = np.zeros((len(x), len(xc)))
                for j in range(xc.shape[1]):
                    for i in range(len(x)):
                        result[i, j] = func(
                            xc[:, j], zc[:, j], x[i], z[i], d_xc[:, j], d_zc[:, j]
                        )
        # 2-D arrays
        elif not isinstance(xc, np.ndarray) or len(xc.shape) == 1:
            result = np.zeros(x.shape)
            for i in range(x.shape[0]):
                for j in range(z.shape[1]):
                    result[i, j] = func(xc, zc, x[i, j], z[i, j], d_xc, d_zc)
        else:
            result = np.zeros([*list(x.shape), xc.shape[1]])
            for k in range(xc.shape[1]):
                for i in range(x.shape[0]):
                    for j in range(z.shape[1]):
                        result[i, j, ..., k] = func(
                            xc[:, k],
                            zc[:, k],
                            x[i, j],
                            z[i, j],
                            d_xc[:, k],
                            d_zc[:, k],
                        )
        return result

    return wrapper


@_array_dispatcher
def semianalytic_Bx(
    xc: float,
    zc: float,
    x: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    d_xc: float,
    d_zc: float,
) -> Union[float, np.ndarray]:
    """
    Calculate the Bx and Bz fields from a rectangular cross-section circular
    coil with a unit current using a semi-analytic reduction of the Biot-Savart
    law.

    Parameters
    ----------
    xc:
        Coil x coordinate [m]
    zc:
        Coil z coordinate [m]
    x:
        Calculation x location
    z:
        Calculation z location
    d_xc:
        The half-width of the coil
    d_zc:
        The half-height of the coil

    Returns
    -------
    Radial magnetic field response (x, z)

    Notes
    -----
    \t:math:`B_{x}=\\dfrac{\\mu_{0}Jx}{2\\pi}\\sum^{2}_{i=1}(-1)^{i+j}`
    \t:math:`P_x(R_{i},Z_{j})`

    References
    ----------
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6019053
    """
    r1, r2, z1, z2, j_tor = _get_working_coords(xc, zc, x, z, d_xc, d_zc)

    Bx = integrate(_full_x_integrand, (r1, r2, z1, z2), 0, np.pi)

    fac = 2e-7 * j_tor * x  # MU_0/(2*np.pi)
    return fac * Bx


@_array_dispatcher
def semianalytic_Bz(
    xc: float,
    zc: float,
    x: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    d_xc: float,
    d_zc: float,
) -> Union[float, np.ndarray]:
    """
    Calculate the Bx and Bz fields from a rectangular cross-section circular
    coil with a unit current using a semi-analytic reduction of the Biot-Savart
    law.

    Parameters
    ----------
    xc:
        Coil x coordinate [m]
    zc:
        Coil z coordinate [m]
    x:
        Calculation x location
    z:
        Calculation z location
    d_xc:
        The half-width of the coil
    d_zc:
        The half-height of the coil

    Returns
    -------
    Vertical magnetic field response at (x, z)

    Notes
    -----
    \t:math:`B_{z}=\\dfrac{\\mu_{0}Jx}{2\\pi}\\sum^{2}_{i=1}(-1)^{i+j}`
    \t:math:`P_z(R_{i},Z_{j})`

    References
    ----------
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6019053
    """
    r1, r2, z1, z2, j_tor = _get_working_coords(xc, zc, x, z, d_xc, d_zc)

    try:
        Bz = integrate(_full_z_integrand, (r1, r2, z1, z2), 0, np.pi)
    except MagnetostaticsIntegrationError:
        # If all else fails, fall back to integration by parts
        Bz = _integrate_z_by_parts(r1, r2, z1, z2)
    fac = 2e-7 * j_tor * x  # MU_0/(2*np.pi)
    return fac * Bz


@jit_llc7
def _full_psi_integrand(x, phi, xc, zc, z, d_xc, d_zc):
    """
    Integrand for psi = xBz
    """
    z = z - zc
    r1, r2 = (xc - d_xc) / x, (xc + d_xc) / x
    z1, z2 = (-d_zc - z) / x, (d_zc - z) / x
    return x**2 * (
        _partial_z_integrand(phi, r1, z1)
        - _partial_z_integrand(phi, r1, z2)
        - _partial_z_integrand(phi, r2, z1)
        + _partial_z_integrand(phi, r2, z2)
    )


@_array_dispatcher
def semianalytic_psi(
    xc: float,
    zc: float,
    x: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    d_xc: float,
    d_zc: float,
) -> Union[float, np.ndarray]:
    """
    Calculate the poloidal magnetic flux from a rectangular cross-section circular
    coil with a unit current using a semi-analytic reduction of the Biot-Savart
    law.

    Parameters
    ----------
    xc:
        Coil x coordinate [m]
    zc:
        Coil z coordinate [m]
    x:
        Calculation x location
    z:
        Calculation z location
    d_xc:
        The half-width of the coil
    d_zc:
        The half-height of the coil

    Returns
    -------
    Poloidal magnetic flux response at (x, z)

    Notes
    -----
    Integrates x*Bz to resolve psi. More analytical approaches are possible and
    will no doubt be faster.
    """
    j_tor = 1 / (4 * d_xc * d_zc)  # Keep current out of the equation
    psi = n_integrate(_full_psi_integrand, (xc, zc, z, d_xc, d_zc), [[0, x], [0, np.pi]])

    return 2e-7 * j_tor * psi  # MU_0/(2*np.pi)
