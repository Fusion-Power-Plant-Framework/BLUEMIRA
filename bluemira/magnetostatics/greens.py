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
Green's functions mappings for psi, Bx, and Bz
"""
from typing import Tuple, Union

import numba as nb
import numpy as np
from scipy.special import ellipe, ellipk

from bluemira.base.constants import MU_0, MU_0_2PI, MU_0_4PI

__all__ = [
    "greens_psi",
    "greens_Bx",
    "greens_Bz",
    "greens_dpsi_dx",
    "greens_dpsi_dz",
    "greens_all",
]

# Offset from 0<x<1
#     Used in calculating Green's functions to avoid np.nan
GREENS_ZERO = 1e-8


@nb.vectorize(nopython=True, cache=True)
def clip_nb(
    val: Union[float, np.ndarray], val_min: float, val_max: float
) -> Union[float, np.ndarray]:
    """
    Clips (limits) val between val_min and val_max. Vectorised for speed and
    compatibility with numba.

    Parameters
    ----------
    val:
        The value to be clipped.
    val_min:
        The minimum value.
    val_max:
        The maximum value.

    Returns
    -------
    The clipped values.
    """
    if val < val_min:
        return val_min
    if val > val_max:
        return val_max
    return val


@nb.vectorize(nopython=True)
def ellipe_nb(k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Vectorised scipy ellipe

    Notes
    -----
    K, E in scipy are set as K(k^2), E(k^2)
    """
    return ellipe(k)


ellipe_nb.__doc__ += ellipe.__doc__


@nb.vectorize(nopython=True)
def ellipk_nb(k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Vectorised scipy ellipk

    Notes
    -----
    K, E in scipy are set as K(k^2), E(k^2)
    """
    return ellipk(k)


ellipk_nb.__doc__ += ellipk.__doc__


def circular_coil_inductance_elliptic(radius: float, rc: float) -> float:
    """
    Calculate the inductance of a circular coil by elliptic integrals.

    Parameters
    ----------
    radius:
        The radius of the circular coil
    rc:
        The radius of the coil cross-section

    Returns
    -------
    The self-inductance of the circular coil [H]
    """
    k = 4 * radius * (radius - rc) / (2 * radius - rc) ** 2
    args = (
        np.array(arg, dtype=np.float64) for arg in (k, GREENS_ZERO, 1.0 - GREENS_ZERO)
    )
    k = clip_nb(*args)
    return MU_0 * (2 * radius - rc) * ((1 - k**2 / 2) * ellipk(k) - ellipe(k))


def circular_coil_inductance_kirchhoff(radius: float, rc: float) -> float:
    """
    Calculate the inductance of a circular coil by Kirchhoff's approximation.

    radius:
        The radius of the circular coil
    rc:
        The radius of the coil cross-section

    Returns
    -------
    The self-inductance of the circular coil [H]
    """
    return MU_0 * radius * (np.log(8 * radius / rc) - 2 + 0.25)


@nb.jit(nopython=True)
def greens_psi(
    xc: Union[float, np.ndarray],
    zc: Union[float, np.ndarray],
    x: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    d_xc: float = 0,  # noqa: ARG001
    d_zc: float = 0,  # noqa: ARG001
) -> Union[float, np.ndarray]:
    """
    Calculate poloidal flux at (x, z) due to a unit current at (xc, zc)
    using a Greens function.

    Parameters
    ----------
    xc:
        Coil x coordinates [m]
    zc:
        Coil z coordinates [m]
    x:
        Calculation x locations
    z:
        Calculation z locations
    d_xc:
        The coil half-width (overload argument)
    d_zc:
        The coil half-height (overload argument)

    Returns
    -------
    Poloidal magnetic flux per radian response at (x, z)

    Raises
    ------
    ZeroDivisionError
        if xc <= 0

    Notes
    -----
    \t:math:`G_{\\psi}(x_{c}, z_{c}; x, z) = \\dfrac{{\\mu}_{0}}{2{\\pi}}`
    \t:math:`\\dfrac{\\sqrt{xx_{c}}}{k}`
    \t:math:`[(2-\\mathbf{K}(k^2)-2\\mathbf{E}(k^2)]`\n
    Where:
    \t:math:`k^{2}\\equiv\\dfrac{4xx_{c}}{(x+x_{c})^{2}+(z-z_{c})^{2}}`\n
    \t:math:`\\mathbf{K} \\equiv` complete elliptic integral of the first kind\n
    \t:math:`\\mathbf{E} \\equiv` complete elliptic integral of the second kind
    """
    k2 = 4 * x * xc / ((x + xc) ** 2 + (z - zc) ** 2)
    # Avoid NaN when coil on grid point
    k2 = clip_nb(k2, GREENS_ZERO, 1.0 - GREENS_ZERO)
    return (
        MU_0_2PI
        * np.sqrt(x * xc)
        * ((2 - k2) * ellipk_nb(k2) - 2 * ellipe_nb(k2))
        / np.sqrt(k2)
    )


@nb.jit(nopython=True)
def greens_dpsi_dx(
    xc: Union[float, np.ndarray],
    zc: Union[float, np.ndarray],
    x: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    d_xc: float = 0,  # noqa: ARG001
    d_zc: float = 0,  # noqa: ARG001
) -> Union[float, np.ndarray]:
    """
    Calculate the radial derivative of the poloidal flux at (x, z)
    due to a unit current at (xc, zc) using a Greens function.

    Parameters
    ----------
    xc:
        Coil x coordinates [m]
    zc:
        Coil z coordinates [m]
    x:
        Calculation x locations
    z:
        Calculation z locations
    d_xc:
        The coil half-width (overload argument)
    d_zc:
        The coil half-height (overload argument)

    Returns
    -------
    Radial derivative of the poloidal flux response at (x, z)

    Notes
    -----
    \t:math:`G_{\\dfrac{\\partial \\psi}{\\partial x}}(x_{c}, z_{c}; x, z) =`
    \t:math:`\\dfrac{\\mu_0}{2\\pi}`
    \t:math:`\\dfrac{1}{u}`
    \t:math:`[\\dfrac{w^2}{d^2}\\mathbf{E}(k^2)+\\mathbf{K}(k^2)]`\n
    Where:
    \t:math:`h^{2}\\equiv z_{c}-z`\n
    \t:math:`u^2\\equiv(x+x_{c})^2+h^2`\n
    \t:math:`d^{2}\\equiv (x - x_{c})^2 + h^2`\n
    \t:math:`w^{2}\\equiv x^2 -x_{c}^2 - h^2`\n
    \t:math:`k^{2}\\equiv\\dfrac{4xx_{c}}{(x+x_{c})^{2}+(z-z_{c})^{2}}`\n
    \t:math:`\\mathbf{K} \\equiv` complete elliptic integral of the first kind\n
    \t:math:`\\mathbf{E} \\equiv` complete elliptic integral of the second kind

    The implementation used here refactors the above to avoid some zero divisions.
    """
    a = ((x + xc) ** 2 + (z - zc) ** 2) ** 0.5
    k2 = 4 * x * xc / a**2
    # Avoid NaN when coil on grid point
    k2 = clip_nb(k2, GREENS_ZERO, 1.0 - GREENS_ZERO)
    i1 = ellipk_nb(k2) / a
    i2 = ellipe_nb(k2) / (a**3 * (1 - k2))
    return MU_0_2PI * x * ((xc**2 - (z - zc) ** 2 - x**2) * i2 + i1)


@nb.jit(nopython=True)
def greens_dpsi_dz(
    xc: Union[float, np.ndarray],
    zc: Union[float, np.ndarray],
    x: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    d_xc: float = 0,  # noqa: ARG001
    d_zc: float = 0,  # noqa: ARG001
) -> Union[float, np.ndarray]:
    """
    Calculate the vertical derivative of the poloidal flux at (x, z)
    due to a unit current at (xc, zc) using a Greens function.

    Parameters
    ----------
    xc:
        Coil x coordinates [m]
    zc:
        Coil z coordinates [m]
    x:
        Calculation x locations
    z:
        Calculation z locations
    d_xc:
        The coil half-width (overload argument)
    d_zc:
        The coil half-height (overload argument)

    Returns
    -------
    Vertical derivative of the poloidal flux response at (x, z)

    Notes
    -----
    \t:math:`G_{\\dfrac{\\partial \\psi}{\\partial z}}(x_{c}, z_{c}; x, z) =`
    \t:math:`\\dfrac{\\mu_0}{2\\pi}`
    \t:math:`\\dfrac{h}{u}`
    \t:math:`[\\mathbf{K}(k^2) - \\dfrac{v^2}{d^2}\\mathbf{E}(k^2)]`\n
    Where:
    \t:math:`h^{2}\\equiv z_{c}-z`\n
    \t:math:`u^2\\equiv(x+x_{c})^2+h^2`\n
    \t:math:`d^{2}\\equiv (x - x_{c})^2 + h^2`\n
    \t:math:`v^{2}\\equiv x^2 +x_{c}^2 + h^2`\n
    \t:math:`k^{2}\\equiv\\dfrac{4xx_{c}}{(x+x_{c})^{2}+(z-z_{c})^{2}}`\n
    \t:math:`\\mathbf{K} \\equiv` complete elliptic integral of the first kind\n
    \t:math:`\\mathbf{E} \\equiv` complete elliptic integral of the second kind

    The implementation used here refactors the above to avoid some zero divisions.
    """
    a = ((x + xc) ** 2 + (z - zc) ** 2) ** 0.5
    k2 = 4 * x * xc / a**2
    k2 = clip_nb(k2, GREENS_ZERO, 1.0 - GREENS_ZERO)
    i1 = ellipk_nb(k2) / a
    i2 = ellipe_nb(k2) / (a**3 * (1 - k2))
    return MU_0_2PI * ((z - zc) * (i1 - i2 * ((z - zc) ** 2 + x**2 + xc**2)))


@nb.jit(nopython=True)
def greens_Bx(
    xc: Union[float, np.ndarray],
    zc: Union[float, np.ndarray],
    x: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    d_xc: float = 0,
    d_zc: float = 0,
) -> Union[float, np.ndarray]:
    """
    Calculate radial magnetic field at (x, z) due to unit current at (xc, zc)
    using a Greens function.

    Parameters
    ----------
    xc:
        Coil x coordinates [m]
    zc:
        Coil z coordinates [m]
    x:
        Calculation x locations
    z:
        Calculation z locations
    d_xc:
        The coil half-width (overload argument)
    d_zc:
        The coil half-height (overload argument)

    Returns
    -------
    Radial magnetic field response at (x, z)

    Raises
    ------
    ZeroDivisionError
        if x == 0

    Notes
    -----
    \t:math:`G_{B_{x}}(x_{c}, z_{c}; x, z) = -\\dfrac{1}{x}`
    \t:math:`G_{\\dfrac{\\partial \\psi}{\\partial z}}`
    \t:math:`(x_{c}, z_{c}; x, z)`
    """
    return -1 / x * greens_dpsi_dz(xc, zc, x, z, d_xc, d_zc)


@nb.jit(nopython=True)
def greens_Bz(
    xc: Union[float, np.ndarray],
    zc: Union[float, np.ndarray],
    x: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    d_xc: float = 0,
    d_zc: float = 0,
) -> Union[float, np.ndarray]:
    """
    Calculate vertical magnetic field at (x, z) due to unit current at (xc, zc)
    using a Greens function.

    Parameters
    ----------
    xc:
        Coil x coordinates [m]
    zc:
        Coil z coordinates [m]
    x:
        Calculation x locations
    z:
        Calculation z locations
    d_xc:
        The coil half-width (overload argument)
    d_zc:
        The coil half-height (overload argument)

    Returns
    -------
    Vertical magnetic field response at (x, z)

    Raises
    ------
    ZeroDivisionError
        if x == 0

    Notes
    -----
    \t:math:`G_{B_{z}}(x_{c}, z_{c}; x, z) = \\dfrac{1}{x}`
    \t:math:`G_{\\dfrac{\\partial \\psi}{\\partial x}}`
    \t:math:`(x_{c}, z_{c}; x, z)`
    """
    return 1 / x * greens_dpsi_dx(xc, zc, x, z, d_xc, d_zc)


@nb.jit(nopython=True)
def greens_all(
    xc: Union[float, np.ndarray],
    zc: Union[float, np.ndarray],
    x: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Speed optimisation of Green's functions for psi, Bx, and Bz

    Parameters
    ----------
    xc:
        Coil x coordinates [m]
    zc:
        Coil z coordinates [m]
    x:
        Calculation x locations
    z:
        Calculation z locations

    Returns
    -------
    psi:
        Poloidal magnetic flux per radian response at (x, z)
    Bx:
        Radial magnetic field response at (x, z)
    Bz:
        Vertical magnetic field response at (x, z)

    Raises
    ------
    ZeroDivisionError
        if xc <= 0
        if x <= 0
    """
    a = np.hypot((x + xc), (z - zc))
    k2 = 4 * x * xc / a**2
    # Avoid NaN when coil on grid point
    k2 = clip_nb(k2, GREENS_ZERO, 1.0 - GREENS_ZERO)
    e, k = ellipe_nb(k2), ellipk_nb(k2)
    i_1 = 4 * k / a
    i_2 = 4 * e / (a**3 * (1 - k2))
    a_part = (z - zc) ** 2 + x**2 + xc**2
    b_part = -2 * x * xc
    g_bx = MU_0_4PI * xc * (z - zc) * (i_1 - i_2 * a_part) / b_part
    g_bz = MU_0_4PI * xc * ((xc + x * a_part / b_part) * i_2 - i_1 * x / b_part)
    g_psi = MU_0_4PI * a * ((2 - k2) * k - 2 * e)
    return g_psi, g_bx, g_bz
