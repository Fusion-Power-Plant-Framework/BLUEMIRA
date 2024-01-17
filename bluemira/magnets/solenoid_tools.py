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
Tools for simple solenoid calculations.
"""

from typing import Tuple

import numpy as np

from bluemira.magnetostatics.semianalytic_2d import semianalytic_Bx, semianalytic_Bz


def calculate_B_max(
    rho_j: float, r_inner: float, r_outer: float, height: float, z_0: float = 0.0
) -> float:
    """
    Calculate the maximum self-field in a solenoid. This is always located
    at (r_inner, z_0)

    Parameters
    ----------
    rho_j:
        Current density across the solenoid winding pack [A/m^2]
    r_inner:
        Solenoid inner radius [m]
    r_outer:
        Solenoid outer radius [m]
    height:
        Solenoid vertical extent [m]

    Returns
    -------
    Maximum field in a solenoid [T]

    Notes
    -----
    Cross-checked graphically with k data from Boom and Livingstone, "Superconducting
    solenoids", 1962, Fig. 6
    """
    dxc = 0.5 * (r_outer - r_inner)
    xc = r_inner + dxc
    dzc = 0.5 * height
    x_bmax = r_inner
    current = rho_j * (height * (r_outer - r_inner))
    Bx_max = current * semianalytic_Bx(xc, z_0, x_bmax, z_0, dxc, dzc)
    Bz_max = current * semianalytic_Bz(xc, z_0, x_bmax, z_0, dxc, dzc)
    return np.hypot(Bx_max, Bz_max)


def calculate_hoop_radial_stress(
    B_in: float,
    B_out: float,
    rho_j: float,
    r_inner: float,
    r_outer: float,
    r: float,
    poisson_ratio: float = 0.3,
) -> Tuple[float]:
    """
    Calculate the hoop and radial stress at a radial location in a solenoid

    Parameters
    ----------
    B_in:
        Field at the inside edge of the solenoid [T]
    B_out:
        Field at the outside edge of the solenoid [T]
    rho_j:
        Current density across the solenoid winding pack [A/m^2]
    r_inner:
        Solenoid inner radius [m]
    r_outer:
        Solenoid outer radius [m]
    r:
        Radius at which to calculate [m]
    poisson_ratio:
        Poisson ratio of the material

    Returns
    -------
    hoop_stress:
        Hoop stress at the radial location [Pa]
    radial_stress:
        Radial stress at the radial location [Pa]

    Notes
    -----
    Wilson, Superconducting Magnets, 1982, equations 4.10 and 4.11
    Must still factor in the fraction of load-bearing material
    """
    alpha = r_outer / r_inner
    eps = r / r_inner
    nu = poisson_ratio
    alpha2 = alpha**2
    eps2 = eps**2
    ratio2 = alpha2 / eps2

    K = (alpha * B_in - B_out) * rho_j * r_inner / (alpha - 1)  # noqa: N806
    M = (B_in - B_out) * rho_j * r_inner / (alpha - 1)  # noqa: N806
    a = K * (2 + nu) / (3 * (alpha + 1))
    c = M * (3 + nu) / 8

    b = alpha2 + alpha + 1
    b1 = ratio2 - eps * (1 + 2 * nu) * (alpha + 1) / (2 + nu)
    b2 = -ratio2 - eps * (alpha + 1)

    d = alpha2 + 1 + ratio2 - eps2 * (1 + 3 * nu) / (3 + nu)
    e = alpha2 + 1 - ratio2 - eps2

    hoop_stress = a * (b + b1) - c * d
    radial_stress = a * (b + b2) - c * e

    return hoop_stress, radial_stress


def calculate_flux_max(B_max: float, r_inner: float, r_outer: float) -> float:
    """
    Calculate the maximum flux achievable from a solenoid

    Parameters
    ----------
    B_max:
        Maximum field in the solenoid [T]
    r_inner:
        Solenoid inner radius [m]
    r_outer:
        Solenoid outer radius [m]

    Returns
    -------
    Maximum flux achievable from a solenoid [V.s]
    """
    return np.pi / 3 * B_max * (r_outer**2 + r_inner**2 + r_outer * r_inner)
