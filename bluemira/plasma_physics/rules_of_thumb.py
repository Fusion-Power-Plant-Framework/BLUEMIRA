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
A collection of simple 0-D rules of thumb for tokamak plasmas.
"""

import numpy as np

from bluemira.base.constants import EV_TO_J, K_BOLTZMANN, MU_0
from bluemira.plasma_physics.collisions import coulomb_logarithm, spitzer_conductivity


def estimate_loop_voltage(
    R_0: float, B_t: float, Z_eff: float, T_e: float, n_e: float, q_0: float
) -> float:
    """
    A 0-D estimate of the loop voltage during burn

    Parameters
    ----------
    R_0:
        Major radius [m]
    B_t:
        Toroidal field on axis [T]
    Z_eff:
        Effective charge [a.m.u.]
    T_e:
        Electron temperature on axis [eV]
    n_e:
        Electron density [1/m^3]
    q_0:
        Safety factor on axis

    Returns
    -------
    Loop voltage during burn [V]

    Notes
    -----
    H. Zohm, W. Morris (2022)

    \t:math:`v_{loop}=2\\pi R_{0}\\dfrac{2\\pi B_{t}}{\\mu_{0}q_{0}\\sigma_{0}R_{0}}`

    where :math:`\\sigma_{0}` is the Spitzer conductivity on axis:
    \t:math:`\\sigma_{0} = 1.92e4 (2-Z_{eff}^{-1/3}) \\dfrac{T_{e}^{3/2}}{Z_{eff}ln\\Lambda}`

    Assumes no non-inductive current on axis

    Assumes a circular cross-section on axis

    There is no neo-classical resistivity on axis because there are no trapped particles
    """  # noqa: W505, E501
    ln_lambda = coulomb_logarithm(T_e * EV_TO_J / K_BOLTZMANN, n_e)
    sigma = spitzer_conductivity(Z_eff, T_e, ln_lambda)

    # Current density on axis
    j_0 = 2 * B_t / (MU_0 * q_0 * R_0)
    return 2 * np.pi * R_0 * j_0 / sigma


def estimate_Le(A: float, kappa: float) -> float:  # noqa: N802
    """
    Estimate the normalised external plasma self-inductance.

    Parameters
    ----------
    A:
        Last closed flux surface aspect ratio
    kappa:
        Last closed flux surface elongation

    Returns
    -------
    Normalised plasma external inductance

    Notes
    -----
    Hirshman and Neilson, 1986
    https://pubs.aip.org/aip/pfl/article/29/3/790/944223/External-inductance-of-an-axisymmetric-plasma
    Assuming a LCFS parameterisation as per:
    :py:func:`bluemira.equilibria.shapes.flux_surface_hirshman`
    """
    eps = 1 / A
    sqrt_eps = np.sqrt(eps)

    a = (
        (1 + 1.81 * sqrt_eps + 2.05 * eps) * np.log(8 * A)
        - 2.0
        - 9.25 * sqrt_eps
        + 1.21 * eps
    )
    b = 0.73 * sqrt_eps * (1 + 2 * eps**4 - 6 * eps**5 + 3.7 * eps**6)
    return a * (1 - eps) / (1 - eps + b * kappa)


def estimate_M(A: float, kappa: float) -> float:  # noqa: N802
    """
    Estimate the plasma mutual inductance.

    Parameters
    ----------
    A:
        Last closed flux surface aspect ratio
    kappa:
        Last closed flux surface elongation

    Returns
    -------
    Plasma mutual inductance

    Notes
    -----
    Hirshman and Neilson, 1986
    https://pubs.aip.org/aip/pfl/article/29/3/790/944223/External-inductance-of-an-axisymmetric-plasma
    Assuming a LCFS parameterisation as per:
    :py:func:`bluemira.equilibria.shapes.flux_surface_hirshman`
    """
    eps = 1 / A

    c = 1 + 0.98 * eps**2 + 0.49 * eps**4 + 1.47 * eps**6
    d = 0.25 * eps * (1 + 0.84 * eps - 1.44 * eps**2)
    return (1 - eps) ** 2 / ((1 - eps) ** 2 * c + d * np.sqrt(kappa))


def calc_cyl_safety_factor(R_0: float, A: float, B_0: float, I_p: float) -> float:
    """
    Calculate the cylindrical safety factor.

    Parameters
    ----------
    R_0:
        Plasma major radius [m]
    A:
        Plasma aspect ratio
    B_0:
        Toroidal field at major radius [T]
    I_p:
        Plasma current [A]

    Returns
    -------
    Cylindrical safety factor

    Notes
    -----
    Sometimes also written with :math:`\\dfrac{2\\pi}{\\mu_{0}} = 5` and I_p in [MA]
    """
    a = R_0 / A
    return 2 * np.pi * a**2 * B_0 / (MU_0 * R_0 * I_p)


def calc_qstar_freidberg(
    R_0: float, A: float, B_0: float, I_p: float, kappa: float
) -> float:
    """
    Calculate the kink safety factor at the plasma edge

    \t:math:`q_{*}=\\dfrac{2\\pi a^2 B_0}{\\mu_0 R_0 I_p}`
    \t:math:`\\bigg(\\dfrac{1+\\kappa^2}{2}\\bigg)`

    Parameters
    ----------
    R_0:
        Plasma major radius [m]
    A:
        Plasma aspect ratio
    B_0:
        Toroidal field at major radius [T]
    I_p:
        Plasma current [A]
    kappa:
        Plasma elongation

    Returns
    -------
    Kink safety factor

    Notes
    -----
    Freidberg, Ideal MHD, p 131
    """
    shape_factor = 0.5 * (1 + kappa**2)
    q_cyl = calc_cyl_safety_factor(R_0, A, B_0, I_p)
    return q_cyl * shape_factor


def calc_qstar_uckan(
    R_0: float, A: float, B_0: float, I_p: float, kappa: float, delta: float
) -> float:
    """
    Calculate the cylindrical equivalent safety factor at the plasma edge

    Parameters
    ----------
    R_0:
        Plasma major radius [m]
    A:
        Plasma aspect ratio
    B_0:
        Toroidal field at major radius [T]
    I_p:
        Plasma current [A]
    kappa:
        Plasma elongation
    delta:
        Plasma triangularity

    Returns
    -------
    Cylindrical equivalent safety factor

    Notes
    -----
    Uckan et al., ITER Physics Design Guidelines, 1989, sec. 2.3
    https://inis.iaea.org/search/search.aspx?orig_q=RN:21068960
    """
    shape_factor = 0.5 + 0.5 * kappa**2 * (1 + 2 * delta**2 - 1.2 * delta**3)
    q_cyl = calc_cyl_safety_factor(R_0, A, B_0, I_p)
    return q_cyl * shape_factor


def estimate_q95_uckan(
    R_0: float, A: float, B_0: float, I_p: float, kappa: float, delta: float
) -> float:
    """
    Estimate safety factor at the 95th percentile flux surface based on an empirical fit.

    Parameters
    ----------
    R_0:
        Plasma major radius [m]
    A:
        Plasma aspect ratio
    B_0:
        Toroidal field at major radius [T]
    I_p:
        Plasma current [A]
    kappa:
        Plasma elongation
    delta:
        Plasma triangularity

    Notes
    -----
    Uckan et al., ITER Physics Design Guidelines, 1989, sec. 2.3
    https://inis.iaea.org/search/search.aspx?orig_q=RN:21068960
    Ref [11] in the above does not appear to include the geometry factor
    """
    eps = 1 / A
    geometry_factor = (1.17 - 0.65 * eps) / (1 - eps**2) ** 2
    q_star = calc_qstar_uckan(R_0, A, B_0, I_p, kappa, delta)
    return q_star * geometry_factor


def estimate_li_wesson(
    q_star: float,
    q_0: float = 1.0,
) -> float:
    """
    Estimate the normalised plasma internal inductance based on an empirical fit.

    Parameters
    ----------
    q_star:
        Cylindrical equivalent safety factor
    q_0:
        Safety factor on axis

    Returns
    -------
    Normalised lasma internal inductance

    Notes
    -----
    Wesson, Tokamaks 3rd edition, page 120

    This appears to give high values for li, even when using q* at rho=0.95
    """
    nu = q_star / q_0 - 1.0
    return np.log(1.65 + 0.89 * nu)
