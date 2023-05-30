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
Fatigue model
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ConductorInfo:
    """
    Cable in conduit conductor information for Paris fatigue model
    """

    tk_radial: float
    width: float  # in the loaded direction
    max_hoop_stress: float
    residual_stress: float
    walker_coeff: float


@dataclass
class ParisFatigueMaterial:
    """
    Material properties for the Paris fatigue model
    """

    C: float  # Paris law material constant
    m: float  # Paris law material exponent
    K_ic: float  # Fracture toughness


@dataclass
class ParisFatigueSafetyFactors:
    """
    Safety factors for the Paris fatigue model
    """

    sf_n_cycle: float
    sf_radial_crack: float
    sf_width_crack: float
    sf_fracture: float


@dataclass
class Crack:
    """
    Crack description for the Paris fatigue model
    """

    depth: float  # a
    width: float  # c
    angle: float = 0.5 * np.pi


def calculate_n_pulses(
    conductor: ConductorInfo,
    initial_crack: Crack,
    material: ParisFatigueMaterial,
    safety: ParisFatigueSafetyFactors,
) -> int:
    """
    Calculate the number of plasma pulses possible prior to fatigue.

    Parameters
    ----------

    Returns
    -------
    Number of plasma pulses

    Notes
    -----
    Assumes two stress cycles per pulse.
    Calculates using the lifecycle method.
    """
    mean_stress_ratio = conductor.residual_stress / (
        conductor.max_hoop_stress + conductor.residual_stress
    )

    C_r = material.C * (1 - mean_stress_ratio) ** (
        material.m * (conductor.walker_coeff - 1)
    )

    max_crack_depth = conductor.tk_radial / safety.sf_radial_crack
    max_crack_width = conductor.width / safety.sf_width_crack
    max_stress_intensity = material.K_ic / safety.sf_fracture

    a = initial_crack.depth
    c = initial_crack.width
    K_max = 0.0  # noqa: N806
    n_cycles = 0

    delta = 1e-4  # Crack size increment

    while a < max_crack_depth and c < max_crack_width and K_max < max_stress_intensity:
        Ka = _calc_semi_elliptical_surface_SIF(  # noqa: N806
            conductor.max_hoop_stress,
            conductor.tk_radial,
            conductor.width,
            a,
            c,
            initial_crack.angle,
        )
        Km = 0.0  # noqa: N806
        K_max = max(Ka, Km)

        a += delta / (Ka / K_max) ** material.m
        c += delta / (Km / K_max) ** material.m
        n_cycles += delta / (C_r * K_max**material.m)

    n_cycles /= safety.sf_n_cycle

    return n_cycles // 2


def _calc_semi_elliptical_surface_SIF(  # noqa: N802
    hoop_stress: float, t: float, w: float, a: float, c: float, phi: float
) -> float:
    """
    Calculate semi-elliptical surface crack stress intensity factor (SIF).

    Parameters
    ----------

    Returns
    -------

    Notes
    -----
    Newman and Raju, 1984, Stress-intensity factor equations for cracks in
    three-dimensional finite bodies subjected to tension and bending loads
    https://ntrs.nasa.gov/api/citations/19840015857/downloads/19840015857.pdf
    """
    bend_stress = 0.0
    a_d_t = a / t

    if a <= c:  # a/c <= 1
        ratio = a / c
        m1 = 1.13 - 0.09 * ratio
        m2 = -0.54 + 0.89 / (0.2 + ratio)
        m3 = 0.5 - 1.0 / (0.65 + ratio) + 14.0 * (1 - ratio) ** 24
        g = 1.0 + (0.1 + 0.35 * ratio**2) * (1 - np.sin(phi)) ** 2
        f_phi = (ratio**2 * np.cos(phi) ** 2 + np.sin(phi) ** 2) ** 4
        f_w = np.sqrt(1.0 / np.cos(np.sqrt(a_d_t) * np.pi * c / (2 * w)))
        g21 = -1.22 - 0.12 * ratio
        g22 = 0.55 - 1.05 * ratio**0.75 + 0.47 * ratio**1.5
        h1 = 1.0 - 0.34 * a_d_t - 0.11 * ratio * a_d_t
        h2 = 1 + g21 * a_d_t + g22 * a_d_t**2

    else:  # a/c > 1
        ratio = c / a
        m1 = np.sqrt(ratio) * (1.0 + 0.04 * ratio)
        m2 = 0.2 * ratio**4
        m3 = -0.11 * ratio**4
        g = 1.0 + (0.1 + 0.35 * ratio * a_d_t**2) * (1 - np.sin(phi)) ** 2
        g11 = -0.04 - 0.41 * ratio
        g12 = 0.55 - 1.93 * ratio**0.75 + 1.38 * ratio**1.5
        g21 = -2.11 + 0.77 * ratio
        g22 = 0.55 - 0.72 * ratio**0.75 + 0.14 * ratio**1.5
        h1 = 1.0 + g11 * a_d_t + g12 * a_d_t**2
        h2 = 1.0 + g21 * a_d_t + g22 * a_d_t**2

    p = 0.2 + ratio + 0.6 * a_d_t
    H_s = h1 + (h2 - h1) * np.sin(phi) ** p  # noqa: N806
    Q = 1.0 + 1.464 * ratio**1.65  # noqa: N806
    F_s = (m1 + m2 * a_d_t**2 + m3 * a_d_t**4) * g * f_phi * f_w  # noqa: N806

    return hoop_stress + H_s * bend_stress * F_s * np.sqrt(np.pi * a / Q)
