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
    K_max = 0.0
    n_cycles = 0

    delta = 1e-4  # Crack size increment

    while a < max_crack_depth and c < max_crack_width and K_max < max_stress_intensity:
        Ka = _calc_stress_intensity_factor(
            conductor.max_hoop_stress,
            conductor.tk_radial,
            conductor.width,
            a,
            c,
            initial_crack.angle,
        )
        Km = 0.0
        K_max = max(Ka, Km)

        a += delta / (Ka / K_max) ** material.m
        c += delta / (Km / K_max) ** material.m
        n_cycles += delta / (C_r * K_max**material.m)

    n_cycles /= safety.sf_n_cycle

    return n_cycles // 2


def _calc_stress_intensity_factor(hoop_stress, t, w, a, c, phi):
    bend_stress = 0.0

    if a <= c:  # a/c <= 1
        ratio = a / c
        m1 = 1.13 - 0.09 * ratio
        m2 = -0.54 + 0.89 / (0.2 + ratio)
        m3 = 0.5 - 1.0 / (0.65 + ratio) + 14 * (1 - ratio) ** 24
        g = 1.0 + (0.1 + 0.35 * ratio**2) * (1 - np.sin(phi)) ** 2
        f_phi = (ratio**2 * np.cos(phi) ** 2 + np.sin(phi) ** 2) ** 4
        f_w = np.sqrt(1.0 / np.cos(np.sqrt(a / t) * np.pi * c / (2 * w)))
        g21 = -1.22 - 0.12 * ratio
        g22 = 0.55 - 1.05 * ratio**0.75 + 0.47 * ratio**1.5
        h1 = 1.0 - 0.34 * a / t - 0.11 * ratio * (a / t)
        h2 = 1 + g21

    else:  # a/c > 1
        ratio = c / a

    p = 0.2 + ratio + 0.6 * a / t
    H_s = h1 + (h2 - h1) * np.sin(phi) ** p
    Q = 1.0 + 1.464 * ratio**1.65
    F_s = (m1 + m2 * (a / t) ** 2 + m3 * (a / t) ** 4) * g * f_phi * f_w

    return hoop_stress + H_s * bend_stress * np.sqrt(np.pi * a * F_s / Q)
