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
Tools for Coilgroups
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bluemira.equilibria.coils import CoilSet

import numpy as np

from bluemira.magnetostatics.greens import circular_coil_inductance_elliptic, greens_psi


def make_mutual_inductance_matrix(coilset: CoilSet) -> np.ndarray:
    """
    Calculate the mutual inductance matrix of a coilset.

    Parameters
    ----------
    coilset:
        Coilset for which to calculate the mutual inductance matrix

    Returns
    -------
    The symmetric mutual inductance matrix [H]

    Notes
    -----
    Single-filament coil formulation; serves as a useful approximation.
    """
    n_coils = coilset.n_coils()
    M = np.zeros((n_coils, n_coils))  # noqa: N806
    xcoord = coilset.x
    zcoord = coilset.z
    dx = coilset.dx
    dz = coilset.dz
    n_turns = coilset.n_turns

    itri, jtri = np.triu_indices(n_coils, k=1)

    M[itri, jtri] = (
        n_turns[itri]
        * n_turns[jtri]
        * greens_psi(xcoord[itri], zcoord[itri], xcoord[jtri], zcoord[jtri])
    )
    M[jtri, itri] = M[itri, jtri]

    radius = np.hypot(dx, dz)
    for i in range(n_coils):
        M[i, i] = n_turns[i] ** 2 * circular_coil_inductance_elliptic(
            xcoord[i], radius[i]
        )

    return M


def _get_symmetric_coils(coilset: CoilSet):
    """
    Coilset symmetry utility
    """
    x, z, dx, dz, currents = coilset.to_group_vecs()
    coil_matrix = np.array([x, np.abs(z), dx, dz, currents]).T

    sym_stack = [[coil_matrix[0], 1]]
    for i in range(1, len(x)):
        coil = coil_matrix[i]

        for j, sym_coil in enumerate(sym_stack):
            if np.allclose(coil, sym_coil[0]):
                sym_stack[j][1] += 1
                break

        else:
            sym_stack.append([coil, 1])

    return sym_stack


def check_coilset_symmetric(coilset: CoilSet) -> bool:
    """
    Check whether or not a CoilSet is purely symmetric about z=0.

    Parameters
    ----------
    coilset:
        CoilSet to check for symmetry

    Returns
    -------
    Whether or not the CoilSet is symmetric about z=0
    """
    sym_stack = _get_symmetric_coils(coilset)
    for coil, count in sym_stack:
        if count != 2 and not np.isclose(coil[1], 0.0):  # noqa: PLR2004
            # therefore z = 0
            return False
    return True


def get_max_current(dx: float, dz: float, j_max: float) -> float:
    """
    Get the maximum current in a rectangular coil cross-sectional area

    Parameters
    ----------
    dx:
        Coil half-width [m]
    dz:
        Coil half-height [m]
    j_max:
        Coil current density [A/m^2]

    Returns
    -------
    Maximum current [A]
    """
    return abs(j_max * (4 * dx * dz))
