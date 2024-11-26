# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Tools for Coilgroups
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bluemira.magnetostatics.greens import (
    circular_coil_inductance_elliptic,
    greens_psi,
    square_coil_inductance_kirchhoff,
)

if TYPE_CHECKING:
    from bluemira.equilibria.coils import CoilSet


def make_mutual_inductance_matrix(
    coilset: CoilSet, *, square_coil: bool = False, with_quadratures: bool = False
) -> np.ndarray:
    """
    Calculate the mutual inductance matrix of a coilset.

    Parameters
    ----------
    coilset:
        Coilset for which to calculate the mutual inductance matrix

    Returns
    -------
    :
        The symmetric mutual inductance matrix [H]

    Notes
    -----
    Single-filament coil formulation; serves as a useful approximation.

    - **Off-diagonal terms** (:math:`i \\neq j`):

        .. math::
            M_{ij} = n_i n_j G(x_i, z_i, x_j, z_j)

        where :math:`G` is the Green's function for mutual inductance.

    - **Diagonal terms** (:math:`i = j`):

        .. math::
            M_{ii} = n_i^2 L_i

        with :math:`L_i` as the self-inductance using elliptic integrals.

    """
    if with_quadratures:
        n_coils = coilset._quad_dx.ravel().size
        xcoord = coilset._quad_x
        zcoord = coilset._quad_z
        dx = coilset._quad_dx
        dz = coilset._quad_dz
    else:
        n_coils = coilset.n_coils()
        xcoord = coilset.x
        zcoord = coilset.z
        dx = coilset.dx
        dz = coilset.dz

    M = np.zeros((n_coils, n_coils))  # noqa: N806
    n_turns = coilset.n_turns

    itri, jtri = np.triu_indices(n_coils, k=1)
    M[itri, jtri] = (
        n_turns[itri]
        * n_turns[jtri]
        * greens_psi(xcoord[itri], zcoord[itri], xcoord[jtri], zcoord[jtri]).ravel()
    )
    M[jtri, itri] = M[itri, jtri]

    ind = np.diag_indices_from(M)

    if square_coil:
        M[ind] = square_coil_inductance_kirchhoff(xcoord, dx, dz)
    else:
        radius = np.hypot(dx, dz)
        M[ind] = n_turns**2 * circular_coil_inductance_elliptic(xcoord, radius)

    return M


def _get_symmetric_coils(coilset: CoilSet) -> list[list]:
    """
    Coilset symmetry utility

    Returns
    -------
    :
        Symmetric coilset
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


def get_max_current(
    dx: float | np.ndarray, dz: float | np.ndarray, j_max: float | np.ndarray
) -> np.float64 | np.ndarray:
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

    Notes
    -----
    .. math::
        I_{\\text{max}} = j_{\\text{max}} \\cdot (4 \\cdot dx \\cdot dz)

    """
    return np.abs(j_max * (4 * dx * dz))
