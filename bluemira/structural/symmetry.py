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
Symmetry boundary conditions
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from bluemira.structural.geometry import Geometry

import numpy as np

from bluemira.geometry.coordinates import (
    get_angle_between_points,
    project_point_axis,
    rotation_matrix,
)
from bluemira.structural.error import StructuralError
from bluemira.structural.matrixops import cyclic_decomposition


class CyclicSymmetry:
    """
    Utility class for implementing a cyclical symmetry boundary condition in
    FiniteElementModel

    Parameters
    ----------
    geometry:
        The geometry upon which to apply the cyclic symmetry
    cycle_sym_ids:
        The list of left and right DOF ids
    """

    def __init__(self, geometry: Geometry, cycle_sym_ids: List[List[int], List[int]]):
        self.geometry = geometry
        self.cycle_sym_ids = cycle_sym_ids

        # Constructors
        self.t_block = None
        self.t_matrix = None
        self.left_nodes = None
        self.right_nodes = None
        self.theta = None
        self.n = None
        self.axis = None
        self.selections = None

        if self.cycle_sym_ids:
            self._prepare_cyclic_symmetry()

    def _prepare_cyclic_symmetry(self):
        n = []
        left_nodes = []
        right_nodes = []
        for left, right, p1, p2 in self.cycle_sym_ids:
            left_nodes.append(left)
            right_nodes.append(right)
            axis = np.array(p2) - np.array(p1)

            l_point = self.geometry.node_xyz[left]
            r_point = self.geometry.node_xyz[right]

            pa = project_point_axis(l_point, axis)

            angle = get_angle_between_points(r_point, pa, l_point)

            n.append(2 * np.pi / angle)

        n = np.round(n).astype(np.int32)

        if not np.all(n == n[0]):
            raise StructuralError(
                "CyclicSymmetry: cyclic symmetry boundary condition is "
                "incorrectly specified:\n"
                f"periodicity is not uniform: {n}"
            )

        n = n[0]
        theta = angle
        self.t_block = rotation_matrix(theta, axis)
        self._build_t_matrix(6 * len(left_nodes))
        self.left_nodes = left_nodes
        self.right_nodes = right_nodes
        self.n = n
        self.axis = axis
        self.theta = theta

    def _build_t_matrix(self, n):
        t_m = np.zeros((n, n))
        for _i in range(int(n / 6)):
            i = _i * 6
            t_m[i : i + 3, i : i + 3] = self.t_block
            t_m[i + 3 : i + 6, i + 3 : i + 6] = self.t_block
        self.t_matrix = t_m

    def apply_cyclic_symmetry(
        self, k: np.ndarray, p: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the cyclic symmetry condition to the matrices.

        Will simply return k and p if no cyclic symmetry is detected in the
        FiniteElementModel.

        Parameters
        ----------
        k:
            The geometry stiffness matrix
        p:
            The model load vector

        Returns
        -------
        k:
            The partitioned block stiffness matrix
        p:
            The partitioned block load vector
        """
        if not self.cycle_sym_ids:
            # Do nothing
            return k, p

        k_cyc, p_cyc, self.selections = cyclic_decomposition(
            k, p, self.left_nodes, self.right_nodes
        )

        # Unpack the decomposition
        (k_rr, k_rl, k_ri), (k_lr, k_ll, k_li), (k_ir, k_il, k_ii) = k_cyc
        p_r, p_l, p_i = p_cyc

        e = 1  # np.exp(1j * self.n * self.theta)
        en = 1  # np.exp(-1j * self.n * self.theta)

        t_m = self.t_matrix

        # Build block matrix components
        k_11 = k_rr + t_m @ k_ll @ t_m.T + e * k_rl @ t_m.T + en * t_m @ k_lr
        k_12 = k_ri + en * t_m @ k_li
        k_21 = k_ir + e * k_il @ t_m.T
        k_22 = k_ii

        k = np.block([[k_11, k_12], [k_21, k_22]])

        # This is a hack, is it not? Must get to the bottom of this...
        # It seems it is a hack...
        p = np.block([p_r + en * t_m @ p_l, p_i])
        return k, p

    def reorder(self, u_original: np.ndarray) -> np.ndarray:
        """
        Re-order the displacement vector correctly, so that the deflections
        may be applied to the correct nodes.

        If not cyclic symmetry is detected in the FiniteElementModel, simply
        returns u_original.

        Parameters
        ----------
        u_original:
            The original deflection vector

        Returns
        -------
        The re-ordered deflection vector
        """
        if not self.cycle_sym_ids:
            # Do nothing
            return u_original

        u_ordered = np.zeros((6 * self.geometry.n_nodes,))
        left, right, interior = self.selections

        u_right = u_original[: len(right)]
        u_left = self.t_matrix.T @ u_right
        u_interior = u_original[len(right) :]

        u_ordered[interior] = u_interior
        u_ordered[right] = u_right
        u_ordered[left] = u_left

        return u_ordered
