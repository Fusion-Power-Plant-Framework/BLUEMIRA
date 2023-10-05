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

import itertools

import numpy as np
from matplotlib import pyplot as plt

from bluemira.structural.node import Node
from bluemira.structural.transformation import (
    _direction_cosine_matrix,
    _direction_cosine_matrix_debugging,
    lambda_matrix,
)


class TestLambdaTransformationMatrices:
    """
    The absolute nightmare you had with these dcms...
    """

    global_cs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    rng = np.random.default_rng()

    @staticmethod
    def assert_maths_good(dcm, msg=""):  # noqa: ARG004
        # First, test some mathematical properties such as orthagonality
        inv = np.linalg.inv(dcm)
        assert np.allclose(dcm.T, inv)
        assert np.isclose(abs(np.linalg.det(dcm)), 1)

    @staticmethod
    def plot_nodes(node1, node2, local, fig=None, ax=None):
        # Visualise transform
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111, projection="3d")

        # Global coordinate system
        ax.plot([0], [0], [0], marker="s", color="k", ms=20)
        ax.quiver(0, 0, 0, 1, 0, 0, color="k")
        ax.quiver(0, 0, 0, 0, 1, 0, color="k")
        ax.quiver(0, 0, 0, 0, 0, 1, color="k")

        # Local coordinate system
        ax.plot([node1.x], [node1.y], [node1.z], marker="o", color="b", ms=20)
        ax.plot([node2.x], [node2.y], [node2.z], marker="o", color="b", ms=20)
        ax.plot([node1.x, node2.x], [node1.y, node2.y], [node1.z, node2.z])
        ax.quiver(node1.x, node1.y, node1.z, *local[0], color="r")
        ax.quiver(node1.x, node1.y, node1.z, *local[1], color="g")
        ax.quiver(node1.x, node1.y, node1.z, *local[2], color="b")

        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        plt.show()
        plt.close(fig)

    def assert_works_good(self, dcm, local, msg=""):  # noqa: ARG002
        global_check = dcm @ local
        local_check = dcm.T @ self.global_cs
        # Check matrix is what you want (performs as intended)
        assert np.allclose(self.global_cs, global_check)
        assert np.allclose(local, local_check)

    def test_math_property(self):
        for _ in range(100):
            l_matrix = lambda_matrix(*self.rng.random(3))
            assert np.allclose(l_matrix.T, np.linalg.inv(l_matrix))

    # =========================================================================
    # Debugging mode tests
    # =========================================================================
    def test_development(self):
        n_1 = Node(4.0, 5.0, 6.0, 0)
        n_2 = Node(5.0, 6.0, 7.0, 1)
        dx = n_2.x - n_1.x
        dy = n_2.y - n_1.y
        dz = n_2.z - n_1.z
        dcm, _ = _direction_cosine_matrix_debugging(dx, dy, dz, debug=True)
        self.assert_maths_good(dcm)

    def test_random(self):
        for _ in range(10):
            v = self.rng.random(3)
            dcm, local = _direction_cosine_matrix_debugging(*v, debug=True)
            self.assert_maths_good(dcm, msg=f"coords: {v}")

            self.assert_works_good(dcm, local, msg=f"coords: {v}")
            self.plot_nodes(
                Node(1, 1, 1, 0), Node(1 + v[0], 1 + v[1], 1 + v[2], 1), local
            )

    def test_big_random(self):
        for _ in range(100):
            v = 10000 * self.rng.random(3)
            dcm, local = _direction_cosine_matrix_debugging(*v, debug=True)
            self.assert_maths_good(dcm, msg=f"coords: {v}")
            self.assert_works_good(dcm, local, msg=f"coords: {v}")

    def test_edges(self):
        edge_coords = list(itertools.product([0, 1], [0, 1], [0, 1]))
        edge_coords.remove((0, 0, 0))  # Special case
        for coord_system in edge_coords:
            dcm, local = _direction_cosine_matrix_debugging(*coord_system, debug=True)
            self.assert_maths_good(dcm, msg=f"coords: {coord_system}")
            self.assert_works_good(dcm, local, msg=f"coords: {coord_system}")

    def test_negative_edges(self):
        edge_coords = list(itertools.product([0, 1, -1], [0, 1, -1], [0, 1, -1]))
        edge_coords.remove((0, 0, 0))  # Special case
        for coord_system in edge_coords:
            dcm, local = _direction_cosine_matrix_debugging(*coord_system, debug=True)
            self.assert_maths_good(dcm, msg=f"coords: {coord_system}")
            self.assert_works_good(dcm, local, msg=f"coords: {coord_system}")

    # =========================================================================
    # Normal mode tests
    # =========================================================================

    def test_translation(self):
        dcm = _direction_cosine_matrix(1, 0, 0)
        self.assert_maths_good(dcm)

        self.assert_works_good(dcm, self.global_cs)

    def test_random2(self):
        for _ in range(100):
            v = self.rng.random(3)
            dcm = _direction_cosine_matrix(*v)
            self.assert_maths_good(dcm, msg=f"coords: {v}")

    def test_big_random2(self):
        for _ in range(100):
            v = 10000 * self.rng.random(3)
            dcm = _direction_cosine_matrix(*v)
            self.assert_maths_good(dcm, msg=f"coords: {v}")

    def test_edges2(self):
        axis_coord_combo = {
            (1, 0, 0): np.eye(3),
            (-1, 0, 0): np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            (0, 0, 1): np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
            (0, 0, -1): np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
            (0, 1, 0): np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            (0, -1, 0): np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
        }

        for axis, local in axis_coord_combo.items():
            dcm = _direction_cosine_matrix(*axis)
            self.assert_maths_good(dcm, msg=f"coords: {axis}")

            self.assert_works_good(dcm, local, msg=f"coords: {axis}")
