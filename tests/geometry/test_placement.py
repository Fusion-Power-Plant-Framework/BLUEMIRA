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

import numpy as np
import pytest

from bluemira.geometry.error import GeometryError
from bluemira.geometry.placement import BluemiraPlacement


class TestPlacement:
    def test_instantiation_xy(self):
        xy_placement = BluemiraPlacement([0, 0, 0], [0, 0, 1], 0.0)
        xy_placement_2 = BluemiraPlacement.from_3_points(
            [0, 0, 0], [1, -1, 0], [1, 1, 0]
        )
        assert np.allclose(xy_placement.axis, xy_placement_2.axis)
        assert np.allclose(xy_placement.base, xy_placement_2.base)
        assert xy_placement.angle == xy_placement_2.angle

    def test_instantiation_xz(self):
        xz_placement = BluemiraPlacement([0, 0, 0], [0, -1, 0], 0.0)
        xz_placement_2 = BluemiraPlacement.from_3_points(
            [0, 0, 0], [1, 0, -1], [1, 0, 1]
        )
        assert np.allclose(xz_placement.axis, xz_placement_2.axis)
        assert np.allclose(xz_placement.base, xz_placement_2.base)
        assert xz_placement.angle == xz_placement_2.angle

    def test_instantiation_yz(self):
        yz_placement = BluemiraPlacement([0, 0, 0], [1, 0, 0], 0.0)
        yz_placement_2 = BluemiraPlacement.from_3_points(
            [0, 0, 0], [0, 1, -1], [0, 1, 1]
        )
        assert np.allclose(yz_placement.axis, yz_placement_2.axis)
        assert np.allclose(yz_placement.base, yz_placement_2.base)
        assert yz_placement.angle == yz_placement_2.angle

    def test_instantiation_xyz(self):
        xyz_placement = BluemiraPlacement([0, 0, 0], [1, 1, 1], 0.0)
        xyz_placement_2 = BluemiraPlacement.from_3_points(
            [0, 0, 0], [1, -1, 0], [1, 0, -1]
        )
        assert np.allclose(xyz_placement.axis, xyz_placement_2.axis)
        assert np.allclose(xyz_placement.base, xyz_placement_2.base)
        assert xyz_placement.angle == xyz_placement_2.angle

    def test_colinear(self):
        bad_sets = [
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]],  # co-linear
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # all the same zero
            [[0, 0, 1], [0, 0, 0], [0, 0, -1]],  # co-linear bad order
            [[0, 0, 0], [0, 0, 0], [0, 0, 1]],  # two the same #1
            [[0, 0, 1], [0, 0, 0], [0, 0, 0]],  # two the same #2
            [
                [
                    1,
                    1,
                    1,
                ],
                [1, 1, 1],
                [1, 1, 1],
            ],  # all the same non-zero
        ]
        for bad in bad_sets:
            with pytest.raises(GeometryError):
                BluemiraPlacement.from_3_points(*bad)

    def test_instantiation_matrix(self):
        xy_placement = BluemiraPlacement([0, 0, 0], [0, 0, 1], 0.0)
        matrix = xy_placement.to_matrix()
        xy2_placement = BluemiraPlacement.from_matrix(matrix)
        self._check_placements_equal(xy_placement, xy2_placement)
        rng = np.random.default_rng()
        for _ in range(10):
            placement = BluemiraPlacement(rng.random(3), rng.random(3), rng.random())
            matrix = placement.to_matrix()
            matrix[:3, :3] *= 2  # Test that it gets normalised
            placement2 = BluemiraPlacement.from_matrix(matrix)
            self._check_placements_equal(placement, placement2)
            assert np.allclose(placement.axis, placement2.axis)

    @staticmethod
    def _check_placements_equal(placement, placement2):
        assert np.allclose(placement.to_matrix(), placement2.to_matrix())
        assert np.isclose(placement.angle, placement2.angle)
        assert np.allclose(placement.base, placement2.base)
        assert np.allclose(placement.axis, placement2.axis)
