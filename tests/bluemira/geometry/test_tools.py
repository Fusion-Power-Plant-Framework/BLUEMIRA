# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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

# from bluemira.geometry.base import BluemiraGeo

from bluemira.geometry.tools import (
    signed_distance,
    signed_distance_2D_polygon,
    _signed_distance_2D,
    make_polygon,
    wire_plane_intersect,
    plane_intersect,
)
from bluemira.geometry.plane import BluemiraPlane


class TestSignedDistanceFunctions:
    @classmethod
    def setup_class(cls):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
        y = np.array([0.0, -1.0, -1.0, -3.0, -4.0, -2.0, 1.0, 2.5, 3.0, 1.0, 0.0])
        z = np.zeros(len(x))

        cls.subject_2D_array = np.array([x, y]).T
        cls.subject_wire = make_polygon(np.array([x, y, z]).T)

    def test_sdf_2d(self):

        p1 = np.array([0, 0])  # Corner point
        p2 = np.array([0.5, -0.5])  # Mid edge point
        p3 = np.array([3, 0])  # Inside point
        p4 = np.array([-0.1, 0])  # Just outside point
        d1 = _signed_distance_2D(p1, self.subject_2D_array)
        assert d1 == 0
        d2 = _signed_distance_2D(p2, self.subject_2D_array)
        assert d2 == 0
        d3 = _signed_distance_2D(p3, self.subject_2D_array)
        assert d3 > 0
        d4 = _signed_distance_2D(p4, self.subject_2D_array)
        assert d4 == -0.1
        d = np.array([d1, d2, d3, d4])

        d_array = signed_distance_2D_polygon(
            np.array([p1, p2, p3, p4]), self.subject_2D_array
        )

        assert np.allclose(d, d_array)

    def test_sdf(self):
        # Overlapping
        target = make_polygon(
            [[0, 0, 0], [4, 0, 0], [4, 2.5, 0], [0, 2.5, 0], [0, 0, 0]]
        )
        sd = signed_distance(self.subject_wire, target)
        assert sd > 0
        # Touching
        target = make_polygon(
            [[0, 0, 0], [-4, 0, 0], [-4, -2.5, 0], [0, -2.5, 0], [0, 0, 0]]
        )
        sd = signed_distance(self.subject_wire, target)
        assert sd == 0
        # Not overlapping
        target = make_polygon(
            [[-1, 3.5, 0], [-1, -5, 0], [6, -5, 0], [6, 3.5, 0], [-1, 3.5, 0]]
        )
        sd = signed_distance(self.subject_wire, target)
        assert sd < 0


class TestWirePlaneIntersect:
    def test_simple(self):
        loop = make_polygon(
            [[0, 0, -1], [1, 0, -1], [2, 0, -1], [2, 0, 1], [0, 0, 1], [0, 0, -1]]
        )

        xy_plane = BluemiraPlane(axis=[0, 0, 1])
        intersect = plane_intersect(loop, xy_plane)
        e = np.array([[0, 0, 0], [2, 0, 0]])
        e.sort(axis=0)
        intersect.sort(axis=0)
        assert np.allclose(intersect, e)

    def test_complex(self):
        wire = make_polygon(
            [
                [0.0, 0.0, -1.0],
                [1.0, 0.0, -2.0],
                [2.0, 0.0, -3.0],
                [3.0, 0.0, -4.0],
                [4.0, 0.0, -5.0],
                [5.0, 0.0, -6.0],
                [6.0, 0.0, -7.0],
                [7.0, 0.0, -8.0],
                [8.0, 0.0, -4.0],
                [9.0, 0.0, -2.0],
                [10.0, 0.0, 3.0],
                [8.0, 0.0, 2.0],
                [6.0, 0.0, 4.0],
                [4.0, 0.0, 2.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
        xy_plane = BluemiraPlane(axis=[0, 0, 1])
        intersect = plane_intersect(wire, xy_plane)
        assert intersect.shape[0] == 2

        xy_plane = BluemiraPlane(base=[0, 0, 2.7], axis=[0, 1, 0])
        intersect = plane_intersect(wire, xy_plane)
        print(intersect)
        assert intersect.shape[0] == 4

        plane = BluemiraPlane.from_3_points(
            [0, 0, 4], [1, 0, 4], [0, 1, 4]
        )  # x-y offset
        intersect = plane_intersect(wire, plane)
        assert intersect.shape[0] == 1

        plane = BluemiraPlane.from_3_points(
            [0, 0, 4.0005], [1, 0, 4.0005], [0, 1, 4.0005]
        )  # x-y offset
        intersect = plane_intersect(wire, plane)
        assert intersect is None

    def test_other_dims(self):
        wire = make_polygon(
            [
                [0.0, -1.0, 0.0],
                [1.0, -2.0, 0.0],
                [2.0, -3.0, 0.0],
                [3.0, -4.0, 0.0],
                [4.0, -5.0, 0.0],
                [5.0, -6.0, 0.0],
                [6.0, -7.0, 0.0],
                [7.0, -8.0, 0.0],
                [8.0, -4.0, 0.0],
                [9.0, -2.0, 0.0],
                [10.0, 3.0, 0.0],
                [8.0, 2.0, 0.0],
                [6.0, 4.0, 0.0],
                [4.0, 2.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        shift = 0
        for plane in [
            BluemiraPlane.from_3_points(
                [0, shift, 0], [1, shift, 0], [0, shift, 1]
            ),  # x-z
            BluemiraPlane(axis=[0, 1, 0]),
        ]:
            intersect = plane_intersect(wire, plane)
            assert intersect.shape[0] == 2

        shift = 10
        for plane in [
            BluemiraPlane.from_3_points(
                [0, shift, 0], [1, shift, 0], [0, shift, 1]
            ),  # x-z
        ]:
            intersect = plane_intersect(wire, plane)
            assert intersect is None

    def test_xyzplane(self):
        wire = make_polygon(
            [
                [0.0, -1.0, 0.0],
                [1.0, -2.0, 0.0],
                [2.0, -3.0, 0.0],
                [3.0, -4.0, 0.0],
                [4.0, -5.0, 0.0],
                [5.0, -6.0, 0.0],
                [6.0, -7.0, 0.0],
                [7.0, -8.0, 0.0],
                [8.0, -4.0, 0.0],
                [9.0, -2.0, 0.0],
                [10.0, 3.0, 0.0],
                [8.0, 2.0, 0.0],
                [6.0, 4.0, 0.0],
                [4.0, 2.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        wire.translate((-2, 0, 0))
        plane = BluemiraPlane.from_3_points([0, 0, 0], [1, 1, 1], [2, 0, 0])  # x-y-z
        intersect = plane_intersect(wire, plane)
        assert intersect.shape[0] == 2

    def test_flat_intersect(self):
        # test that a shared segment with plane only gives two intersects
        wire = make_polygon(
            [
                [0.0, 0.0, -1.0],
                [2.0, 0.0, -1.0],
                [2.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )

        plane = BluemiraPlane.from_3_points([0, 0, 1], [0, 1, 1], [1, 0, 1])
        inter = plane_intersect(wire, plane)
        true = np.array([[0, 0, 1], [2, 0, 1]])
        true.sort(axis=0)
        inter.sort(axis=0)
        assert np.allclose(inter, true)
