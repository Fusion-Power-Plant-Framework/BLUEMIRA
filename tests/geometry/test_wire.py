# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from bluemira.base.file import get_bluemira_path
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.error import GeometryError
from bluemira.geometry.tools import (
    deserialize_shape,
    make_bezier,
    make_circle,
    make_polygon,
)
from bluemira.geometry.wire import BluemiraWire

TEST_PATH = get_bluemira_path("geometry/test_data", subfolder="tests")


class TestWire:
    def test_start_point_given_polygon(self):
        n_coords = 10
        coords = np.zeros((3, n_coords))
        coords[0, :] = np.linspace(0, 2, n_coords)
        coords[2, :] = np.linspace(-5, 0, n_coords)
        wire = make_polygon(coords)

        start_point = wire.start_point()

        assert isinstance(start_point, Coordinates)
        np.testing.assert_equal(start_point, np.array([[0], [0], [-5]]))

    def test_end_point_given_polygon(self):
        n_coords = 10
        coords = np.zeros((3, n_coords))
        coords[0, :] = np.linspace(0, 2, n_coords)
        coords[2, :] = np.linspace(-5, 0, n_coords)
        wire = make_polygon(coords)

        end_point = wire.end_point()

        assert isinstance(end_point, Coordinates)
        np.testing.assert_equal(end_point, np.array([[2], [0], [0]]))

    def test_start_point_eq_end_point_given_circle(self):
        wire = make_circle(radius=2, center=(0, 0, 0))

        start_point = wire.start_point()
        end_point = wire.end_point()

        assert start_point == end_point

    def test_start_point_given_bezier(self):
        n_coords = 11
        coords = np.zeros((3, n_coords))
        coords[0, :] = np.linspace(-5, 5, n_coords)
        coords[1, :] = np.linspace(0.1, 0.2, n_coords)
        coords[2, :] = np.linspace(100, 90, n_coords)
        wire = make_bezier(coords)

        start_point = wire.start_point()

        np.testing.assert_equal(start_point, np.array([[-5], [0.1], [100]]))

    def test_end_point_given_bezier(self):
        n_coords = 11
        coords = np.zeros((3, n_coords))
        coords[0, :] = np.linspace(-5, 5, n_coords)
        coords[1, :] = np.linspace(0.1, 0.2, n_coords)
        coords[2, :] = np.linspace(100, 90, n_coords)
        wire = make_bezier(coords)

        start_point = wire.end_point()

        np.testing.assert_equal(start_point, np.array([[5], [0.2], [90]]))

    def test_vertices_are_ordered(self):
        points = Coordinates({
            "x": [0, 1, 2, 1, 0, -1, 0],
            "y": [-2, -1, 0, 1, 2, 1, -2],
            "z": 0,
        })
        wire = make_polygon(points, closed=True)
        vertices = wire.vertexes
        vertices.set_ccw([0, 0, -1])
        np.testing.assert_allclose(points.xyz[:, :-1], vertices.xyz)

    def test_3_vertices_correct(self):
        p1 = [0, 1, 2]
        p2 = [4, 5, 6]
        p3 = [7, 8, 9]
        w1 = make_polygon([p1, p2])
        w2 = make_polygon([p2, p3])
        wire = BluemiraWire([w1, w2])
        vertexes = wire.vertexes
        np.testing.assert_allclose(vertexes[:, 0], p1)
        np.testing.assert_allclose(vertexes[:, 1], p2)
        np.testing.assert_allclose(vertexes[:, 2], p3)

    def test_end_points(self):
        """Test SHOULD FAIL after change of behaviour"""
        # Toughest case
        rightmost_point = [10.538050403959396, 0.0, -5.974513810642086]
        straight_line_end = {
            "LineSegment": {
                "StartPoint": [9.598817369491535, 0.0, -5.72284707755179],
                "EndPoint": rightmost_point,
            }
        }
        arc_start = {
            "ArcOfCircle": {
                "Radius": 0.8508213264708893,
                "Center": [9.558782242701866, 0.0, -6.572725961982168],
                "Axis": [-0.0, 1.0, -0.0],
                "StartAngle": 180.0,
                "EndAngle": 272.69703056741383,
                "StartPoint": [8.707960916230977, 0.0, -6.572725961982168],
                "EndPoint": [9.598817369491535, 0.0, -5.722847077551793],
            }
        }

        def wrap_as_BMWdict(cadapi_item, label=""):
            return {
                "BluemiraWire": {"label": label, "boundary": [{"Wire": [cadapi_item]}]}
            }

        w1 = deserialize_shape(wrap_as_BMWdict(straight_line_end))
        w2 = deserialize_shape(wrap_as_BMWdict(arc_start))
        wrong_wire = BluemiraWire([
            w1,
            w2,
        ])  # we expect this line to potentially throw an error in future versions??
        right_wire = BluemiraWire([w2, w1])
        np.testing.assert_allclose(wrong_wire.end_point(), Coordinates(rightmost_point))
        np.testing.assert_allclose(right_wire.end_point(), Coordinates(rightmost_point))


class ValueParameterBase:
    @classmethod
    def setup_class(cls):
        cls.square = make_polygon(
            {"x": [0, 2, 2, 0], "y": 0, "z": [0, 0, 2, 2]}, closed=True
        )
        line = make_polygon([[0, 0, 0], [1, 0, 0]])
        semicircle = make_circle(
            1, center=(1, 0, -1), start_angle=90, end_angle=270, axis=(0, -1, 0)
        )
        line2 = make_polygon([[1, 0, -2], [2, 0, -2]])
        cls.mixed = BluemiraWire([line, semicircle, line2])
        cls.circle = make_circle(
            5, center=(5, 0, 0), start_angle=90, end_angle=270, axis=(0, -1, 0)
        )


wireat_parametrize = (
    (0, [0, 0, 0]),
    (0.25, [2, 0, 0]),
    (0.375, [2, 0, 1]),
    (0.5, [2, 0, 2]),
    (0.75, [0, 0, 2]),
    (1.0, [0, 0, 0]),
    (1.1, [0, 0, 0]),
    (-1.0, [0, 0, 0]),
)


class TestWireValueAt(ValueParameterBase):
    @pytest.mark.parametrize(("alpha", "expected_point"), wireat_parametrize)
    def test_square_alpha(self, alpha, expected_point):
        np.testing.assert_allclose(
            self.square.value_at(alpha=alpha), np.array(expected_point)
        )

    @pytest.mark.parametrize(("l_frac", "expected_point"), wireat_parametrize)
    def test_square_distance(self, l_frac, expected_point):
        length = self.square.length
        np.testing.assert_allclose(
            self.square.value_at(distance=l_frac * length), np.array(expected_point)
        )

    @pytest.mark.parametrize(
        ("alpha", "expected_point"),
        [(0.0, [5, 0, 5]), (0.5, [0, 0, 0]), (1.0, [5, 0, -5])],
    )
    def test_circle_alpha(self, alpha, expected_point):
        np.testing.assert_allclose(
            self.circle.value_at(alpha=alpha), np.array(expected_point), atol=1e-10
        )

    def test_mixed_alpha(self):
        assert np.allclose(self.mixed.value_at(alpha=0.0), np.array([0, 0, 0]))
        assert np.allclose(self.mixed.value_at(alpha=0.5), np.array([0, 0, -1]))
        assert np.allclose(self.mixed.value_at(alpha=1.0), np.array([2, 0, -2]))

    def test_mixed_distance(self):
        length = self.mixed.length
        assert np.allclose(self.mixed.value_at(distance=0.0), np.array([0, 0, 0]))
        assert np.allclose(
            self.mixed.value_at(distance=0.5 * length), np.array([0, 0, -1])
        )
        assert np.allclose(self.mixed.value_at(distance=length), np.array([2, 0, -2]))

    def test_point_along_wire_at_length_2d(self):
        # Line in 2d: z = 3x - 4
        coords = np.array([[1, 2, 3, 4, 5], [0, 0, 0, 0, 0], [-1, 2, 5, 8, 11]])
        wire = make_polygon(coords)
        desired_len = np.sqrt(2.5)
        point = wire.value_at(distance=desired_len)

        np.testing.assert_almost_equal(point, [1.5, 0, 0.5], decimal=2)

    def test_point_along_wire_at_length_3d(self):
        circle = make_circle(radius=1, axis=[1, 1, 1])
        semi_circle = make_circle(radius=1, axis=[1, 1, 1], end_angle=180)

        point = circle.value_at(distance=np.pi)

        np.testing.assert_almost_equal(point, semi_circle.end_point().T[0], decimal=2)

    def test_two_value_error(self):
        line = make_polygon([[0, 0, 0], [1, 0, 0]])
        with pytest.raises(GeometryError):
            line.value_at(alpha=0.5, distance=0.5)

    def test_no_value_error(self):
        line = make_polygon([[0, 0, 0], [1, 0, 0]])
        with pytest.raises(GeometryError):
            line.value_at()


class TestWireParameterAt(ValueParameterBase):
    @pytest.mark.parametrize(
        ("point", "alpha"),
        [
            ([0, 0, 0], 0),
            ([2, 0, 0], 0.25),
            ([2, 0, 1], 0.375),
            ([2, 0, 2], 0.5),
            ([0, 0, 2], 0.75),
            ([0, 0, 1e-6], 1.0),  # last point (closure point maps to 0.0)
        ],
    )
    def test_square_vertex(self, point, alpha):
        assert np.isclose(self.square.parameter_at(point), alpha)

    @pytest.mark.parametrize(
        ("point", "alpha"), [([5, 0, 5], 0), ([0, 0, 0], 0.5), ([5, 0, -5], 1.0)]
    )
    def test_circle_vertex(self, point, alpha):
        assert np.isclose(self.circle.parameter_at(point, tolerance=1e-6), alpha)

    def test_mixed_alpha(self):
        assert np.isclose(self.mixed.parameter_at([0, 0, 0]), 0.0)
        assert np.isclose(self.mixed.parameter_at([1, 0, 0]), 1 / (2 + np.pi))
        assert np.isclose(self.mixed.parameter_at([0, 0, -1]), 0.5)
        assert np.isclose(self.mixed.parameter_at([1, 0, -2]), (1 + np.pi) / (2 + np.pi))
        assert np.isclose(self.mixed.parameter_at([2, 0, -2]), 1.0)

    @pytest.mark.parametrize("tolerance", [1e-5, 1e-6, 1e-17])
    def test_tolerance(self, tolerance):
        line = make_polygon([[0, 0, 0], [1, 0, 0]])
        alpha = line.parameter_at([-tolerance, 0, 0], tolerance=tolerance)
        assert np.isclose(alpha, 0.0)

    @pytest.mark.parametrize("tolerance", [1e-5, 1e-7, 1e-16])
    def test_error(self, tolerance):
        with pytest.raises(GeometryError):
            make_polygon([[0, 0, 0], [1, 0, 0]]).parameter_at(
                [-2 * tolerance, 0, 0], tolerance=tolerance
            )


class TestWireDiscretise:
    line = make_polygon({"x": [0, 1], "z": [0, 0]})

    @pytest.mark.parametrize("n", [-1, 0, 1])
    def test_low_ndiscr(self, n):
        with pytest.raises(ValueError):  # noqa: PT011
            self.line.discretise(n, byedges=False)

    @pytest.mark.parametrize("dl", [-10.0, 0])
    def test_low_dl(self, dl):
        with pytest.raises(ValueError):  # noqa: PT011
            self.line.discretise(ndiscr=3, dl=dl, byedges=False)

    @pytest.mark.parametrize("byedges", [True, False])
    def test_ndiscr_3(self, byedges):
        coords = self.line.discretise(ndiscr=3, byedges=byedges)
        assert np.all(coords.x[1:] > 0.0)
