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

import difflib
import json
import re
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from numpy.linalg import norm

import bluemira.codes._freecadapi as cadapi
from bluemira.base.constants import EPS
from bluemira.base.file import get_bluemira_path
from bluemira.geometry.error import GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import (
    PictureFrame,
    PolySpline,
    PrincetonD,
    TripleArc,
)
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import (
    _signed_distance_2D,
    boolean_fragments,
    chamfer_wire_2D,
    convex_hull_wires_2d,
    deserialize_shape,
    extrude_shape,
    fallback_to,
    fillet_wire_2D,
    find_clockwise_angle_2d,
    force_wire_to_spline,
    interpolate_bspline,
    log_geometry_on_failure,
    make_circle,
    make_circle_arc_3P,
    make_ellipse,
    make_polygon,
    mirror_shape,
    offset_wire,
    point_inside_shape,
    revolve_shape,
    save_as_STP,
    save_cad,
    signed_distance,
    signed_distance_2D_polygon,
    slice_shape,
)
from bluemira.geometry.wire import BluemiraWire
from tests._helpers import combine_text_mock_write_calls

generic_wire = make_polygon(
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


class TestMakePolygon:
    def test_open_wire(self):
        wire = make_polygon({"x": [0, 1, 1], "y": 0, "z": [1, 1, 2]})
        assert not wire.is_closed()
        assert np.isclose(wire.length, 2.0, rtol=0.0, atol=EPS)

    def test_closed_wire_with_closed(self):
        wire = make_polygon({"x": [0, 1, 1], "y": 0, "z": [1, 1, 2]}, closed=True)
        assert wire.is_closed()
        assert np.isclose(wire.length, 2.0 + np.sqrt(2), rtol=0.0, atol=EPS)

    @pytest.mark.parametrize("closed", [True, False])
    def test_closed_wire_with_points(self, closed):
        wire = make_polygon(
            {"x": [1, 2, 2, 1, 1], "y": 0, "z": [1, 1, 2, 2, 1]}, closed=closed
        )
        assert wire.is_closed()
        assert np.isclose(wire.length, 4.0)

    @pytest.mark.parametrize("closed", [True, False])
    def test_closed_wire_with_triangle_points(self, closed):
        wire = make_polygon(
            {"x": [1, 2, 2, 1], "y": 0, "z": [1, 1, 2, 1]}, closed=closed
        )
        assert wire.is_closed()
        assert np.isclose(wire.bounding_box.y_min, 0.0)
        assert np.isclose(wire.bounding_box.y_max, 0.0)
        assert np.isclose(wire.length, 2 + np.sqrt(2))


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

        xy_plane = BluemiraPlane(base=(0, 0, 0), axis=(0, 0, 1))
        intersect = slice_shape(loop, xy_plane)
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
        xy_plane = BluemiraPlane(base=(0, 0, 0), axis=(0, 0, 1))
        intersect = slice_shape(wire, xy_plane)
        assert intersect.shape[0] == 2

        xy_plane = BluemiraPlane(base=(0, 0, 2.7), axis=(0, 0, 1))
        intersect = slice_shape(wire, xy_plane)
        print(intersect)
        assert intersect.shape[0] == 4

        plane = BluemiraPlane.from_3_points(
            [0, 0, 4], [1, 0, 4], [0, 1, 4]
        )  # x-y offset
        intersect = slice_shape(wire, plane)
        assert intersect.shape[0] == 1

        plane = BluemiraPlane.from_3_points(
            [0, 0, 4.0005], [1, 0, 4.0005], [0, 1, 4.0005]
        )  # x-y offset
        intersect = slice_shape(wire, plane)
        assert intersect is None

    def test_other_dims(self):
        shift = 0
        for plane in [
            BluemiraPlane.from_3_points(
                [0, shift, 0], [1, shift, 0], [0, shift, 1]
            ),  # x-z
            BluemiraPlane(axis=[0, 1, 0]),
        ]:
            intersect = slice_shape(generic_wire, plane)
            assert intersect.shape[0] == 2

        shift = 10
        for plane in [
            BluemiraPlane.from_3_points(
                [0, shift, 0], [1, shift, 0], [0, shift, 1]
            ),  # x-z
        ]:
            intersect = slice_shape(generic_wire, plane)
            assert intersect is None

    def test_xyzplane(self):
        wire = generic_wire.copy()
        wire.translate((-2, 0, 0))
        plane = BluemiraPlane.from_3_points([0, 0, 0], [1, 1, 1], [2, 0, 0])  # x-y-z
        intersect = slice_shape(wire, plane)
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
        inter = slice_shape(wire, plane)
        true = np.array([[0, 0, 1], [2, 0, 1]])
        true.sort(axis=0)
        inter.sort(axis=0)
        assert np.allclose(inter, true)

    def test_weird_wire(self):
        # test a wire that moves in 3 dimensions
        wire = make_polygon(
            [
                [0.0, -1.0, 0.0],
                [1.0, -2.0, 1.0],
                [2.0, -3.0, 2.0],
                [3.0, -4.0, 1.0],
                [4.0, -5.0, 0.0],
                [5.0, -6.0, -1.0],
                [6.0, -7.0, -2.0],
                [7.0, -8.0, -1.0],
                [8.0, -4.0, 0.0],
                [9.0, -2.0, 1.0],
                [10.0, 3.0, 2.0],
                [8.0, 2.0, 1.0],
                [6.0, 4.0, 0.0],
                [4.0, 2.0, -1.0],
                [2.0, 0.0, -2.0],
                [0.0, -1.0, 0.0],
            ]
        )

        plane = BluemiraPlane.from_3_points([1, -2, -1], [6, 4, 0], [9, -2, 1])

        intersect = slice_shape(wire, plane)
        assert intersect.shape[0] == 4


class TestSolidFacePlaneIntersect:
    big = 10
    small = 5
    centre = 15
    twopi = 2 * np.pi
    offset = 1

    cyl_rect = 2 * big + 2 * offset
    twopir = twopi * small

    xz_plane = BluemiraPlane(axis=[0, 1, 0])
    xy_plane = BluemiraPlane(axis=[0, 0, 1])
    yz_plane = BluemiraPlane(axis=[1, 0, 0])

    @pytest.mark.parametrize(
        ("plane", "length", "hollow"),
        [
            # hollow
            (xz_plane, offset, True),
            (yz_plane, offset, True),
            (xy_plane, twopir, True),
            (BluemiraPlane(base=[0, 0, 0.5], axis=[0, 0, 1]), twopir, True),
            (BluemiraPlane(base=[0, 0, offset], axis=[0, 0, 1]), twopir, True),
            # solid
            (xz_plane, cyl_rect, False),
            (yz_plane, cyl_rect, False),
            # tangent intersecting plane doesnt work at solid base??
            pytest.param(xy_plane, twopir, False, marks=[pytest.mark.xfail]),
            (BluemiraPlane(base=[0, 0, 0.5], axis=[0, 0, 1]), twopir, False),
            (BluemiraPlane(base=[0, 0, offset], axis=[0, 0, 1]), twopir, False),
        ],
    )
    def test_cylinder(self, plane, length, hollow):
        circ = make_circle(self.small)
        if not hollow:
            circ = BluemiraFace(circ)
        cylinder = extrude_shape(circ, (0, 0, self.offset))
        _slice = slice_shape(cylinder, plane)
        assert _slice is not None
        assert all(np.isclose(sl.length, length) for sl in _slice), [
            f"{sl.length}, {length}" for sl in _slice
        ]

    def test_solid_nested_donut(self):
        circ = make_circle(self.small, [0, 0, self.centre], axis=[0, 1, 0])
        circ2 = make_circle(self.big, [0, 0, self.centre], axis=[0, 1, 0])

        face = BluemiraFace([circ2, circ])

        # cant join a face to itself atm 20/12/21
        donut = revolve_shape(face, direction=[1, 0, 0], degree=359)

        _slice = slice_shape(donut, self.xz_plane)

        no_big = 0
        no_small = 0
        for sl in _slice:
            try:
                assert np.isclose(sl.length / self.twopi, self.big)
                no_big += 1
            except AssertionError:  # noqa: PERF203
                assert np.isclose(sl.length / self.twopi, self.small)
                no_small += 1
        assert no_big == 2
        assert no_small == 2

    def test_primitive_cut(self):
        path = PrincetonD({"x2": {"value": self.big}}).create_shape()
        p2 = offset_wire(path, self.offset)
        face = BluemiraFace([p2, path])
        extruded = extrude_shape(face, (0, 1, 0))

        _slice_xy = slice_shape(extruded, self.xy_plane)
        _slice_xz = slice_shape(extruded, BluemiraPlane(base=[0, 1, 0], axis=[0, 1, 0]))

        assert len(_slice_xy) == 2
        assert len(_slice_xz) == 2

    def test_polygon_cut(self):
        face = BluemiraFace(generic_wire)
        _slice_face = slice_shape(face, BluemiraPlane())
        assert generic_wire.length == _slice_face[0].length

        solid = extrude_shape(face, (1, 2, 3))
        _slice_solid = slice_shape(solid, BluemiraPlane(axis=[3, 2, 1]))
        assert len(_slice_solid) == 1


class TestPointInside:
    def test_simple(self):
        polygon = BluemiraFace(
            make_polygon({"x": [-2, 2, 2, -2, -2, -2], "z": [-2, -2, 2, 2, 1.5, -2]})
        )
        in_points = [
            [-1, 0, -1],
            [-1, 0, 0],
            [-1, 0, 1],
            [0, 0, -1],
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, -1],
            [1, 0, 0],
            [1, 0, 1],
        ]
        for point in in_points:
            assert point_inside_shape(point, polygon)

        out_points = [
            [-3, 0, -3],
            [-3, 0, 0],
            [-3, 0, 3],
            [0, 0, -3],
            [3, 0, 3],
            [3, 0, -3],
            [2.005, 0, 0],
            [2.001, 0, -1.9999],
            # TODO: This is not very good FreeCAD..
            # [2.00000009, 0, 0],
            # [-2.0000000001, 0, -1.999999999999],
        ]

        for point in out_points:
            assert not point_inside_shape(point, polygon)


class TestConvexHullWires2d:
    def test_ValueError_given_wires_empty(self):
        with pytest.raises(ValueError):  # noqa: PT011
            convex_hull_wires_2d([], 10)

    def test_hull_around_two_circles_xz_plane(self):
        circle_1 = make_circle(radius=1, center=[-0.5, 0, 0.5], axis=(0, 1, 0))
        circle_2 = make_circle(radius=1, center=[0.5, 0, -0.5], axis=(0, 1, 0))

        hull = convex_hull_wires_2d([circle_1, circle_2], ndiscr=200)

        assert hull.is_closed
        assert np.allclose(hull.center_of_mass, [0, 0, 0])
        bounding_box = hull.bounding_box
        assert bounding_box.z_min == -1.5
        assert bounding_box.z_max == 1.5
        assert bounding_box.x_min == -1.5
        assert bounding_box.x_max == 1.5
        assert bounding_box.y_min == bounding_box.y_max == 0

    def test_hull_around_two_circles_xy_plane(self):
        circle_1 = make_circle(radius=1, center=[-0.5, 1, 0.5], axis=(1, 1, 1))
        circle_2 = make_circle(radius=1, center=[0.5, -2, -0.5], axis=(1, 1, 1))

        hull = convex_hull_wires_2d([circle_1, circle_2], ndiscr=1000, plane="xy")

        assert hull.is_closed
        assert np.allclose(hull.center_of_mass, [0, -0.5, 0])
        bounding_box = hull.bounding_box
        assert bounding_box.z_min == bounding_box.z_max == 0

    @pytest.mark.parametrize("bad_plane", ["ab", "", None, ["x", "y"]])
    def test_ValueError_if_invalid_plane(self, bad_plane):
        circle = make_circle(radius=1)

        with pytest.raises(ValueError):  # noqa: PT011
            convex_hull_wires_2d([circle], 10, plane=bad_plane)


class TestMakeBSpline:
    fixture = (
        (None, None),
        ([0, 0, 1], [0, 0, 1]),
        ([0, 0, -1], [0, 0, -1]),
        ([0, 0, -1], [0, 0, 1]),
        ([0, 0, 1], [0, 0, -1]),
    )

    @pytest.mark.parametrize(("st", "et"), fixture)
    def test_tangencies_open(self, st, et):
        """
        Open spline start and end tangencies.
        """
        points = {"x": np.linspace(0, 1, 4), "y": 0, "z": np.zeros(4)}
        spline = interpolate_bspline(
            points, closed=False, start_tangent=st, end_tangent=et
        )
        # np.testing.assert_allclose(spline.length, expected_length)
        if st and et:
            assert spline.length > 1.0
            e = spline.shape.Edges[0]
            np.testing.assert_allclose(
                e.tangentAt(e.FirstParameter), np.array(st) / norm(st)
            )
            np.testing.assert_allclose(
                e.tangentAt(e.LastParameter), np.array(et) / norm(et)
            )
        else:
            np.testing.assert_allclose(spline.length, 1.0)

    @pytest.mark.parametrize(("st", "et"), fixture)
    def test_tangencies_closed(self, st, et):
        points = {"x": [0, 1, 2, 1], "y": 0, "z": [0, -1, 0, 1]}
        spline = interpolate_bspline(
            points, closed=True, start_tangent=st, end_tangent=et
        )
        if st and et:
            e = spline.shape.Edges[0]
            np.testing.assert_allclose(
                e.tangentAt(e.FirstParameter), np.array(st) / norm(st)
            )

            # if the bspline is closed, end tangency is not considered. Last point is
            # equal to the first point, thus also its tangent.
            np.testing.assert_allclose(
                e.tangentAt(e.LastParameter), np.array(st) / norm(st)
            )

    def test_bspline_closed(self):
        # first != last, closed = True
        points = {"x": [0, 1, 1, 0], "y": 0, "z": [0, 0, 1, 1]}
        spline = interpolate_bspline(points, closed=True)
        assert spline.length == 4.520741504557154

        # first == last, closed = True
        points = {"x": [0, 1, 1, 0, 0], "y": 0, "z": [0, 0, 1, 1, 0]}
        spline = interpolate_bspline(points, closed=True)
        assert spline.length == 4.520741504557154

        # first == last, closed = False (closed is enforced)
        spline = interpolate_bspline(points, closed=False)
        assert spline.length == 4.520741504557154


class TestFindClockwiseAngle2d:
    @pytest.mark.parametrize(
        "fixture",
        [
            (np.array([-1, 0]), np.array([-1, 0]), 0),
            (np.array([-1, 0]), np.array([0, 1]), 90),
            (np.array([-1, 0]), np.array([0, -1]), 270),
            (
                np.array([-1, 0]),
                np.array([[0, 1, 0, -1], [1, 1, -1, -1]]),
                np.array([90, 135, 270, 315]),
            ),
        ],
    )
    def test_output_contains_clockwise_angle_given_valid_input(self, fixture):
        base, vector, expected = fixture
        np.testing.assert_allclose(find_clockwise_angle_2d(base, vector), expected)

    @pytest.mark.parametrize("value", [[0, 1], 100, "not np.ndarray"])
    @pytest.mark.parametrize("vector_name", ["base", "vector"])
    def test_TypeError_given_input_is_not_ndarray(self, value, vector_name):
        params = {
            "base": np.array([0, 1]),
            "vector": np.array([0, 1]),
        }
        params[vector_name] = value

        with pytest.raises(TypeError):
            find_clockwise_angle_2d(**params)

    @pytest.mark.parametrize("size", [0, 3, 10])
    @pytest.mark.parametrize("vector_name", ["base", "vector"])
    def test_ValueError_given_inputs_axis_0_size_not_2(self, size, vector_name):
        params = {
            "base": np.array([0, 1]),
            "vector": np.array([0, 1]),
        }
        params[vector_name] = np.zeros((size, 1))

        with pytest.raises(ValueError):  # noqa: PT011
            find_clockwise_angle_2d(**params)


@log_geometry_on_failure
def naughty_function(wire, var=1, *, var2=(1, 2), **kwargs):  # noqa: ARG001
    raise cadapi.FreeCADError


def naughty_function_result(wire, *, var2=(1, 2), **kwargs):  # noqa: ARG001
    return 41 + kwargs["missing_piece"]


@fallback_to(naughty_function_result, cadapi.FreeCADError)
@log_geometry_on_failure
def naughty_function_fallback(wire, var=1, *, var2=(1, 2), **kwargs):  # noqa: ARG001
    raise cadapi.FreeCADError


class TestLogFailedGeometryOperationSerialisation:
    wires = (
        make_polygon({"x": [0, 2, 2, 0], "y": [-1, -1, 1, 1]}, closed=True),
        make_circle(),
        make_ellipse(),
        PrincetonD().create_shape(),
        PolySpline().create_shape(),
        PictureFrame().create_shape(),
        TripleArc().create_shape(),
    )

    @mock.patch("builtins.open", new_callable=mock.mock_open)
    @pytest.mark.parametrize("wire", wires)
    def test_file_is_made(self, open_mock, wire):
        length = wire.length

        with pytest.raises(cadapi.FreeCADError):
            naughty_function(wire, var2=[1, 2, 3], random_kwarg=np.pi)

        open_mock.assert_called_once()
        written_data = combine_text_mock_write_calls(open_mock)
        data = json.loads(written_data)

        # Check the serialization of the input shape
        assert "var" in data
        assert data["var"] == 1
        assert "var2" in data
        assert data["var2"] == [1, 2, 3]
        assert "random_kwarg" in data
        assert np.isclose(data["random_kwarg"], np.pi)
        saved_wire = deserialize_shape(data["wire"])
        np.testing.assert_almost_equal(saved_wire.length, length, decimal=8)

    @mock.patch("builtins.open", new_callable=mock.mock_open)
    def test_fallback_logs_and_returns(self, open_mock):
        result = naughty_function_fallback(0, missing_piece=1)

        open_mock.assert_called_once()
        call_args = open_mock.call_args[0]
        assert Path(call_args[0]).name.startswith("naughty_function_fallback")
        assert call_args[1] == "w"
        assert result == 42


class TestMakeCircle:
    def test_make_circle_arc_3P(self):
        p1 = [1, 0, 2]
        p2 = [2, 0, 3]
        p3 = [0, 0, 3.2]

        arc = make_circle_arc_3P(p1, p2, p3)
        points = arc.discretize(2).points
        np.testing.assert_allclose(np.array(p1), np.array(points[0]), atol=EPS)
        np.testing.assert_allclose(np.array(p3), np.array(points[1]), atol=EPS)


class TestSavingCAD:
    STP_VERSION_RE = r"(processor)|(translator) [0-9]+\.[0-9]+"

    def setup_method(self):
        fp = get_bluemira_path("geometry/test_data", subfolder="tests")
        self.test_file = Path(fp, "test_circ.stp")
        self.generated_file = "test_generated_circ.stp"
        self.obj = make_circle(5, axis=(1, 1, 1))

    @pytest.mark.xfail(reason="Unknown, passes locally")
    def test_save_as_STP(self, tmpdir):
        self._save_and_check(self.obj, save_as_STP, tmpdir)

    def test_save_cad(self, tmpdir):
        self._save_and_check(self.obj, save_cad, tmpdir)

    def _save_and_check(self, obj, save_func, tmpdir):
        # Can't mock out as written by freecad not python
        self.generated_file = tmpdir.join(self.generated_file)
        save_func(obj, filename=str(self.generated_file).split(".")[0])

        with open(self.test_file) as tf:
            lines1 = tf.readlines()

        with open(self.generated_file) as gf:
            lines2 = gf.readlines()

        # Dont care about date/time differences
        lines = []
        for line in difflib.unified_diff(
            lines1, lines2, fromfile="", tofile="", lineterm="", n=0
        ):
            if not line.startswith(("---", "+++", "@@")):
                try:
                    datetime.fromisoformat(line.split(",")[1].strip("'"))
                except (ValueError, IndexError):
                    # Attempt to ignore version number
                    if not re.search(self.STP_VERSION_RE, line):
                        lines += [line]

        assert lines == []


class TestMirrorShape:
    wire = make_polygon(
        {"x": [4, 6, 6, 4], "y": [5, 5, 5, 5], "z": [0, 0, 2, 2]}, closed=True
    )
    face = BluemiraFace(wire)
    solid = extrude_shape(face, (0, 3, 0))
    shell = solid.boundary[0]

    shapes = (wire, face, solid, shell)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize(
        "direction",
        [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 0, -1), (0, -1, 0), (-1, 0, 0)],
    )
    def test_base_mirror(self, shape, direction):
        m_shape = mirror_shape(shape, (0, 0, 0), direction)
        cog = shape.center_of_mass
        m_cog = m_shape.center_of_mass
        new_cog = cog
        idx = np.nonzero(list(direction))[0]
        new_cog[idx] = -cog[idx]
        assert np.isclose(m_shape.volume, shape.volume)
        assert np.allclose(m_cog, new_cog)

    @pytest.mark.parametrize("shape", shapes)
    def test_awkward_mirror(self, shape):
        m_shape = mirror_shape(shape, (4, 5, 0), (-1, -1, 0))
        assert np.isclose(m_shape.volume, shape.volume)
        cog = shape.center_of_mass
        m_cog = m_shape.center_of_mass
        assert not np.allclose(m_cog, cog)

    @pytest.mark.parametrize("shape", shapes)
    def test_bad_direction(self, shape):
        with pytest.raises(GeometryError):
            mirror_shape(shape, base=(0, 0, 0), direction=(EPS, EPS, EPS))


class TestFilletChamfer2D:
    closed_rectangle = make_polygon({"x": [0, 2, 2, 0], "z": [0, 0, 2, 2]}, closed=True)
    open_rectangle = make_polygon({"x": [0, 2, 2, 0], "z": [0, 0, 2, 2]}, closed=False)

    @pytest.mark.parametrize("wire", [closed_rectangle, open_rectangle])
    @pytest.mark.parametrize("radius", [0, 0.1, 0.2, 0.3, 0.5])
    def test_simple_rectangle_fillet(self, wire, radius):
        n = 4 if wire.is_closed() else 2
        correct_length = wire.length - n * 2 * radius
        correct_length += n * np.pi / 2 * radius
        result = fillet_wire_2D(wire, radius)
        assert np.isclose(result.length, correct_length)

    @pytest.mark.parametrize("wire", [closed_rectangle, open_rectangle])
    @pytest.mark.parametrize("radius", [0, 0.1, 0.2, 0.3, 0.5])
    def test_simple_rectangle_chamfer(self, wire, radius):
        result = chamfer_wire_2D(wire, radius)
        n = 4 if wire.is_closed() else 2
        # I'll be honest, I don't understand why this modified radius happens...
        # I worry about what happens at other angles...
        radius = 0.5 * np.sqrt(2) * radius
        correct_length = wire.length - n * 2 * radius
        correct_length += n * np.sqrt(2 * radius**2)

        assert np.isclose(result.length, correct_length)

    @pytest.mark.parametrize("func", [fillet_wire_2D, chamfer_wire_2D])
    def test_what_happens_with_two_tangent_edges(self, func):
        w1 = make_polygon({"x": [0, 1], "z": [0, 0]})
        w2 = make_polygon({"x": [1, 2], "z": [0, 0]})
        wire = BluemiraWire([w1, w2])

        result = func(wire, 0.2)
        assert wire.length == result.length

    @pytest.mark.parametrize("func", [fillet_wire_2D, chamfer_wire_2D])
    def test_GeometryError_on_non_planar_wire(self, func):
        three_d_wire = make_polygon(
            {
                "x": [0, 1, 2, 3, 4, 5],
                "y": [0, -1, -2, 0, 1, 2],
                "z": [0, 1, 2, 1, 0, -1],
            }
        )
        with pytest.raises(GeometryError):
            func(three_d_wire, 0.2)

    @pytest.mark.parametrize("func", [fillet_wire_2D, chamfer_wire_2D])
    def test_GeometryError_on_negative_radius(self, func):
        with pytest.raises(GeometryError):
            func(self.open_rectangle, -0.01)


class TestBooleanFragments:
    @staticmethod
    def _make_pipes(r1, dr1, r2, dr2, x1, y1, x2, y2):
        p11 = make_circle(r1 + dr1, (x1, y1, 0), axis=(1, 0, 0))
        p12 = make_circle(r1, (x1, y1, 0), axis=(1, 0, 0))
        p21 = make_circle(r2 + dr2, (x2, y2, 0), axis=(0, 1, 0))
        p22 = make_circle(r2, (x2, y2, 0), axis=(0, 1, 0))
        pipe_1 = extrude_shape(BluemiraFace([p11, p12]), vec=(10, 0, 0))
        pipe_2 = extrude_shape(BluemiraFace([p21, p22]), vec=(0, 10, 0))
        return pipe_1, pipe_2

    @pytest.mark.parametrize(
        ("r1", "r2", "n_expected", "n_unique"), [(1.0, 1.0, 1, 9), (2.0, 1.0, 2, 8)]
    )
    def test_pipe_pipe_fragments(self, r1, r2, n_expected, n_unique):
        pipes = self._make_pipes(r1, 0.3, r2, 0.3, 0, 0, 5, -5)
        compound, mapping = boolean_fragments(pipes, tolerance=0.0)
        assert len(mapping) == 2
        assert len(mapping[0]) == 5
        assert len(mapping[1]) == 5
        n_shared = self.get_shared_fragments(*mapping)
        assert n_shared == n_expected
        n_unique_actual = len(compound.solids)
        assert n_unique_actual == n_unique

    @pytest.mark.parametrize(("r1", "r2"), [(2.0, 1.0), (4.0, 2.0)])
    def test_pipe_half_pipe_fragments(self, r1, r2):
        pipes = self._make_pipes(r1, 0.3, r2, 0.3, 0, 0, 5, -10)
        compound, mapping = boolean_fragments(pipes, tolerance=0.0)
        assert len(mapping) == 2
        assert len(mapping[0]) == 3
        assert len(mapping[1]) == 3
        n_shared = self.get_shared_fragments(*mapping)
        assert n_shared == 1
        n_unique = len(compound.solids)
        assert n_unique == 5

    def test_no_shared_fragments(self):
        pipe_1 = extrude_shape(
            BluemiraFace(make_circle(1.0, center=(0, 0, 0), axis=(0, 1, 0))),
            vec=(0, 2, 0),
        )
        pipe_2 = extrude_shape(
            BluemiraFace(make_circle(1.0, center=(3, 0, 0), axis=(0, 1, 0))),
            vec=(0, 2, 0),
        )
        compound, mapping = boolean_fragments([pipe_1, pipe_2])
        assert len(mapping) == 2
        assert len(mapping[0]) == 0
        assert len(mapping[1]) == 0
        n_shared = self.get_shared_fragments(*mapping)
        assert n_shared == 0
        n_unique = len(compound.solids)
        assert n_unique == 2

    @staticmethod
    def get_shared_fragments(group_1, group_2):
        count = 0
        for sol in group_1:
            for sol_2 in group_2:
                if sol.is_same(sol_2):
                    count += 1
        return count


class TestForceWireToSpline:
    filename = "vvts_face_polygon_test.json"

    @classmethod
    def setup_class(cls):
        path = get_bluemira_path("geometry/test_data", subfolder="tests")
        with open(Path(path, cls.filename)) as file:
            data = json.load(file)

        cls.wires = [
            deserialize_shape(wire) for wire in data["BluemiraFace"]["boundary"]
        ]

    @mock.patch("bluemira.geometry.tools.bluemira_warn")
    def test_force_spline(self, bm_warn):
        new_wires = []
        n_edges_max = 200
        for wire in self.wires:
            new_wire = force_wire_to_spline(wire, n_edges_max=n_edges_max)
            new_wires.append(new_wire)
            assert len(new_wire.edges) < n_edges_max
        bm_warn.assert_not_called()

        wires = self.wires
        wires.sort(key=lambda wire: -wire.length)
        new_wires.sort(key=lambda wire: -wire.length)
        for w, nw in zip(wires, new_wires):
            np.testing.assert_almost_equal(w.length, nw.length, decimal=3)

    @mock.patch("bluemira.geometry.tools.bluemira_warn")
    def test_warning_raised(self, bm_warn):
        new_wire = force_wire_to_spline(
            self.wires[0], n_edges_max=200, l2_tolerance=1e-10
        )
        bm_warn.assert_called_once()

    @mock.patch("bluemira.geometry.tools.bluemira_debug")
    def test_simple_wire_returns_itself(self, bm_debug):
        p = make_circle(radius=1.0)
        p2 = force_wire_to_spline(p)
        assert p2 == p
        bm_debug.assert_called_once()
