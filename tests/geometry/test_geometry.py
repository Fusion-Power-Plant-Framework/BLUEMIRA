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

import math

import numpy as np
import pytest
from scipy.special import ellipe

import bluemira.codes._freecadapi as cadapi
from bluemira.base.constants import EPS
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import (
    boolean_cut,
    boolean_fuse,
    circular_pattern,
    extrude_shape,
    make_circle,
    make_circle_arc_3P,
    make_ellipse,
    make_polygon,
    offset_wire,
    revolve_shape,
)
from bluemira.geometry.wire import BluemiraWire


def param_face(*coords):
    return [
        BluemiraFace(
            make_polygon(
                c,
                label=f"wire{no}",
                closed=True,
            )
        )
        for no, c in enumerate(coords, start=1)
    ]


class TestGeometry:
    @classmethod
    def setup_class(cls):
        cls.square_points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ]

    def test_create_wire(self):
        wire = make_polygon(self.square_points, label="test", closed=False)
        assert wire.length == pytest.approx(3.0, rel=0, abs=EPS)
        assert wire.area == pytest.approx(0.0, rel=0, abs=EPS)
        assert wire.volume == pytest.approx(0.0, rel=0, abs=EPS)
        assert wire.label == "test"
        assert not wire.is_closed()

    def test_close_wire(self):
        wire = make_polygon(self.square_points, label="test", closed=True)
        assert wire.length == pytest.approx(4.0, rel=0, abs=EPS)
        assert wire.area == pytest.approx(0.0, rel=0, abs=EPS)
        assert wire.volume == pytest.approx(0.0, rel=0, abs=EPS)
        assert wire.label == "test"
        assert wire.is_closed()

    def test_add_wires(self):
        sq_points = np.array(self.square_points)
        half_sq = sq_points[:3, :].T
        half_sq_2 = sq_points[2:, :].T
        wire1 = make_polygon(half_sq, label="wire1", closed=False)
        wire2 = make_polygon(half_sq_2, label="wire2", closed=False)
        wire3 = wire1 + wire2
        wire3.label = "wire3"
        assert wire1.length == pytest.approx(2.0, rel=0, abs=EPS)
        assert wire2.length == pytest.approx(1.0, rel=0, abs=EPS)
        assert wire3.length == pytest.approx(3.0, rel=0, abs=EPS)
        wire1 += wire2
        assert wire1.length == pytest.approx(3.0, rel=0, abs=EPS)

    def test_make_circle(self):
        radius = 2.0
        center = [1, 0, 3]
        axis = [0, 1, 0]
        bm_circle = make_circle(radius=radius, center=center, axis=axis)
        assert bm_circle.length == pytest.approx(2 * math.pi * radius, rel=0, abs=EPS)

    def test_make_circle_arc_3P(self):
        p1 = [0, 0, 0]
        p2 = [1, 1, 0]
        p3 = [2, 0, 0]
        bm_circle = make_circle_arc_3P(p1, p2, p3)
        assert bm_circle.length == math.pi

    def test_make_ellipse(self):
        major_radius = 5.0
        minor_radius = 2.0

        bm_ellipse = make_ellipse(
            major_radius=major_radius,
            minor_radius=minor_radius,
        )
        edge = bm_ellipse.boundary[0].Edges[0]

        # ellispe eccentricity
        eccentricity = math.sqrt(1 - (minor_radius / major_radius) ** 2)
        assert eccentricity == edge.Curve.Eccentricity

        # theoretical length
        expected_length = 4 * major_radius * ellipe(eccentricity**2)
        assert pytest.approx(edge.Length) == expected_length

        # WARNING: it seems that FreeCAD implements in a different way
        # Wire.Length and Edge.length giving a result slightly different
        # but enough to make the following assert fail. To be investigated.
        # assert pytest.approx(bm_ellipse.length) == expected_length

    def test_copy_deepcopy(self):
        points = self.square_points
        points.append(self.square_points[0])
        wire1 = make_polygon(points[0:4], label="wire1")
        wire2 = make_polygon(points[3:], label="wire2")
        wire = BluemiraWire([wire1, wire2], label="wire")
        wire_copy = wire.copy()
        wire_deepcopy = wire.deepcopy()

        assert wire_copy.label == wire.label
        assert wire_deepcopy.label == wire.label
        assert wire.length == (wire1.length + wire2.length)
        assert wire.length == wire_copy.length
        assert wire.length == wire_deepcopy.length
        w1_len = wire1.length
        w2_len = wire2.length
        w_len = wire.length

        wire.scale(2)
        assert wire.length == 2 * w_len
        assert wire.length == wire_copy.length
        assert w_len == wire_deepcopy.length
        assert wire1.length == 2 * w1_len
        assert wire2.length == 2 * w2_len

        wire_copy = wire.copy("wire_copy")
        wire_deepcopy = wire.deepcopy("wire_deepcopy")

        assert wire_copy.label == "wire_copy"
        assert wire_deepcopy.label == "wire_deepcopy"

    params_for_fuse_wires = (
        pytest.param(
            [
                make_polygon([[0, 1, 1], [0, 0, 1], [0, 0, 0]], label="wire1"),
                make_polygon([[0, 1, 1], [0, 0, 1], [0, 0, 0]], label="wire2"),
            ],
            (2, False),
            id="coincident",
            marks=pytest.mark.xfail(reason="coincident wires"),
        ),
        pytest.param(
            [
                make_polygon([[0, 1, 1], [0, 0, 1], [0, 0, 0]], label="wire1"),
                make_polygon([[1, 0, 0], [1, 1, 0], [0, 0, 0]], label="wire2"),
            ],
            (4, True),
            id="closed",
        ),
        pytest.param(
            [
                make_polygon(
                    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0.5, 1, 0]], label="wire1"
                ),
                make_polygon([[1, 0, 0], [1, 1, 0], [0, 0, 0]], label="wire2"),
            ],
            (4, True),
            id="overlap",
            marks=pytest.mark.xfail(reason="wire partially overlap"),
        ),
        pytest.param(
            [
                make_polygon([[0, 1, -1], [0, 0, 1], [0, 0, -1]], label="wire1"),
                make_polygon([[1, 0, 0], [1, 1, 0], [0, 0, 0]], label="wire2"),
            ],
            (4, True),
            id="intersection",
            marks=pytest.mark.xfail(reason="wires internal intersection"),
        ),
    )

    @pytest.mark.parametrize(("test_input", "expected"), params_for_fuse_wires)
    def test_fuse_wires(self, test_input, expected):
        wire_fuse = boolean_fuse(test_input)
        assert (wire_fuse.length, wire_fuse.is_closed()) == expected

    params_for_fuse_faces = (
        pytest.param(
            param_face(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            ),
            (4, 1),
            id="coincident",
        ),
        pytest.param(
            param_face(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                [[1, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]],
            ),
            (6, 2),
            id="1-edge-coincident",
        ),
        pytest.param(
            param_face(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                [[0, 0, 0], [-1, 0, 0], [-1, -1, 0], [0, -1, 0]],
            ),
            (6, 2),
            id="1-vertex-coincident",
            marks=pytest.mark.xfail(reason="Only one vertex intersection"),
        ),
        pytest.param(
            param_face(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                [[0.5, 0.5, 0], [1.5, 0.5, 0], [1.5, 1.5, 0], [0.5, 1.5, 0]],
            ),
            (6, 1.75),
            id="semi intersection",
        ),
    )

    @pytest.mark.parametrize(("test_input", "expected"), params_for_fuse_faces)
    def test_fuse_faces(self, test_input, expected):
        face_fuse = boolean_fuse(test_input)
        assert (
            face_fuse.length,
            face_fuse.area,
        ) == expected

    params_for_cut_wires = (
        pytest.param(
            [
                make_polygon([[0, 1, 1], [0, 0, 1], [0, 0, 0]], label="wire1"),
                make_polygon([[0, 1, 1], [0, 0, 1], [0, 0, 0]], label="wire2"),
            ],
            ([]),
            id="coincident",
        ),
        pytest.param(
            [
                make_polygon([[0, 1, 1], [0, 0, 1], [0, 0, 0]], label="wire1"),
                make_polygon([[1, 0, 0], [1, 1, 0], [0, 0, 0]], label="wire2"),
            ],
            [(2, False)],
            id="contact at start and end",
        ),
        pytest.param(
            [
                make_polygon(
                    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0.5, 1, 0]], label="wire1"
                ),
                make_polygon([[1, 0, 0], [1, 1, 0], [0, 0, 0]], label="wire2"),
            ],
            [(2, False)],
            id="overlap",
        ),
        pytest.param(
            [
                make_polygon([[0, 1, 1], [0, 0, 2], [0, 0, 0]], label="wire1"),
                make_polygon([[2, 0, 0], [1, 1, 0], [0, 0, 0]], label="wire2"),
            ],
            [(2, False), (1, False)],
            id="intersection",
        ),
    )

    @pytest.mark.parametrize(("test_input", "expected"), params_for_cut_wires)
    def test_cut_wires(self, test_input, expected):
        wire_cut = boolean_cut(test_input[0], test_input[1:])
        output = [(w.length, w.is_closed()) for w in wire_cut]
        assert output == expected

    params_for_cut_faces = (
        pytest.param(
            param_face(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            ),
            [],
            id="coincident",
        ),
        pytest.param(
            param_face(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                [[1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0]],
            ),
            [(4, 1)],
            id="1-edge-coincident",
        ),
        pytest.param(
            param_face(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                [[0, 0, 0], [-1, 0, 0], [-1, -1, 0], [0, -1, 0]],
            ),
            [(4, 1)],
            id="1-vertex-coincident",
            # marks=pytest.mark.xfail(reason="Only one vertex intersection"),
        ),
        pytest.param(
            param_face(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                [[0.5, 0.5, 0], [1.5, 0.5, 0], [1.5, 1.5, 0], [0.5, 1.5, 0]],
            ),
            [(4, 0.75)],
            id="semi intersection",
        ),
    )

    @pytest.mark.parametrize(("test_input", "expected"), params_for_cut_faces)
    def test_cut_faces(self, test_input, expected):
        face_cut = boolean_cut(test_input[0], test_input[1:])
        output = [(f.length, f.area) for f in face_cut]
        assert output == expected

    @staticmethod
    def _compare_fc_bm(fc_shape, bm_shape):
        faces = bm_shape.boundary[0].boundary
        fc_faces = fc_shape.Shells[0].Faces
        for f, fc in zip(faces, fc_faces):
            assert f.area == fc.Area
            assert f._orientation.value == fc.Orientation
            for w, fw in zip(f.boundary, fc.Wires):
                assert w.length == fw.Length
                assert w._orientation.value == fw.Orientation

    def test_cut_hollow(self):
        x_c = 10
        d_xc = 1.0
        d_zc = 1.0
        inner = make_polygon(
            [
                [x_c - d_xc, 0, -d_zc],
                [x_c + d_xc, 0, -d_zc],
                [x_c + d_xc, 0, d_zc],
                [x_c - d_xc, 0, d_zc],
            ],
            closed=True,
        )
        outer = offset_wire(inner, 1.0, join="intersect")
        face = BluemiraFace(outer)
        solid = revolve_shape(face, degree=360)

        face_2 = BluemiraFace(inner)
        solid_2 = revolve_shape(face_2, degree=360)
        solid = boolean_cut(solid, solid_2)[0]

        true_volume = 2 * np.pi * x_c * (4**2 - 2**2)
        assert solid.is_valid()
        assert np.isclose(solid.volume, true_volume)

    def test_cut_hollow_circle(self):
        # TODO: More fun to be had with circles...
        x_c = 10
        radius = 1
        circle = make_circle(radius=radius, center=[10, 0, 0], axis=[0, 1, 0])
        face = BluemiraFace(circle)
        solid = revolve_shape(face, degree=360)

        circle_2 = make_circle(radius=0.5 * radius, center=[10, 0, 0], axis=[0, 1, 0])
        face_2 = BluemiraFace(circle_2)
        solid_2 = revolve_shape(face_2, degree=360)

        solid = boolean_cut(solid, solid_2)[0]

        true_volume = 2 * np.pi * x_c * (np.pi * (radius**2 - (0.5 * radius) ** 2))
        assert solid.is_valid()
        assert np.isclose(solid.volume, true_volume)

    @staticmethod
    def _setup_faces():
        return param_face(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            [[-1, 0, 1], [2, 0, 1], [2, 1, 1], [-1, 1, 1]],
        )

    @pytest.mark.parametrize("direction", [1, -1])
    def test_fuse_solids(self, direction):
        face, face2 = self._setup_faces()
        solid = extrude_shape(face, (0, 0, direction * 5))
        solid2 = extrude_shape(face2, (0, 0, direction))

        result = boolean_fuse([solid, solid2])
        fc_result = cadapi.boolean_fuse([solid.shape, solid2.shape])
        assert result.is_valid()
        assert fc_result.isValid()
        self._compare_fc_bm(fc_result, result)
        assert result.volume > solid.volume
        assert result.volume > solid2.volume

    @pytest.mark.parametrize("direction", [1, -1])
    def test_cut_solids(self, direction):
        face, face2 = self._setup_faces()

        solid = extrude_shape(face, (0, 0, 5))
        solid2 = extrude_shape(face2, (0, 0, direction))

        results = boolean_cut(solid2, solid)
        fc_result = cadapi.boolean_cut(solid2.shape, solid.shape)
        assert len(results) == len(fc_result) == 2

        for fc_shape, bm_shape in zip(fc_result, results):
            fc_shape.isValid()
            bm_shape.is_valid()
            self._compare_fc_bm(fc_shape, bm_shape)
            assert bm_shape.volume < solid2.volume


class TestShapeTransformations:
    @classmethod
    def setup_class(cls):
        cls.wire = make_polygon(
            [
                (4.0, -0.5, 0.0),
                (5.0, -0.5, 0.0),
                (5.0, 0.5, 0.0),
                (4.0, 0.5, 0.0),
            ],
            closed=True,
            label="test_wire",
        )

        cls.face = BluemiraFace(cls.wire.deepcopy(), label="test_face")
        cls.solid = extrude_shape(cls.face.deepcopy(), (0, 0, 1), label="test_solid")

    @staticmethod
    def _centroids_close(new_centroid, centroid, vector):
        return np.allclose(new_centroid, np.array(centroid) + np.array(vector))

    def test_rotate_wire(self):
        base = (0, 0, 0)
        direction = (0, 0, 1)
        degree = 180
        length = self.wire.length
        orientation = self.wire._orientation
        centroid = np.array(self.wire.center_of_mass)
        self.wire.rotate(base, direction, degree)
        assert self.wire.length == length
        assert self.wire.label == "test_wire"
        assert self.wire._orientation == orientation
        assert self._centroids_close(
            self.wire.center_of_mass, centroid, np.array([-2 * centroid[0], 0, 0])
        )

    def test_rotate_face(self):
        base = (0, 0, 0)
        direction = (0, 0, 1)
        degree = 180
        area = self.face.area
        orientation = self.face._orientation
        centroid = np.array(self.face.center_of_mass)
        self.face.rotate(base, direction, degree)
        assert np.isclose(self.face.area, area)
        assert self.face.label == "test_face"
        assert self.face.boundary[0].label == "test_wire"
        assert self.face._orientation == orientation
        assert self._centroids_close(
            self.face.center_of_mass, centroid, np.array([-2 * centroid[0], 0, 0])
        )

    def test_rotate_solid(self):
        base = (0, 0, 0)
        direction = (0, 0, 1)
        degree = 180
        volume = self.solid.volume
        orientation = self.solid._orientation
        centroid = np.array(self.solid.center_of_mass)
        self.solid.rotate(base, direction, degree)
        assert np.isclose(self.solid.volume, volume)
        assert self.solid.label == "test_solid"
        assert self.solid._orientation == orientation
        assert self._centroids_close(
            self.solid.center_of_mass, centroid, np.array([-2 * centroid[0], 0, 0])
        )

    def test_translate_wire(self):
        dx = 1.0
        dy = 2.0
        dz = 3.0
        vector = (dx, dy, dz)
        centroid = self.wire.center_of_mass
        self.wire.translate(vector)
        assert self._centroids_close(self.wire.center_of_mass, centroid, vector)
        assert self.wire.label == "test_wire"

    def test_translate_face(self):
        dx = 1.0
        dy = 2.0
        dz = 3.0
        vector = (dx, dy, dz)
        centroid = self.face.center_of_mass
        self.face.translate(vector)
        assert self._centroids_close(self.face.center_of_mass, centroid, vector)
        assert self.face.label == "test_face"
        assert self.face.boundary[0].label == "test_wire"

    def test_translate_solid(self):
        dx = 1.0
        dy = 2.0
        dz = 3.0
        vector = (dx, dy, dz)
        centroid = self.solid.center_of_mass
        self.solid.translate(vector)
        assert self._centroids_close(self.solid.center_of_mass, centroid, vector)

    def test_scale_wire(self):
        scale_factor = 3
        length = self.wire.length
        self.wire.scale(scale_factor)
        assert np.isclose(self.wire.length, scale_factor * length)
        assert self.wire.label == "test_wire"

    def test_scale_face(self):
        scale_factor = 3
        area = self.face.area
        self.face.scale(scale_factor)
        assert np.isclose(self.face.area, scale_factor**2 * area)
        assert self.face.label == "test_face"
        assert self.face.boundary[0].label == "test_wire"

    def test_scale_solid(self):
        scale_factor = 3
        volume = self.solid.volume
        self.solid.scale(scale_factor)
        assert np.isclose(self.solid.volume, scale_factor**3 * volume)
        assert self.solid.label == "test_solid"


def test_circular_pattern():
    wire = make_polygon(
        [
            (4.0, -0.5, 0.0),
            (5.0, -0.5, 0.0),
            (5.0, 0.5, 0.0),
            (4.0, 0.5, 0.0),
        ],
        closed=True,
    )
    face = BluemiraFace(wire)
    solid = extrude_shape(face, (0, 0, 1))
    shapes = circular_pattern(solid, degree=360, n_shapes=5)
    assert len(shapes) == 5
    for s in shapes:
        assert np.isclose(solid.volume, s.volume)
