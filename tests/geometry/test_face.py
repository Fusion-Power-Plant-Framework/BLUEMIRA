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

from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import (
    PictureFrame,
    PrincetonD,
    SextupleArc,
    TripleArc,
)
from bluemira.geometry.tools import (
    extrude_shape,
    make_bezier,
    make_circle,
    make_ellipse,
    make_polygon,
    offset_wire,
)


class TestBluemiraFace:
    @classmethod
    def setup_class(cls):
        polygon = make_polygon(
            [[4, -2, 0], [6, -2, 0], [6, 2, 0], [4, 2, 0]],
            closed=True,
        )
        princeton = PrincetonD().create_shape(n_points=150)
        triple = TripleArc().create_shape()
        sextuple = SextupleArc().create_shape()
        tapered = PictureFrame(inner="TAPERED_INNER").create_shape()
        cls.shapes = [polygon, princeton, triple, sextuple, tapered]

    def test_single_complicated(self):
        for shape in self.shapes:
            face = BluemiraFace(shape)
            assert face.is_valid()
            assert not face.is_null()
            assert face.area > 0.0

    @pytest.mark.parametrize("offset", [0.5, -0.5])
    def test_two_complicated(self, offset):
        direction = int(offset / abs(offset))
        for shape in self.shapes:
            wire = offset_wire(shape, offset, join="arc")
            face_list = [wire, shape][::direction]
            face = BluemiraFace(face_list)
            assert not face.is_null()
            assert face.is_valid()
            assert np.isclose(
                face.area,
                BluemiraFace(face_list[0]).area - BluemiraFace(face_list[1]).area,
            )

    def test_two_offsets(self):
        for shape in self.shapes:
            outer = offset_wire(shape, 0.5, join="arc")
            inner = offset_wire(shape, -0.5, join="arc")
            face = BluemiraFace([outer, inner])
            assert not face.is_null()
            assert face.is_valid()
            assert face.area > 0.0

    def test_face_vertices(self):
        points = Coordinates(
            {"x": [0, 1, 2, 1, 0, -1, 0], "y": [-2, -1, 0, 1, 2, 1, -2], "z": 0}
        )
        wire = make_polygon(points, closed=True)
        face = BluemiraFace(wire)
        vertices = face.vertexes
        assert len(vertices) == len(points) - 1


class TestNormalAt:
    normals = (
        (0, 1, 0),
        (1, 0, 0),
        (0, 0, 1),
        (0, -1, 0),
        (-1, 0, 0),
        (0, 0, -1),
        (1, 3, -4),
        (-10, -135.2, 234.5),
    )

    wires = (
        make_circle(axis=(1, 0, 0)),
        make_ellipse(major_axis=(0, 0, 1), minor_axis=(0, 1, 0)),
        make_bezier([[0, 0, 0], [1, 0, 0], [1, 1, 0]], closed=True),
    )

    @pytest.mark.parametrize("normal", normals)
    def test_circle_normal(self, normal):
        normal = normal / np.linalg.norm(normal)
        circle = BluemiraFace(make_circle(axis=normal))
        np.testing.assert_allclose(circle.normal_at(), normal)

    @pytest.mark.parametrize(
        "alphas", [np.random.default_rng().random(2) for _ in range(5)]
    )
    def test_xy_polygon_normal(self, alphas):
        xy_polygon = BluemiraFace(
            make_polygon(
                [[4, -2, 0], [6, -2, 0], [6, 2, 0], [4, 2, 0]],
                closed=True,
            )
        )

        np.testing.assert_allclose(xy_polygon.normal_at(*alphas), (0, 0, 1))

    @pytest.mark.parametrize("wire", wires)
    def test_curved_solid_face_normals(self, wire):
        face = BluemiraFace(wire)
        solid = extrude_shape(face, (10, 1, 1))
        biggest_face = sorted(solid.faces, key=lambda face: face.area)[-1]
        normal_1 = biggest_face.normal_at(0, 0)
        normal_2 = biggest_face.normal_at(0.5, 0.5)
        assert not np.allclose(normal_1, normal_2)

    @pytest.mark.parametrize("wire", wires)
    def test_curved_shell_face_normals(self, wire):
        shell = extrude_shape(wire, (10, 1, 1))
        biggest_face = sorted(shell.faces, key=lambda face: face.area)[-1]
        normal_1 = biggest_face.normal_at(0, 0)
        normal_2 = biggest_face.normal_at(0.5, 0.5)
        assert not np.allclose(normal_1, normal_2)
