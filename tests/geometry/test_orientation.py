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

import pytest

import bluemira.codes._freecadapi as cadapi
import bluemira.geometry.tools as geo_tools
from bluemira.geometry.error import MixedOrientationWireError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.wire import BluemiraWire


class TestOrientation:
    def test_wire(self):
        wire = cadapi.make_polygon(
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 0, 0]],
        )

        bm_wire = BluemiraWire(wire)
        assert bm_wire.boundary[0].Orientation == wire.Orientation
        assert bm_wire.shape.Orientation == wire.Orientation

        wire.reverse()

        bm_wire = BluemiraWire(wire)
        assert bm_wire.boundary[0].Orientation == wire.Orientation
        assert bm_wire.shape.Orientation == wire.Orientation

    def test_face(self):
        wire = cadapi.make_polygon(
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 0, 0]],
        )
        face = cadapi.apiFace(wire)
        bm_face = BluemiraFace._create(face)

        assert bm_face.shape.Orientation == face.Orientation

        face.reverse()
        bm_face = BluemiraFace._create(face)

        assert bm_face.shape.Orientation == face.Orientation

    def test_face_with_hole(self):
        wire = cadapi.make_polygon(
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 0, 0]],
        )
        circle = cadapi.make_circle(radius=10, axis=[0, 1, 0])
        face = cadapi.apiFace([circle, wire])
        bm_face = BluemiraFace._create(face)

        assert bm_face.shape.Orientation == face.Orientation
        assert bm_face.area == face.Area

        face.reverse()
        bm_face = BluemiraFace._create(face)

        assert bm_face.shape.Orientation == face.Orientation
        assert bm_face.area == face.Area

    def test_bad_wire(self):
        wire_1 = cadapi.make_polygon([[0, 0, 0], [1, 0, 0]])
        wire_2 = cadapi.make_polygon([[1, 0, 0], [2, 0, 0]])
        wire_2.reverse()
        with pytest.raises(MixedOrientationWireError):
            bm_wire = BluemiraWire([wire_1, wire_2])


class ExtrudeOrientation:
    """
    Helper class for testing orientations are respected after an extrude operation.
    """

    VERTS = (
        [0.5, 0, -0.5],
        [1.5, 0, -0.5],
        [1.5, 0, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0, -0.5],
    )
    _extrusion: float

    def setup_method(self):
        profile = cadapi.make_polygon(self.VERTS)
        face = cadapi.apiFace(profile)
        self.solid_fc = cadapi.extrude_shape(face, (0, self._extrusion, 0))

        profile = geo_tools.make_polygon(self.VERTS)
        face = BluemiraFace(profile)
        self.solid_bm = geo_tools.extrude_shape(face, (0, self._extrusion, 0))

    @pytest.mark.parametrize("shapes_name", ["wires", "faces", "solids"])
    def test_shapes(self, shapes_name: str):
        shapes_name = shapes_name.capitalize()
        shapes_fc = getattr(self.solid_fc, shapes_name)
        shapes_bm = getattr(self.solid_bm.shape, shapes_name)
        assert all(
            fc.Orientation == bm.Orientation for fc, bm in zip(shapes_fc, shapes_bm)
        )

    def test_edges(self):
        self.test_shapes("edges")

    def test_shells(self):
        self.test_shapes("shells")


class TestExtrudeNegativeOrientation(ExtrudeOrientation):
    """
    Test that orientations are respected after a negative extrude operation.
    """

    _extrusion = -1.0

    @pytest.mark.xfail(
        reason=(
            "Inversion of shell orientation under investigation - no effect on other"
            " results seen so far."
        )
    )
    def test_shells(self):
        super().test_shells()


class TestExtrudePositiveOrientation(ExtrudeOrientation):
    """
    Test that orientations are respected after a positive extrude operation.
    """

    _extrusion = 1.0

    @pytest.mark.xfail(
        reason=(
            "Mismatch of edge orientation under investigation - no effect on other"
            " results seen so far."
        )
    )
    def test_edges(self):
        super().test_edges()
