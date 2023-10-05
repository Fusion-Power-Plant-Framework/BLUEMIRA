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

from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import boolean_fuse, extrude_shape, make_circle
from eudemo.maintenance.duct_connection import pipe_pipe_join


class TestPipePipeJoin:
    c1 = make_circle(radius=2.0, center=(5, -5, 0), axis=(0, 1, 0))
    c2 = make_circle(radius=2.2, center=(5, -5, 0), axis=(0, 1, 0))
    face = BluemiraFace([c2, c1])
    void_face = BluemiraFace(c1)
    target = extrude_shape(face, vec=(0, 10, 0))
    target_void = extrude_shape(void_face, vec=(0, 10, 0))

    @pytest.mark.parametrize(("tool_length", "n_faces"), [(5, 7), (10, 10)])
    def test_pipe_pipe_join(self, tool_length, n_faces):
        c1 = make_circle(radius=1.0, center=(5, 0, 5), axis=(0, 0, 1))
        c2 = make_circle(radius=1.1, center=(5, 0, 5), axis=(0, 0, 1))
        face = BluemiraFace([c2, c1])
        void_face = BluemiraFace(c1)
        tool = extrude_shape(face, vec=(0, 0, -tool_length))
        tool_void = extrude_shape(void_face, vec=(0, 0, -tool_length))
        new_shape_pieces = pipe_pipe_join(self.target, self.target_void, tool, tool_void)
        n_actual = len(boolean_fuse(new_shape_pieces).faces)
        assert n_actual == n_faces
