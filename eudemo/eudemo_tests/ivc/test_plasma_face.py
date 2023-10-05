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
Test PlasmaFaceDesigner
"""

from typing import ClassVar

import pytest

from bluemira.geometry.tools import make_polygon
from eudemo.ivc import PlasmaFaceDesigner


class TestPlasmaFaceDesigner:
    _params: ClassVar = {
        "div_type": {"value": "SN", "unit": ""},
        "c_rm": {"value": 0.02, "unit": "m"},
    }

    @classmethod
    def setup_class(cls):
        cls.wall_boundary = make_polygon(
            [[1, 0, -10], [1, 0, 10], [11, 0, 10], [11, 0, -10]]
        )
        cls.divertor_silhouette = (
            make_polygon([[1, 0, -10], [6, 0, -15]]),
            make_polygon([[11, 0, -10], [6, 0, -15]]),
        )
        cls.ivc_boundary = make_polygon(
            [[0, 0, -16], [0, 0, 11], [12, 0, 11], [12, 0, -16]], closed=True
        )

    def test_wall_boundary_is_cut_below_x_point_in_z_axis(self):
        designer = PlasmaFaceDesigner(
            self._params, self.ivc_boundary, self.wall_boundary, self.divertor_silhouette
        )

        blanket_face, divertor_face = designer.execute()
        # U shape
        bf_area = 20 * 2 + 1 * 12
        assert pytest.approx(blanket_face.area) == bf_area - designer.params.c_rm.value
        # Rectangle with a triangle cut out
        # cut removed a slice of triangle with rm section
        div_area = 6 * 12 - 10 * 5 * 0.5
        extra_triangles = 0.00005 * 2
        assert (
            pytest.approx(divertor_face.area)
            == div_area - designer.params.c_rm.value - extra_triangles
        )
