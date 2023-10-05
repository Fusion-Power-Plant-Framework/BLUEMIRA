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
Test ivc boundary designer.
"""
from typing import ClassVar

import pytest

from bluemira.base.error import DesignError
from bluemira.geometry.parameterisations import PictureFrame
from bluemira.geometry.tools import make_circle, signed_distance
from eudemo.ivc import IVCBoundaryDesigner


class TestIVCBoundaryDesigner:
    picture_frame: ClassVar = PictureFrame(
        {"ro": {"value": 6}, "ri": {"value": 3}}
    ).create_shape()
    params: ClassVar = {
        "tk_bb_ib": {"value": 0.8, "unit": "m"},
        "tk_bb_ob": {"value": 1.1, "unit": "m"},
        "ib_offset_angle": {"value": 45, "unit": "degree"},
        "ob_offset_angle": {"value": 175, "unit": "degree"},
    }

    def test_DesignError_given_wall_shape_not_closed(self):
        wall_shape = make_circle(end_angle=180)

        with pytest.raises(DesignError):
            IVCBoundaryDesigner(self.params, wall_shape)

    def test_design_returns_boundary_that_does_not_intersect_wire(self):
        designer = IVCBoundaryDesigner(self.params, self.picture_frame)

        wire = designer.execute()

        assert signed_distance(wire, self.picture_frame) < 0
