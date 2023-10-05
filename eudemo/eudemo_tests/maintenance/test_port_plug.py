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

from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_polygon
from eudemo.maintenance.port_plug import make_castellated_plug


class TestCastellationBuilder:
    """Tests the Castellation Builder"""

    @pytest.mark.parametrize(
        ("xi", "xo", "zh", "yw", "vec", "x_offsets", "c_offsets", "exp_v"),
        zip(
            [9.0, 9.0, 6.0],  # x_inboard
            [16.0, 15.0, 9.0],  # x_outboard
            [5.0, 4.0, 2.0],  # z_height
            [3.0, 2.0, 1.0],  # y_widths
            [(1, 0, 0), (1, 0, 0), (1, 0, 0.5)],  # extrusion vectors
            [[1.0], [1.0, 1.0], [0.5]],  # y/z castellation_offsets
            [[3.0], [2.0, 4.0], [1.0]],  # x castellation_positions
            [185.0, 160.0, 12.521980674],  # volume check value of Eq. Ports
        ),
    )
    def test_cst_builder(self, xi, xo, zh, yw, vec, x_offsets, c_offsets, exp_v):
        """Test Castellation Builder"""
        y = (yw / 2.0, -yw / 2.0, -yw / 2.0, yw / 2.0)
        z = (-zh / 2.0, -zh / 2.0, zh / 2.0, zh / 2.0)
        yz_profile = BluemiraFace(make_polygon({"x": xi, "y": y, "z": z}, closed=True))

        out_cst = make_castellated_plug(yz_profile, vec, xo - xi, x_offsets, c_offsets)
        assert np.isclose(out_cst.volume, exp_v)
