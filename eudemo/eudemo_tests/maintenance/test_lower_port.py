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
Tests for EU-DEMO Lower Port
"""

import numpy as np
import pytest

from bluemira.base.parameter_frame import make_parameter_frame
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_circle, make_polygon
from bluemira.geometry.wire import BluemiraWire
from eudemo.maintenance.lower_port.builder import TSLowerPortDuctBuilder
from eudemo.maintenance.lower_port.duct_designer import (
    LowerPortKOZDesigner,
    LowerPortKOZDesignerParams,
)


class TestLowerPort:
    """Test Lower Port Duct Designer"""

    @classmethod
    def setup_class(cls):
        cls.divertor_xz_silhouette = BluemiraWire(
            [
                make_polygon(
                    [
                        [4, 6],
                        [0, 0],
                        [0, 0],
                    ]
                ),
                make_circle(
                    1, center=(5, 0, 0), start_angle=-180, end_angle=0, axis=(0, -1, 0)
                ),
            ]
        )
        cls.tf_coils_outer_boundary = BluemiraWire(
            [
                make_polygon(
                    [
                        [3, 3],
                        [0, 0],
                        [8, -8],
                    ]
                ),
                make_circle(
                    8, center=(3, 0, 0), start_angle=-90, end_angle=90, axis=(0, -1, 0)
                ),
            ]
        )

    def setup_method(self):
        self.duct_des_params = make_parameter_frame(
            {
                "n_TF": {"value": 10, "unit": "dimensionless"},
                "n_div_cassettes": {"value": 3, "unit": "dimensionless"},
                "lower_port_angle": {"value": -30, "unit": "degrees"},
                "g_ts_tf": {"value": 0.05, "unit": "m"},
                "tk_ts": {"value": 0.05, "unit": "m"},
                "g_vv_ts": {"value": 0.05, "unit": "m"},
                "tk_vv_single_wall": {"value": 0.06, "unit": "m"},
                "tf_wp_depth": {"value": 0.5, "unit": "m"},
                "lp_duct_div_pad_ob": {"value": 0.3, "unit": "m"},
                "lp_duct_div_pad_ib": {"value": 0.1, "unit": "m"},
                "lp_height": {"value": 4.5, "unit": "m"},
                "lp_width": {"value": 3, "unit": "m"},
            },
            LowerPortKOZDesignerParams,
        )

    @pytest.mark.parametrize("duct_angle", [0, -30, -45, -60, -90])
    def test_duct_angle(self, duct_angle):
        self.duct_des_params.lower_port_angle.value = duct_angle

        (
            lp_duct_xz_void_space,
            lp_duct_xz_koz,
            lp_duct_angled_nowall_extrude_boundary,
            lp_duct_straight_nowall_extrude_boundary,
        ) = LowerPortKOZDesigner(
            self.duct_des_params,
            {},
            self.divertor_xz_silhouette,
            self.tf_coils_outer_boundary,
        ).execute()

        builder = TSLowerPortDuctBuilder(
            self.duct_des_params,
            {},
            lp_duct_angled_nowall_extrude_boundary,
            lp_duct_straight_nowall_extrude_boundary,
            15,
        )
        lp_duct = builder.build()

        # make angle plane
        x_angled_start = lp_duct_angled_nowall_extrude_boundary.vertexes.x[0]
        z_angled_start = lp_duct_angled_nowall_extrude_boundary.vertexes.z[0]
        pl = BluemiraFace(
            make_polygon(
                [
                    [
                        x_angled_start,
                        x_angled_start + 1,
                        x_angled_start + 1,
                        x_angled_start,
                    ],
                    [0.1, 0.1, -0.1, -0.1],
                    [z_angled_start] * 4,
                ],
                closed=True,
            )
        )
        pl.rotate(
            degree=duct_angle,
            base=(x_angled_start, 0, z_angled_start),
            direction=(0, -1, 0),
        )
        pl.rotate(degree=180 / self.duct_des_params.n_TF.value)

        duct_xyz_cad = lp_duct.get_component("xyz").get_component(builder.name).shape

        angled_face = duct_xyz_cad.faces[0]  # this was an angled face when I tested it

        np.testing.assert_allclose(angled_face.normal_at(), pl.normal_at(), atol=1e-8)
