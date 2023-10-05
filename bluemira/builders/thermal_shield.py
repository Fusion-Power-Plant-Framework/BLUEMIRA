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
Thermal shield builders
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Type, Union

import numpy as np
from scipy.spatial import ConvexHull

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.builders.tools import (
    apply_component_display_options,
    build_sectioned_xy,
    build_sectioned_xyz,
    find_xy_plane_radii,
    make_circular_xy_ring,
)
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.error import GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import (
    _offset_wire_discretised,
    boolean_cut,
    boolean_fuse,
    force_wire_to_spline,
    make_polygon,
    offset_wire,
)
from bluemira.geometry.wire import BluemiraWire
from bluemira.materials.cache import Void


@dataclass
class VVTSBuilderParams(ParameterFrame):
    """
    VVTS builder parameters
    """

    g_vv_ts: Parameter[float]
    n_TF: Parameter[int]
    tk_ts: Parameter[float]


class VVTSBuilder(Builder):
    """
    Vacuum vessel thermal shield builder
    """

    VVTS = "VVTS"
    VOID = "VVTS voidspace"
    param_cls: Type[VVTSBuilderParams] = VVTSBuilderParams

    def __init__(
        self,
        params: Union[VVTSBuilderParams, Dict],
        build_config: Dict,
        keep_out_zone: BluemiraWire,
    ):
        super().__init__(params, build_config)
        self.keep_out_zone = keep_out_zone

    def build(self) -> Component:
        """
        Build the vacuum vessel thermal shield component.
        """
        xz_vvts, xz_vvts_void = self.build_xz(self.keep_out_zone)
        vvts_face: BluemiraFace = xz_vvts.get_component_properties("shape")
        vvts_void_face: BluemiraFace = xz_vvts_void.get_component_properties("shape")

        return self.component_tree(
            xz=[xz_vvts, xz_vvts_void],
            xy=self.build_xy(vvts_face),
            xyz=self.build_xyz(vvts_face, vvts_void_face, degree=0),
        )

    def build_xz(self, koz: BluemiraWire) -> Tuple[PhysicalComponent, ...]:
        """
        Build the x-z components of the vacuum vessel thermal shield.

        Parameters
        ----------
        koz:
            keep out zone for the thermal shield
        """
        # This split hack works round #1319
        # _offset_wire_discretised used because
        # the cad offset regularly doesn't work properly here.
        # due to topology but unknown why here particularly
        ex_args = {
            "join": "intersect",
            "open_wire": False,
            "ndiscr": 600,
        }
        vvts_inner_wire = _offset_wire_discretised(
            koz, self.params.g_vv_ts.value, **ex_args
        )
        vvts_outer_wire = _offset_wire_discretised(
            koz, self.params.tk_ts.value + self.params.g_vv_ts.value, **ex_args
        )
        vvts_inner_wire = force_wire_to_spline(vvts_inner_wire, n_edges_max=100)
        vvts_outer_wire = force_wire_to_spline(vvts_outer_wire, n_edges_max=100)
        vvts_face = BluemiraFace([vvts_outer_wire, vvts_inner_wire])
        self.vvts_face = vvts_face

        vvts = PhysicalComponent(self.VVTS, vvts_face)
        vvts_void = PhysicalComponent(
            self.VOID, BluemiraFace(vvts_face.boundary[1]), material=Void("vacuum")
        )

        apply_component_display_options(vvts, color=BLUE_PALETTE["TS"][0])
        apply_component_display_options(vvts_void, color=(0, 0, 0))
        return vvts, vvts_void

    @staticmethod
    def build_xy(vvts_face: BluemiraFace) -> List[PhysicalComponent]:
        """
        Build the x-y components of the vacuum vessel thermal shield.

        Parameters
        ----------
        vvts_face:
            xz face to build vvts
        """
        return build_sectioned_xy(vvts_face, BLUE_PALETTE["TS"][0])

    def build_xyz(
        self, vvts_face: BluemiraFace, vvts_void_face: BluemiraFace, degree: float = 360
    ) -> List[PhysicalComponent]:
        """
        Build the x-y-z components of the vacuum vessel thermal shield

        Parameters
        ----------
        vvts_face:
            xz face to build vvts
        degree:
            Revolution degrees
        """
        return build_sectioned_xyz(
            [vvts_face, vvts_void_face],
            [self.VVTS, self.VOID],
            self.params.n_TF.value,
            [BLUE_PALETTE["TS"][0], (0, 0, 0)],
            degree,
            material=[None, Void("vacuum")],
        )


@dataclass
class CryostatTSBuilderParams(ParameterFrame):
    """
    Cryostat thermal shield builder parameters
    """

    g_ts_pf: Parameter[float]
    g_ts_tf: Parameter[float]
    n_TF: Parameter[int]
    tk_ts: Parameter[float]


class CryostatTSBuilder(Builder):
    """
    Cryostat thermal shield builder
    """

    CRYO_TS = "Cryostat TS"
    VOID = "Cryostat voidspace"

    param_cls: Type[CryostatTSBuilderParams] = CryostatTSBuilderParams

    def __init__(
        self,
        params: Union[CryostatTSBuilderParams, Dict],
        build_config: Dict,
        pf_keep_out_zones: List[BluemiraWire],
        tf_keep_out_zone: BluemiraWire,
    ):
        super().__init__(params, build_config)
        self.pf_keep_out_zones = pf_keep_out_zones
        self.tf_keep_out_zone = tf_keep_out_zone

    def build(self) -> Component:
        """
        Build the cryostat thermal shield component.
        """
        xz_cts, xz_cts_void = self.build_xz(
            self.pf_keep_out_zones, self.tf_keep_out_zone
        )
        cts_face: BluemiraFace = xz_cts.get_component_properties("shape")
        cts_void_face: BluemiraFace = xz_cts_void.get_component_properties("shape")

        return self.component_tree(
            xz=[xz_cts, xz_cts_void],
            xy=[self.build_xy(cts_face)],
            xyz=self.build_xyz(cts_face, cts_void_face, degree=0),
        )

    def build_xz(
        self, pf_kozs: List[BluemiraWire], tf_koz: BluemiraWire
    ) -> PhysicalComponent:
        """
        Build the x-z components of the thermal shield.
        """
        x, z = [], []
        for coil in pf_kozs:
            bb = coil.bounding_box
            x.extend([bb.x_min, bb.x_max, bb.x_max, bb.x_min])
            z.extend([bb.z_min, bb.z_min, bb.z_max, bb.z_max])

        # Project extrema slightly beyond axis (might be bad for NT) - will get cut later
        x.extend([-0.5, -0.5])  # [m]
        z.extend([np.min(z), np.max(z)])
        x, z = np.array(x), np.array(z)
        hull_idx = ConvexHull(np.array([x, z]).T).vertices

        pf_o_wire = offset_wire(
            make_polygon({"x": x[hull_idx], "y": 0, "z": z[hull_idx]}, closed=True),
            self.params.g_ts_pf.value + self.params.tk_ts.value,
            open_wire=False,
            ndiscr=600,
        )

        tf_o_wire = offset_wire(
            tf_koz,
            self.params.g_ts_tf.value,
            join="arc",
            open_wire=False,
            ndiscr=600,
        )

        try:
            cts_inner = boolean_fuse(
                [BluemiraFace(pf_o_wire), BluemiraFace(tf_o_wire)]
            ).boundary[0]
        except GeometryError:
            # TODO: boolean_fuse probably shouldn't throw an error here...
            # the TF offset face is probably enclosed by the PF offset face
            cts_inner = pf_o_wire

        cts_outer = offset_wire(cts_inner, self.params.tk_ts.value, ndiscr=600)
        cts_face = BluemiraFace([cts_outer, cts_inner])
        bb = cts_face.bounding_box
        x_in, x_out = 0, -bb.x_max
        cutter = BluemiraFace(
            make_polygon(
                {
                    "x": [x_in, x_out, x_out, x_in],
                    "y": 0,
                    "z": [bb.z_min, bb.z_min, bb.z_max, bb.z_max],
                },
                closed=True,
            )
        )

        cts = boolean_cut(cts_face, cutter)[0]
        cts_void_wire = boolean_cut(cts_inner, cutter)[0]
        cts_void_wire.close()
        cts_void_face = BluemiraFace(cts_void_wire)

        cryostat_ts = PhysicalComponent(self.CRYO_TS, cts)
        cryostat_ts_void = PhysicalComponent(
            self.VOID, cts_void_face, material=Void("vacuum")
        )

        apply_component_display_options(cryostat_ts, color=BLUE_PALETTE["TS"][0])
        apply_component_display_options(cryostat_ts_void, color=(0, 0, 0))
        return cryostat_ts, cryostat_ts_void

    def build_xy(self, cts_face: BluemiraFace) -> PhysicalComponent:
        """
        Build the x-y components of the thermal shield.
        """
        mid_plane = BluemiraPlane.from_3_points([0, 0, 0], [1, 0, 0], [1, 1, 0])
        r_in, r_out = find_xy_plane_radii(cts_face.boundary[0], mid_plane)

        cryostat_ts = PhysicalComponent(self.CRYO_TS, make_circular_xy_ring(r_in, r_out))
        apply_component_display_options(cryostat_ts, color=BLUE_PALETTE["TS"][0])

        return cryostat_ts

    def build_xyz(
        self, cts_face: BluemiraFace, cts_void_face: BluemiraFace, degree: float = 360
    ) -> List[PhysicalComponent]:
        """
        Build the x-y-z components of the thermal shield.
        """
        return build_sectioned_xyz(
            [cts_face, cts_void_face],
            [self.CRYO_TS, self.VOID],
            self.params.n_TF.value,
            [BLUE_PALETTE["TS"][0], (0, 0, 0)],
            degree,
            enable_sectioning=True,
            material=[None, Void("vacuum")],
        )
