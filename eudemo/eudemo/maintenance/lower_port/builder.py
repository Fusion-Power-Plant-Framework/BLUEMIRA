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
EUDEMO Lower Port Builder
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

if TYPE_CHECKING:
    from bluemira.geometry.solid import BluemiraSolid
    from bluemira.geometry.wire import BluemiraWire

import numpy as np

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.error import BuilderError
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.builders.tools import apply_component_display_options
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import boolean_cut, boolean_fuse, extrude_shape, offset_wire
from bluemira.materials import Void


@dataclass
class TSLowerPortDuctBuilderParams(ParameterFrame):
    """Thermal Shield Lower Port Duct Builder Parameters"""

    n_TF: Parameter[int]
    lower_port_angle: Parameter[float]
    tk_ts: Parameter[float]


class TSLowerPortDuctBuilder(Builder):
    """
    Thermal Shield Lower Port Duct Builder
    """

    param_cls = TSLowerPortDuctBuilderParams

    def __init__(
        self,
        params: Union[ParameterFrame, Dict],
        build_config: Dict,
        duct_angled_nowall_extrude_boundary: BluemiraWire,
        duct_straight_nowall_extrude_boundary: BluemiraWire,
        x_straight_end: float,
    ):
        super().__init__(params, build_config)
        self.duct_angled_boundary = duct_angled_nowall_extrude_boundary
        self.duct_straight_boundary = duct_straight_nowall_extrude_boundary
        self.x_straight_end = x_straight_end

    def build(self) -> Component:
        """
        Build the thermal shield lower port.
        """
        return self.component_tree(
            xz=None,
            xy=None,
            xyz=self.build_xyz(),
        )

    def build_xyz(self) -> List[PhysicalComponent]:
        """
        Build the thermal shield lower port in x-y-z.
        """
        duct, void = build_lower_port_xyz(
            self.duct_angled_boundary,
            self.duct_straight_boundary,
            self.params.n_TF.value,
            self.params.lower_port_angle.value,
            self.params.tk_ts.value,
            self.x_straight_end,
        )

        pc = PhysicalComponent(self.name, duct)
        void = PhysicalComponent(self.name + " voidspace", void, material=Void("vacuum"))
        apply_component_display_options(pc, color=BLUE_PALETTE["TS"][0])
        apply_component_display_options(void, color=(0, 0, 0))

        return [pc, void]


@dataclass
class VVLowerPortDuctBuilderParams(ParameterFrame):
    """Vacuum Vessel Lower Port Duct Builder Parameters"""

    n_TF: Parameter[int]
    lower_port_angle: Parameter[float]
    tk_ts: Parameter[float]
    tk_vv_single_wall: Parameter[float]
    g_vv_ts: Parameter[float]


class VVLowerPortDuctBuilder(Builder):
    """
    Vacuum Vessel Lower Port Duct Builder
    """

    param_cls = VVLowerPortDuctBuilderParams

    def __init__(
        self,
        params: Union[ParameterFrame, Dict],
        build_config: Dict,
        duct_angled_nowall_extrude_boundary: BluemiraWire,
        duct_straight_nowall_extrude_boundary: BluemiraWire,
        x_straight_end: float,
    ):
        super().__init__(params, build_config)
        offset_value = -(self.params.tk_ts.value + self.params.g_vv_ts.value)
        self.duct_angled_boundary = offset_wire(
            duct_angled_nowall_extrude_boundary, offset_value
        )
        self.duct_straight_boundary = offset_wire(
            duct_straight_nowall_extrude_boundary, offset_value
        )
        self.duct_straight_boundary.translate((-offset_value, 0, 0))
        self.x_straight_end = x_straight_end

    def build(self) -> Component:
        """
        Build the vacuum vessel lower port.
        """
        return self.component_tree(
            xz=None,
            xy=None,
            xyz=self.build_xyz(),
        )

    def build_xyz(self) -> List[PhysicalComponent]:
        """
        Build the vacuum vessel lower port in x-y-z.
        """
        duct, void = build_lower_port_xyz(
            self.duct_angled_boundary,
            self.duct_straight_boundary,
            self.params.n_TF.value,
            self.params.lower_port_angle.value,
            self.params.tk_vv_single_wall.value,
            self.x_straight_end,
        )

        pc = PhysicalComponent(self.name, duct)
        void = PhysicalComponent(self.name + " voidspace", void, material=Void("vacuum"))
        apply_component_display_options(pc, color=BLUE_PALETTE["VV"][0])
        apply_component_display_options(void, color=(0, 0, 0))

        return [pc, void]


def _face_and_void_from_outer_boundary(
    outer_boundary: BluemiraWire, thickness: float
) -> Tuple[BluemiraFace]:
    inner_boundary = offset_wire(outer_boundary, -thickness)
    return BluemiraFace([outer_boundary, inner_boundary]), BluemiraFace(inner_boundary)


def build_lower_port_xyz(
    duct_angled_boundary: BluemiraWire,
    duct_straight_boundary: BluemiraWire,
    n_TF: int,
    duct_angle: float,
    wall_tk: float,
    x_straight_end: float,
) -> Tuple[BluemiraSolid]:
    """
    Build lower port solid geometry, including void (estimate)

    Parameters
    ----------
    duct_angled_boundary:
        Outer x-y boundary wire of the angled port cross-section
    duct_straight_boundary:
        Outer x-y boundary wire of the straight port cross-section
    n_TF:
        Number of TF coils
    duct_angle:
        Angle of the lower port duct [degrees]
    wall_tk:
        Wall thickness of the lower port
    x_straight_end:
        Radial coordinate of the end point of the straight duct

    Returns
    -------
    duct:
        Solid of the lower port duct
    void:
        Solid of the lower port void (estimate)
    """
    straight_duct_extrude_extent = (
        x_straight_end - duct_straight_boundary.bounding_box.x_min
    )
    if straight_duct_extrude_extent <= 0:
        BuilderError(
            "End radial coordinates of the straight duct is lower than it's start"
            " coordinate."
        )

    duct_angle = np.deg2rad(duct_angle)

    angled_duct_face, angled_void_face = _face_and_void_from_outer_boundary(
        duct_angled_boundary, wall_tk
    )
    straight_duct_face, straight_void_face = _face_and_void_from_outer_boundary(
        duct_straight_boundary, wall_tk
    )
    straight_duct_backwall_face = BluemiraFace(duct_straight_boundary)

    angled_bb = duct_angled_boundary.bounding_box
    strait_bb = duct_straight_boundary.bounding_box
    if duct_angle < -0.25 * np.pi:
        # -2 to make sure it goes through
        angled_duct_extrude_extent = abs(
            (angled_bb.z_max - (strait_bb.z_max - 2.0)) / np.sin(duct_angle)
        )
    else:
        # +2 to make sure it goes through
        angled_duct_extrude_extent = abs(
            (angled_bb.x_min - (strait_bb.x_min + 2.0)) / np.cos(duct_angle)
        )

    ext_vector = angled_duct_extrude_extent * np.array(
        [np.cos(duct_angle), 0, np.sin(duct_angle)]
    )
    angled_duct = extrude_shape(angled_duct_face, ext_vector)
    angled_void = extrude_shape(angled_void_face, ext_vector)

    straight_duct_backwall = extrude_shape(
        straight_duct_backwall_face,
        straight_duct_backwall_face.normal_at() * wall_tk,
    )

    ext_vector = straight_duct_face.normal_at() * -straight_duct_extrude_extent
    straight_duct_length = extrude_shape(straight_duct_face, ext_vector)
    straight_duct_void = extrude_shape(straight_void_face, ext_vector)

    straight_duct = boolean_fuse([straight_duct_backwall, straight_duct_length])

    angled_pieces = boolean_cut(angled_duct, [straight_duct])
    angled_top = sorted(
        angled_pieces, key=lambda s: np.hypot(s.center_of_mass[0], s.center_of_mass[2])
    )[0]
    angled_void_pieces = boolean_cut(angled_void, [straight_duct_void])
    angled_void_piece = sorted(angled_void_pieces, key=lambda s: -s.center_of_mass[2])[0]
    void = boolean_fuse([angled_void_piece, straight_duct_void])

    straight_with_hole = boolean_cut(straight_duct, [angled_top])[0]

    duct = boolean_fuse([angled_top, straight_with_hole])
    duct = boolean_cut(duct, [angled_void_piece])[0]

    # rotate pieces to correct positions
    duct.rotate(degree=180 / n_TF)
    void.rotate(degree=180 / n_TF)
    return duct, void
