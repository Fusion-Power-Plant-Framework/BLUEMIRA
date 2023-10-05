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
Creating ducts for the port
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Union

if TYPE_CHECKING:
    from bluemira.base.reactor_config import ConfigParams
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
from bluemira.geometry.tools import (
    boolean_fragments,
    boolean_fuse,
    extrude_shape,
    make_polygon,
    offset_wire,
    point_inside_shape,
)
from bluemira.materials import Void


@dataclass
class TSUpperPortDuctBuilderParams(ParameterFrame):
    """Thermal shield upper port duct builder Parameter Frame"""

    n_TF: Parameter[int]
    tf_wp_depth: Parameter[float]
    g_ts_tf: Parameter[float]
    tk_ts: Parameter[float]
    g_cr_ts: Parameter[float]


class TSUpperPortDuctBuilder(Builder):
    """Thermal shield upper port duct builder"""

    params: TSUpperPortDuctBuilderParams
    param_cls = TSUpperPortDuctBuilderParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame, ConfigParams, None],
        port_koz: BluemiraFace,
        cryostat_ts_xz: BluemiraWire,
    ):
        super().__init__(params, None)
        self.x_min = port_koz.bounding_box.x_min
        self.x_max = port_koz.bounding_box.x_max
        self.z_max = cryostat_ts_xz.bounding_box.z_max + 0.5 * self.params.g_cr_ts.value

        if self.params.tk_ts.value <= 0:
            raise ValueError("Port wall thickness must be > 0")

        self.y_offset = self.params.tf_wp_depth.value + self.params.g_ts_tf.value

    def build(self) -> Component:
        """Build upper port"""
        xy_face = make_upper_port_xy_face(
            self.params.n_TF.value,
            self.x_min,
            self.x_max,
            self.params.tk_ts.value,
            self.params.tk_ts.value,
            self.y_offset,
        )

        return self.component_tree(
            None, [self.build_xy(xy_face)], self.build_xyz(xy_face)
        )

    def build_xyz(self, xy_face: BluemiraFace) -> PhysicalComponent:
        """Build upper port xyz"""
        xy_voidface = BluemiraFace(xy_face.boundary[1])
        xy_outface = BluemiraFace(xy_face.boundary[0])
        port = extrude_shape(xy_face, (0, 0, self.z_max))
        # Add start-cap for future boolean fragmentation help
        cap = extrude_shape(xy_outface, vec=(0, 0, 0.1))
        port = boolean_fuse([port, cap])
        comp = PhysicalComponent(self.name, port)
        apply_component_display_options(comp, BLUE_PALETTE["TS"][0])
        void = PhysicalComponent(
            self.name + " voidspace",
            extrude_shape(xy_voidface, (0, 0, self.z_max)),
            material=Void("vacuum"),
        )
        apply_component_display_options(void, color=(0, 0, 0))
        return [comp, void]

    def build_xy(self, face: BluemiraFace) -> PhysicalComponent:
        """Build upport port xy face"""
        comp = PhysicalComponent(self.name, face)
        apply_component_display_options(comp, BLUE_PALETTE["TS"][0])
        return comp


@dataclass
class TSEquatorialPortDuctBuilderParams(ParameterFrame):
    """Thermal shield upper port duct builder Parameter Frame"""

    n_TF: Parameter[int]
    R_0: Parameter[float]
    tf_wp_depth: Parameter[float]
    g_ts_tf: Parameter[float]
    g_vv_ts: Parameter[float]
    tk_ts: Parameter[float]
    g_cr_ts: Parameter[float]
    tk_vv_single_wall: Parameter[float]
    ep_width: Parameter[float]
    ep_height: Parameter[float]
    ep_z_position: Parameter[float]


class TSEquatorialPortDuctBuilder(Builder):
    """Thermal shield upper port duct builder"""

    params: TSEquatorialPortDuctBuilderParams
    param_cls = TSEquatorialPortDuctBuilderParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame, ConfigParams, None],
        cryostat_xz: BluemiraWire,
    ):
        super().__init__(params, None)
        # Put the end of the equatorial port half-way between cryostat ts and
        # cryostat
        self.x_max = cryostat_xz.bounding_box.x_max + 0.5 * self.params.g_cr_ts.value

    def build(self) -> Component:
        """Build equatorial port"""
        offset = (
            self.params.tk_vv_single_wall.value
            + self.params.g_vv_ts.value
            + self.params.tk_ts.value
        )
        y_val = 0.5 * self.params.ep_width.value + offset
        z_ref = self.params.ep_z_position.value
        z_val = 0.5 * self.params.ep_height.value + offset
        yz_face = make_equatorial_port_yz_face(
            self.x_max,
            -y_val,
            y_val,
            z_ref - z_val,
            z_ref + z_val,
            self.params.tk_ts.value,
        )

        return self.component_tree(None, None, self.build_xyz(yz_face))

    def build_xyz(self, yz_face: BluemiraFace) -> PhysicalComponent:
        """Build equatorial port xyz"""
        yz_voidface = BluemiraFace(yz_face.boundary[1])
        degree = 180 / self.params.n_TF.value
        vec = (self.params.R_0.value - self.x_max, 0, 0)
        port = extrude_shape(yz_face, vec)
        port.rotate(degree=degree)
        comp = PhysicalComponent(self.name, port)

        void = extrude_shape(yz_voidface, vec)
        void.rotate(degree=degree)
        void = PhysicalComponent(self.name + " voidspace", void, material=Void("vacuum"))

        apply_component_display_options(comp, BLUE_PALETTE["VV"][0])
        apply_component_display_options(void, color=(0, 0, 0))
        return [comp, void]


@dataclass
class VVUpperPortDuctBuilderParams(ParameterFrame):
    """Vacuum vessel upper port duct builder Parameter Frame"""

    n_TF: Parameter[int]
    tf_wp_depth: Parameter[float]
    g_ts_tf: Parameter[float]
    tk_ts: Parameter[float]
    g_vv_ts: Parameter[float]
    g_cr_ts: Parameter[float]
    tk_vv_double_wall: Parameter[float]
    tk_vv_single_wall: Parameter[float]


class VVUpperPortDuctBuilder(Builder):
    """Vacuum vessel upper port duct builder"""

    params: VVUpperPortDuctBuilderParams
    param_cls = VVUpperPortDuctBuilderParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame, ConfigParams, None],
        port_koz: BluemiraFace,
        cryostat_ts_xz: BluemiraWire,
    ):
        super().__init__(params, None)
        koz_offset = self.params.tk_ts.value + self.params.g_vv_ts.value
        self.x_min = port_koz.bounding_box.x_min + koz_offset
        self.x_max = port_koz.bounding_box.x_max - koz_offset
        self.z_max = cryostat_ts_xz.bounding_box.z_max + 0.5 * self.params.g_cr_ts.value

        if (
            self.params.tk_vv_double_wall.value <= 0
            or self.params.tk_vv_single_wall.value <= 0
        ):
            raise ValueError("Port wall thickness must be > 0")

        self.y_offset = (
            self.params.tf_wp_depth.value
            + self.params.g_ts_tf.value
            + self.params.tk_ts.value
            + self.params.g_vv_ts.value
        )

    def build(self) -> Component:
        """Build upper port"""
        xy_face = make_upper_port_xy_face(
            self.params.n_TF.value,
            self.x_min,
            self.x_max,
            self.params.tk_vv_double_wall.value,
            self.params.tk_vv_single_wall.value,
            self.y_offset,
        )

        return self.component_tree(
            None,
            self.build_xy(xy_face),
            self.build_xyz(xy_face),
        )

    def build_xyz(self, xy_face: BluemiraFace) -> PhysicalComponent:
        """Build upper port xyz"""
        xy_voidface = BluemiraFace(xy_face.boundary[1])
        xy_outface = BluemiraFace(xy_face.boundary[0])
        port = extrude_shape(xy_face, (0, 0, self.z_max))
        # Add start-cap for future boolean fragmentation help
        cap = extrude_shape(xy_outface, vec=(0, 0, 0.1))
        port = boolean_fuse([port, cap])

        comp = PhysicalComponent(self.name, port)
        apply_component_display_options(comp, BLUE_PALETTE["VV"][0])
        void = PhysicalComponent(
            self.name + " voidspace",
            extrude_shape(xy_voidface, (0, 0, self.z_max)),
            material=Void("vacuum"),
        )
        apply_component_display_options(void, color=(0, 0, 0))
        return [comp, void]

    def build_xy(self, xy_face: BluemiraFace) -> PhysicalComponent:
        """Build upper port xy face"""
        xy_voidface = BluemiraFace(xy_face.boundary[1])
        comp = PhysicalComponent(self.name, xy_face)
        apply_component_display_options(comp, BLUE_PALETTE["VV"][0])
        void = PhysicalComponent(
            self.name + " voidspace", xy_voidface, material=Void("vacuum")
        )
        apply_component_display_options(void, color=(0, 0, 0))
        return [comp, void]


@dataclass
class VVEquatorialPortDuctBuilderParams(ParameterFrame):
    """Vacuum vessel equatorial port duct builder Parameter Frame"""

    R_0: Parameter[float]
    n_TF: Parameter[int]
    g_cr_ts: Parameter[float]
    ep_z_position: Parameter[float]
    ep_width: Parameter[float]
    ep_height: Parameter[float]
    tk_vv_single_wall: Parameter[float]


class VVEquatorialPortDuctBuilder(Builder):
    """Vacuum vessel upper port duct builder"""

    params: VVEquatorialPortDuctBuilderParams
    param_cls = VVEquatorialPortDuctBuilderParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame, ConfigParams, None],
        cryostat_xz: BluemiraWire,
    ):
        super().__init__(params, None)
        # Put the end of the equatorial port half-way between cryostat ts and
        # cryostat
        self.x_max = cryostat_xz.bounding_box.x_max + 0.5 * self.params.g_cr_ts.value

    def build(self) -> Component:
        """Build equatorial port"""
        y_val = 0.5 * self.params.ep_width.value
        z_ref = self.params.ep_z_position.value
        z_val = 0.5 * self.params.ep_height.value
        yz_face = make_equatorial_port_yz_face(
            self.x_max,
            -y_val,
            y_val,
            z_ref - z_val,
            z_ref + z_val,
            self.params.tk_vv_single_wall.value,
        )

        return self.component_tree(
            None,
            None,
            self.build_xyz(yz_face),
        )

    def build_xyz(self, yz_face: BluemiraFace) -> PhysicalComponent:
        """Build equatorial port xyz"""
        yz_voidface = BluemiraFace(yz_face.boundary[1])
        degree = 180 / self.params.n_TF.value
        vec = (self.params.R_0.value - self.x_max, 0, 0)
        port = extrude_shape(yz_face, vec)
        port.rotate(degree=degree)
        comp = PhysicalComponent(self.name, port)

        void = extrude_shape(yz_voidface, vec)
        void.rotate(degree=degree)
        void = PhysicalComponent(self.name + " voidspace", void, material=Void("vacuum"))

        apply_component_display_options(comp, BLUE_PALETTE["VV"][0])
        apply_component_display_options(void, color=(0, 0, 0))
        return [comp, void]


def make_upper_port_xy_face(
    n_TF: int,
    x_min: float,
    x_max: float,
    wall_end_tk: float,
    wall_side_tk: float,
    y_offset: float,
) -> BluemiraFace:
    """
    Creates a xy cross section of the upper port

    translates the port koz to the origin,
    builds the port at the origin and moves it back

    Parameters
    ----------
    n_TF:
        Number of TF coils
    x_min:
        Inner radius of the port keep-out zone
    x_max:
        Outer radius of the port keep-out zone
    wall_end_tk:
        Port wall end thickness
    wall_size_tk:
        Port wall side thickness
    y_offset:
        Offset value from the x-z plane at which to start building the port
        (excluding port side wall thickness)

    Returns
    -------
    xy_face:
        x-y face of the upper port

    Notes
    -----
    the port koz is slightly trimmed to allow for square ends to the port
    """
    half_beta = np.pi / n_TF
    cos_hb = np.cos(half_beta)
    tan_hb = np.tan(half_beta)

    y_tf_out = y_offset / cos_hb
    y_tf_in = y_tf_out + wall_side_tk / cos_hb

    x1 = x_min

    # This is the correct way to retrieve the outer radius of the port, accounting
    # for the TF thickness
    # It's just the intersection of a line and a circle in the positive quadrant.
    a1 = 1 + tan_hb**2
    b1 = -2 * y_tf_out * tan_hb
    c1 = y_tf_out**2 - x_max**2
    discriminant = b1**2 - 4 * a1 * c1
    x4 = 0.5 * (-b1 + np.sqrt(discriminant)) / a1

    x2, x3 = x1 + wall_end_tk, x4 - wall_end_tk

    if x2 >= x3:
        raise BuilderError("Port dimensions too small")

    y1 = x1 * tan_hb - y_tf_out

    if y1 < 0:
        # Triangular outer port wall
        y1 = 0
        x1 = y_tf_out / tan_hb
        x2 = x1 + wall_end_tk

    y2, y3 = x2 * tan_hb - y_tf_in, x3 * tan_hb - y_tf_in

    if y3 <= 0:
        raise BuilderError("Port dimensions too small")

    if y2 < 0:
        # Triangular inner port wall
        y2 = 0
        c = y3 - tan_hb * x3
        x2 = -c / tan_hb
        x1 = x2 - wall_end_tk

    y1, y4 = x1 * tan_hb - y_tf_out, x4 * tan_hb - y_tf_out

    inner_wire = make_polygon(
        {"x": [x2, x3, x3, x2], "y": [-y2, -y3, y3, y2]}, closed=True
    )
    outer_wire = make_polygon(
        {"x": [x1, x4, x4, x1], "y": [-y1, -y4, y4, y1]}, closed=True
    )

    xy_face = BluemiraFace((outer_wire, inner_wire))
    xy_face.rotate(degree=np.rad2deg(half_beta))

    return xy_face


def make_equatorial_port_yz_face(
    x_ref: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    wall_side_tk: float,
) -> BluemiraFace:
    """
    Creates a yz cross section of the equatorial port

    builds the port at the origin

    Parameters
    ----------
    x_ref:
        Reference x coordinate of the y-z plane
    y_min:
        Minimum y coordinate of the port void
    y_max:
        Maximum y coordinate of the port void
    z_min:
        Minimum z coordinate of the port void
    z_max:
        Maximum z coordinate of the port void
    wall_side_tk:
        Thickness of the port walss

    Returns
    -------
    yz_face:
        y-z face of the equatorial port
    """
    y = np.array([y_min, y_min, y_max, y_max])
    z = np.array([z_min, z_max, z_max, z_min])
    inner = make_polygon({"x": x_ref, "y": y, "z": z}, closed=True)
    outer = offset_wire(inner, wall_side_tk, open_wire=False)
    return BluemiraFace([outer, inner])


def pipe_pipe_join(
    target_shape: BluemiraSolid,
    target_void: BluemiraSolid,
    tool_shape: BluemiraSolid,
    tool_void: BluemiraSolid,
) -> List[BluemiraSolid]:
    """
    Join two hollow, intersecting pipes.

    Parameters
    ----------
    target_shape:
        Solid of the target shape
    target_void:
        Solid of the target void
    tool_shape:
        Solid of the tool shape
    tool_void:
        Solid of the tool void

    Returns
    -------
    shape:
        Solid of the joined pipe-pipe shape
    void:
        Solid of the joined pipe-pipe void

    Notes
    -----
    This approach is more brittle than a classic fuse, fuse, cut operation, but is
    substantially faster. If the parts do not fully intersect, undesired results
    are to be expected.
    """
    _, (target_fragments, tool_fragments) = boolean_fragments([target_shape, tool_shape])

    # Keep the largest piece of the target by volume (opinionated)
    # This is in case its COG is inside the tool void
    target_fragments.sort(key=lambda solid: -solid.volume)
    new_shape_pieces = [target_fragments[0]]
    target_fragments = target_fragments[1:]
    for targ_frag in target_fragments:
        # Find the target piece(s) that are inside the tool
        com = targ_frag.center_of_mass
        if not point_inside_shape(com, tool_void):
            new_shape_pieces.append(targ_frag)

    for tool_frag in tool_fragments:
        # Find the tool piece(s) that are inside the target
        com = tool_frag.center_of_mass
        if not point_inside_shape(com, target_void):
            new_shape_pieces.append(tool_frag)
        else:
            # Find the union piece(s)
            new_shape_pieces.extend(
                [
                    tool_frag
                    for targ_frag in target_fragments
                    if tool_frag.is_same(targ_frag)
                ]
            )

    return new_shape_pieces
