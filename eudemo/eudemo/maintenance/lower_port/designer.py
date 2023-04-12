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
EU-DEMO Lower Port
"""
from typing import Dict, Iterable, List, Tuple, Type, Union

import numpy as np

from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import ParameterFrame
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.error import GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import (
    boolean_cut,
    boolean_fuse,
    distance_to,
    make_polygon,
    offset_wire,
)
from bluemira.geometry.wire import BluemiraWire
from eudemo.maintenance.lower_port.parameterisations import LowerPortDesignerParams


class LowerPortDesigner(Designer):
    """
    Lower Port Designer

    Notes
    -----
    Retrictions on the lower_port_angle are between [-90, -0.5]

    """

    param_cls: Type[ParameterFrame] = LowerPortDesignerParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame],
        build_config: Dict,
        divertor_xz: BluemiraFace,
        tf_coil_xz_boundary: BluemiraWire,
    ):
        super().__init__(params, build_config)
        self.divertor_face = divertor_xz
        self.tf_coil_xz_boundary = tf_coil_xz_boundary

        # validate params
        if not -90 <= self.params.lower_duct_angle.value <= -0.5:
            raise ValueError(
                f"{self.params.lower_duct_angle.name}"
                " must be within the range [-90, -0.5] degrees:"
                f" {self.params.lower_duct_angle.value}"
            )
        if self.params.tf_coil_thickness.value <= 0:
            raise ValueError(
                f"{self.params.tf_coil_thickness.name}"
                " must be greater than 0:"
                f" {self.params.tf_coil_thickness.value}"
            )
        if self.params.lower_duct_wall_tk.value <= 0:
            raise ValueError(
                f"{self.params.lower_duct_wall_tk.name}"
                " must be greater than 0:"
                f" {self.params.lower_duct_wall_tk.value}"
            )

        # these are meant to be params, but their names may change,
        # so leaving them here for now
        self.lower_duct_div_padding = 0.2

        self.lower_port_tf_z_offset = 3
        self.lower_port_z_length = 3

        self.lower_duct_hole_x_offset = 1
        self.lower_duct_hole_y_offset = 1

        self.lower_duct_wall_tk = self.params.lower_duct_wall_tk.value
        self.lower_duct_angle = self.params.lower_duct_angle.value

        self.tf_coil_thickness = self.params.tf_coil_thickness.value
        self.n_TF = self.params.n_TF.value

    def run(self) -> Tuple[BluemiraFace, BluemiraWire]:
        """Run method of Designer"""
        div_z_top = self.divertor_face.bounding_box.z_max
        div_z_bot = self.divertor_face.bounding_box.z_min

        div_x_inner = self.divertor_face.bounding_box.x_min
        div_x_outer = self.divertor_face.bounding_box.x_max

        if abs(div_z_top - div_z_bot) > self.lower_port_z_length:
            raise GeometryError(
                "Divertor z-length is greater than the z-length of the lower port: "
                f"{div_z_top - div_z_bot} > {self.lower_port_z_length}"
            )

        (
            padded_top_outer_point,
            padded_bottom_inner_point,
        ) = self._pad_angled_duct_points(
            xz_top_outer_point=(div_x_outer, div_z_top),
            xz_bottom_inner_point=(div_x_inner, div_z_bot),
        )

        (
            duct_w_wall_xz_boundary,
            straight_duct_start_top,
            straight_duct_start_bot,
        ) = self._duct_w_wall_xz_boundary(
            top_outer_point=padded_top_outer_point,
            bottom_inner_point=padded_bottom_inner_point,
        )
        duct_w_wall_xz_koz = BluemiraFace(duct_w_wall_xz_boundary)

        outer_point, inner_point = self._get_wall_points(
            duct_w_wall_xz_koz,
            padded_top_outer_point,
            padded_bottom_inner_point,
        )

        angled_duct_extrude_boundary, angled_y_len = self._angled_duct_extrude_boundary(
            outer_point,
            inner_point,
        )

        straight_duct_extrude_boundary = self._straight_duct_extrude_boundary(
            straight_duct_start_top,
            straight_duct_start_bot,
            angled_y_len,
        )

        return (
            duct_w_wall_xz_koz,
            angled_duct_extrude_boundary,
            straight_duct_extrude_boundary,
        )

    def _pad_angled_duct_points(
        self,
        xz_top_outer_point: Tuple,
        xz_bottom_inner_point: Tuple,
    ):
        points_grad = (xz_top_outer_point[1] - xz_bottom_inner_point[1]) / (
            xz_top_outer_point[0] - xz_bottom_inner_point[0]
        )  # z/x
        points_len, _ = distance_to(
            [xz_top_outer_point[0], 0, xz_top_outer_point[1]],
            [xz_bottom_inner_point[0], 0, xz_bottom_inner_point[1]],
        )

        (
            _,
            padded_top_outer_point,
        ) = LowerPortDesigner._xz_points_dist_away_from(
            xz_bottom_inner_point,
            points_grad,
            points_len + self.lower_duct_div_padding / 2,
        )
        (padded_bot_inner_point, _) = LowerPortDesigner._xz_points_dist_away_from(
            xz_top_outer_point,
            points_grad,
            points_len + self.lower_duct_div_padding / 2,
        )
        return padded_top_outer_point, padded_bot_inner_point

    def _get_wall_points(
        self,
        duct_w_wall_koz: BluemiraFace,
        padded_top_outer_point: Tuple,
        padded_bottom_inner_point: Tuple,
    ):
        wall_top_outer_point = duct_w_wall_koz.vertexes.points[
            duct_w_wall_koz.vertexes.argmin(
                [
                    padded_top_outer_point[0],
                    0,
                    padded_top_outer_point[1],
                ]
            )
        ]
        wall_bot_inner_point = duct_w_wall_koz.vertexes.points[
            duct_w_wall_koz.vertexes.argmin(
                [
                    padded_bottom_inner_point[0],
                    0,
                    padded_bottom_inner_point[1],
                ]
            )
        ]

        return wall_top_outer_point, wall_bot_inner_point

    def _straight_duct_extrude_boundary(
        self,
        start_top_point: Tuple,
        start_bot_point: Tuple,
        angled_y_len: float,
    ) -> BluemiraFace:
        x_point = start_top_point[0]
        y_size = angled_y_len + self.lower_duct_hole_y_offset

        return make_polygon(
            {
                "x": np.array(
                    [
                        x_point,
                        x_point,
                        x_point,
                        x_point,
                    ]
                ),
                "y": np.array(
                    [
                        y_size,
                        -y_size,
                        -y_size,
                        y_size,
                    ]
                ),
                "z": np.array(
                    [
                        start_top_point[1],
                        start_top_point[1],
                        start_bot_point[1],
                        start_bot_point[1],
                    ]
                ),
            },
            closed=True,
        )

    def _angled_duct_extrude_boundary(
        self,
        outer_point: Iterable,
        inner_point: Iterable,
    ) -> BluemiraFace:
        # outer_point and inner_point are np arrays of length 3: x,y,z

        def _calc_y_point(x_point):
            half_beta = np.pi / self.n_TF

            x_meet = self.tf_coil_thickness / np.sin(half_beta)
            x_len = x_point - x_meet

            if x_len < 0:
                raise GeometryError(
                    "LowerPortDesigner: tf_coil_thickness is too large for the"
                    f" space between TF coils at x={x_point}."
                )

            y_at_x_proj = x_len * np.tan(half_beta)
            return y_at_x_proj

        outer_y_point = _calc_y_point(outer_point[0])
        inner_y_point = _calc_y_point(inner_point[0])

        return (
            make_polygon(
                {
                    "x": np.array(
                        [
                            outer_point[0],
                            outer_point[0],
                            inner_point[0],
                            inner_point[0],
                        ]
                    ),
                    "y": np.array(
                        [
                            outer_y_point,
                            -outer_y_point,
                            -inner_y_point,
                            inner_y_point,
                        ]
                    ),
                    "z": np.array(
                        [
                            outer_point[2],
                            outer_point[2],
                            inner_point[2],
                            inner_point[2],
                        ]
                    ),
                },
                closed=True,
            ),
            outer_y_point,
        )

    def _duct_w_wall_xz_boundary(
        self,
        top_outer_point: Tuple,
        bottom_inner_point: Tuple,
    ):
        angled_duct_boundary = self._angled_duct_boundary(
            top_outer_point,
            bottom_inner_point,
        )
        angled_duct = BluemiraFace(
            angled_duct_boundary,
        )

        straight_duct_z_top = (
            self.tf_coil_xz_boundary.bounding_box.z_min
            - self.lower_port_tf_z_offset
            # the wall thickness is adds to the z, so subtract it here
            # to make the actual top of the duct at the correct z
            # (i.e. self.tf_coil_xz_boundary.bounding_box.z_min
            # - self.lower_duct_tf_z_offset)
            - self.lower_duct_wall_tk
        )

        straight_duct_boundary, bottom_left_point = self._straight_duct_boundary(
            angled_duct_boundary,
            z_top_point=straight_duct_z_top,
        )
        straight_duct = BluemiraFace(
            straight_duct_boundary,
        )

        angled_cuts = boolean_cut(
            angled_duct,
            [straight_duct],
        )
        top_piece = angled_cuts[0]
        duct_inner = boolean_fuse([top_piece, straight_duct])

        nowall_in_x_straight_duct_start_top = (
            bottom_left_point[0],
            straight_duct_z_top + self.lower_duct_wall_tk,
        )
        nowall_in_x_straight_duct_start_bot = (
            bottom_left_point[0],
            bottom_left_point[1] - self.lower_duct_wall_tk,
        )

        return (
            offset_wire(
                duct_inner.wires[0],
                self.lower_duct_wall_tk,
            ),
            nowall_in_x_straight_duct_start_top,
            nowall_in_x_straight_duct_start_bot,
        )

    def _angled_duct_boundary(
        self,
        top_outer_point: Tuple,
        bottom_inner_point: Tuple,
    ):
        r_search = 50  # must just be large

        angled_duct_grad = np.tan(np.deg2rad(self.lower_duct_angle))  # z/x

        # get "search" points to construct a really long angle duct
        # that will intersect the straight duct
        (
            _,
            to_intc_search_point,
        ) = LowerPortDesigner._xz_points_dist_away_from(
            top_outer_point,
            angled_duct_grad,
            r_search,
        )
        (
            _,
            bi_intc_search_point,
        ) = LowerPortDesigner._xz_points_dist_away_from(
            bottom_inner_point,
            angled_duct_grad,
            r_search,
        )

        return make_polygon(
            {
                "x": np.array(
                    [
                        top_outer_point[0],
                        to_intc_search_point[0],
                        bi_intc_search_point[0],
                        bottom_inner_point[0],
                    ]
                ),
                "y": 0,
                "z": np.array(
                    [
                        top_outer_point[1],
                        to_intc_search_point[1],
                        bi_intc_search_point[1],
                        bottom_inner_point[1],
                    ]
                ),
            },
            closed=True,
        )

    def _straight_duct_boundary(
        self,
        angled_duct_boundary: BluemiraWire,
        z_top_point: float,
    ):
        x_duct_extent = 10  # must extend past the outer rad. shield

        # find the other point along the horizontal
        hori_search_wire = LowerPortDesigner._make_xz_wire_from_points(
            (0, z_top_point),
            (x_duct_extent, z_top_point),
        )
        intc_points = LowerPortDesigner._intersection_points(
            angled_duct_boundary,
            hori_search_wire,
        )
        if not intc_points:
            raise GeometryError(
                "There was no intersection "
                "between the straight and angled ducts. "
                f"Making {self.params.lower_duct_angle.name} more negative ",
                "or raising the z-point of the straight duct may help.",
            )

        # left-most
        intc_point = min(intc_points, key=lambda p: p[0])
        intc_point_inset = (
            intc_point[0] - self.lower_duct_hole_x_offset,
            intc_point[2],
        )

        x_duct_start = intc_point_inset[0]
        z_duct_top = intc_point_inset[1]
        z_duct_bottom = intc_point_inset[1] - self.lower_port_z_length

        return (
            make_polygon(
                {
                    "x": np.array(
                        [
                            x_duct_start,
                            x_duct_extent,
                            x_duct_extent,
                            x_duct_start,
                        ]
                    ),
                    "y": 0,
                    "z": np.array(
                        [
                            z_duct_top,
                            z_duct_top,
                            z_duct_bottom,
                            z_duct_bottom,
                        ]
                    ),
                },
                closed=True,
            ),
            # bottom left point
            (x_duct_start, z_duct_bottom),
        )

    @staticmethod
    def _xz_points_dist_away_from(
        starting_xz_point: Union[Tuple, List],
        gradient: float,
        distance: float,
    ) -> Tuple:
        s_x = starting_xz_point[0]
        s_z = starting_xz_point[1]
        sqrt_value = np.sqrt(distance**2 / (1 + gradient**2))
        f_x_pve = sqrt_value + s_x
        f_z_pve = gradient * sqrt_value + s_z
        f_x_nve = -sqrt_value + s_x
        f_z_nve = gradient * -sqrt_value + s_z
        return (
            [f_x_nve, f_z_nve],
            [f_x_pve, f_z_pve],
        )

    @staticmethod
    def _make_xz_wire_from_points(
        a_xz_point: Union[Tuple, List],
        b_xz_point: Union[Tuple, List],
    ) -> BluemiraWire:
        return make_polygon(
            {
                "x": np.array([a_xz_point[0], b_xz_point[0]]),
                "y": 0,
                "z": np.array([a_xz_point[1], b_xz_point[1]]),
            },
        )

    @staticmethod
    def _intersection_points(
        wire_a: BluemiraGeo,
        wire_b: BluemiraGeo,
    ) -> List[Tuple]:
        dist, vects, _ = wire_a.shape.distToShape(wire_b.shape)
        if dist > 0.0001:  # not intersecting
            return []
        pois = []
        for vect_pair in vects:
            v = vect_pair[0]
            pois.append((v.x, v.y, v.z))
        return pois
