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
Coil support builders
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.designer import Designer
from bluemira.base.error import BuilderError
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.builders.tools import apply_component_display_options
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.compound import BluemiraCompound
from bluemira.geometry.constants import VERY_BIG
from bluemira.geometry.coordinates import Coordinates, get_intersect
from bluemira.geometry.error import GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import (
    boolean_cut,
    boolean_fuse,
    extrude_shape,
    make_polygon,
    mirror_shape,
    offset_wire,
    signed_distance_2D_polygon,
    slice_shape,
    sweep_shape,
)
from bluemira.geometry.wire import BluemiraWire
from bluemira.optimisation import OptimisationProblem
from bluemira.optimisation.typing import ConstraintT
from bluemira.utilities.optimiser import Optimiser as _DeprecatedOptimiser


@dataclass
class ITERGravitySupportBuilderParams(ParameterFrame):
    """
    ITER-like gravity support parameters
    """

    x_g_support: Parameter[float]
    z_gs: Parameter[float]
    tf_wp_width: Parameter[float]
    tf_wp_depth: Parameter[float]
    tk_tf_side: Parameter[float]
    tf_gs_tk_plate: Parameter[float]
    tf_gs_g_plate: Parameter[float]
    tf_gs_base_depth: Parameter[float]


class ITERGravitySupportBuilder(Builder):
    """
    ITER-like gravity support builder

    Parameters
    ----------
    params:
        Parameters to use
    build_config:
        Build config to use
    tf_kz_keep_out_zone:
        TF coil wire keep-out-zone for the outer edge of the TF coil (including casing)
        Note that this should be on the y=0 plane.
    """

    param_cls: Type[ITERGravitySupportBuilderParams] = ITERGravitySupportBuilderParams

    def __init__(
        self,
        params: Union[ITERGravitySupportBuilderParams, Dict],
        build_config: Dict,
        tf_xz_keep_out_zone: BluemiraWire,
    ):
        super().__init__(params, build_config)
        self.tf_xz_keep_out_zone = tf_xz_keep_out_zone

    def build(self) -> Component:
        """
        Build the ITER-like gravity support component.
        """
        xyz = self.build_xyz()
        return self.component_tree([self.build_xz(xyz)], self.build_xy(), [xyz])

    @staticmethod
    def build_xz(xyz_component):
        """
        Build the x-z component of the ITER-like gravity support.
        """
        xz_plane = BluemiraPlane((0, 0, 0), (0, 1, 0))
        slice_result = slice_shape(xyz_component.shape, xz_plane)

        # Process UGLY SLICE
        wires = sorted(slice_result, key=lambda wire: wire.length)
        wire_list = [wires.pop()]
        wire_list.extend(wires)
        shape = BluemiraFace(wire_list)

        component = PhysicalComponent("ITER-like gravity support", shape)
        apply_component_display_options(component, color=BLUE_PALETTE["TF"][2])
        return component

    def build_xy(self):
        """
        Build the x-y component of the ITER-like gravity support.
        """

    def _get_intersection_wire(self, width):
        x_g_support = self.params.x_g_support.value
        x_inner_line = x_g_support - 0.5 * width
        x_outer_line = x_g_support + 0.5 * width
        z_min = self.tf_xz_keep_out_zone.bounding_box.z_min
        z_max = z_min + 0.5 * (self.tf_xz_keep_out_zone.bounding_box.z_max - z_min)
        x_min = self.tf_xz_keep_out_zone.bounding_box.x_min + 0.5 * width
        x_max = self.tf_xz_keep_out_zone.bounding_box.x_max - 0.5 * width
        if (x_g_support < x_min) | (x_g_support > x_max):
            raise BuilderError(
                "The gravity support footprint is not contained within the provided TF"
                " coil geometry!"
            )

        if (self.params.z_gs.value - 6 * self.params.tf_gs_tk_plate.value) > z_min:
            raise BuilderError(
                "The gravity support floor is not lower than where the TF coil is!"
            )
        z_min = self.params.z_gs.value - 6 * self.params.tf_gs_tk_plate.value

        cut_box = make_polygon(
            {
                "x": [x_inner_line, x_inner_line, x_outer_line, x_outer_line],
                "y": 0,
                "z": [z_min, z_max, z_max, z_min],
            },
            closed=True,
        )

        cut_result = boolean_cut(self.tf_xz_keep_out_zone, cut_box)

        if cut_result is None:
            raise BuilderError(
                "Boolean cutting returned nothing... check your geometry please."
            )

        return sorted(cut_result, key=lambda wire: wire.length)[0]

    def _make_connection_block(self, width, v1, v4, intersection_wire):
        """
        Make the connection block of the gravity support with the TF coil
        """
        z_block_lower = min(v1.z[0], v4.z[0]) - 5 * self.params.tf_gs_tk_plate.value
        v2 = Coordinates(np.array([v1.x[0], 0, z_block_lower]))
        v3 = Coordinates(np.array([v4.x[0], 0, z_block_lower]))

        points = np.concatenate([v1.xyz, v2.xyz, v3.xyz, v4.xyz], axis=1)
        closing_wire = make_polygon(points, closed=False)
        face = BluemiraFace(BluemiraWire([intersection_wire, closing_wire]))

        # Then extrude that face in both directions to get the connection block
        face.translate(vector=(0, -0.5 * width, 0))
        return extrude_shape(face, vec=(0, width, 0))

    def _make_plates(self, width, v1x, v4x, z_block_lower):
        """
        Make the gravity support vertical plates
        """
        plate_list = []
        yz_profile = Coordinates(
            {
                "x": 4 * [v1x],
                "y": [
                    -0.5 * width,
                    0.5 * width,
                    0.5 * self.params.tf_gs_base_depth.value,
                    -0.5 * self.params.tf_gs_base_depth.value,
                ],
                "z": [
                    z_block_lower,
                    z_block_lower,
                    self.params.z_gs.value,
                    self.params.z_gs.value,
                ],
            },
        )
        yz_profile = make_polygon(yz_profile, closed=True)

        plating_width = v4x - v1x
        plate_and_gap = (
            self.params.tf_gs_g_plate.value + self.params.tf_gs_tk_plate.value
        )
        n_plates = int((plating_width + self.params.tf_gs_g_plate.value) / plate_and_gap)
        total_width = (
            n_plates * self.params.tf_gs_tk_plate.value
            + (n_plates - 1) * self.params.tf_gs_g_plate.value
        )
        delta_width = plating_width - total_width
        yz_profile.translate(vector=(0.5 * delta_width, 0, 0))

        plate = extrude_shape(
            BluemiraFace(yz_profile), vec=(self.params.tf_gs_tk_plate.value, 0, 0)
        )
        plate_list.append(plate)
        for _ in range(n_plates - 1):
            plate = plate.deepcopy()
            plate.translate(
                vector=(
                    plate_and_gap,
                    0,
                    0,
                )
            )
            plate_list.append(plate)
        return plate_list

    def _make_floor_block(self, v1x, v4x):
        xz_profile = Coordinates(
            {
                "x": [v1x, v1x, v4x, v4x],
                "y": [
                    -0.5 * self.params.tf_gs_base_depth.value,
                    0.5 * self.params.tf_gs_base_depth.value,
                    0.5 * self.params.tf_gs_base_depth.value,
                    -0.5 * self.params.tf_gs_base_depth.value,
                ],
                "z": 4 * [self.params.z_gs.value],
            },
        )
        xz_profile = BluemiraFace(make_polygon(xz_profile, closed=True))
        return extrude_shape(
            xz_profile, vec=(0, 0, -5 * self.params.tf_gs_tk_plate.value)
        )

    def build_xyz(
        self,
    ) -> PhysicalComponent:
        """
        Build the x-y-z component of the ITER-like gravity support.
        """
        shape_list = []
        # First, project upwards at the radius of the GS into the keep-out-zone
        # and get a x-z face of the boolean difference.

        # Get the square width
        width = self.params.tf_wp_depth.value + 2 * self.params.tk_tf_side.value

        intersection_wire = self._get_intersection_wire(width)
        v1 = intersection_wire.start_point()
        v4 = intersection_wire.end_point()
        if v1.x > v4.x:
            v1, v4 = v4, v1

        connection_block = self._make_connection_block(width, v1, v4, intersection_wire)
        shape_list.append(connection_block)

        # Next, make the plates in a linear pattern, in y-z, along x
        z_block_lower = connection_block.bounding_box.z_min
        shape_list.extend(
            self._make_plates(width, float(v1.x), float(v4.x), z_block_lower)
        )

        # Finally, make the floor block
        shape_list.append(self._make_floor_block(float(v1.x), float(v4.x)))
        shape = boolean_fuse(shape_list)
        component = PhysicalComponent("ITER-like gravity support", shape)
        apply_component_display_options(component, color=BLUE_PALETTE["TF"][2])
        return component


@dataclass
class PFCoilSupportBuilderParams(ParameterFrame):
    """
    PF coil support parameters
    """

    tf_wp_width: Parameter[float]
    tf_wp_depth: Parameter[float]
    tk_tf_side: Parameter[float]
    pf_s_tk_plate: Parameter[float]
    pf_s_n_plate: Parameter[int]
    pf_s_g: Parameter[float]


class PFCoilSupportBuilder(Builder):
    """
    PF coil support builder
    """

    param_cls: Type[PFCoilSupportBuilderParams] = PFCoilSupportBuilderParams

    def __init__(
        self,
        params: Union[PFCoilSupportBuilderParams, Dict],
        build_config: Dict,
        tf_xz_keep_out_zone: BluemiraWire,
        pf_coil_xz: BluemiraWire,
    ):
        super().__init__(params, build_config, verbose=False)
        self.tf_xz_keep_out_zone = tf_xz_keep_out_zone
        self.pf_coil_xz = pf_coil_xz
        self.name = f"{self.name} {self.build_config.get('support_number', 0)}"

    def build(self) -> Component:
        """
        Build the PF coil support component.
        """
        xyz = self.build_xyz()
        return self.component_tree([self.build_xz(xyz)], self.build_xy(), [xyz])

    def build_xy(self):
        """
        Build the x-y components of the PF coil support.
        """

    def build_xz(self, xyz):
        """
        Build the x-z components of the PF coil support.
        """
        result = slice_shape(xyz.shape, BluemiraPlane(axis=(0, 1, 0)))
        result.sort(key=lambda wire: -wire.length)
        face = BluemiraFace(result)
        component = PhysicalComponent(self.name, face)
        apply_component_display_options(component, color=BLUE_PALETTE["TF"][2])
        return component

    def _build_support_xs(self):
        bb = self.pf_coil_xz.bounding_box
        width = self.params.tf_wp_depth.value + 2 * self.params.tk_tf_side.value
        half_width = 0.5 * width

        if bb.x_min < half_width:
            raise BuilderError("PF coil has too small a minimum radius!")

        alpha = np.arcsin(half_width / bb.x_min)
        inner_dr = half_width * np.tan(alpha)

        beta = np.arcsin(half_width / bb.x_max)
        outer_dr = half_width * np.tan(beta)

        x_min = bb.x_min - self.params.pf_s_g.value - inner_dr
        x_max = bb.x_max + self.params.pf_s_g.value + outer_dr
        z_min = bb.z_min
        z_max = bb.z_max
        box_inner = make_polygon(
            {
                "x": [x_min, x_max, x_max, x_min],
                "y": 0,
                "z": [z_min, z_min, z_max, z_max],
            },
            closed=True,
        )
        box_outer = offset_wire(box_inner, self.params.pf_s_tk_plate.value)
        return BluemiraFace([box_outer, box_inner])

    @staticmethod
    def _get_first_intersection(point, angle, wire):
        """
        Get the first intersection from a point along an angle with a wire.
        """
        point = np.array(point)
        x_out = point[0] + np.cos(angle) * VERY_BIG
        z_out = point[2] + np.sin(angle) * VERY_BIG
        dir_point = np.array([x_out, 0, z_out])

        correct_direction = dir_point - point
        correct_direction /= np.linalg.norm(correct_direction)

        plane = BluemiraPlane.from_3_points(point, dir_point, [x_out, 1, z_out])
        intersections = slice_shape(wire, plane)
        distances = []
        if intersections is None:
            return None

        directed_intersections = []
        for inter in intersections:
            direction = inter - point
            direction /= np.linalg.norm(direction)
            if np.dot(correct_direction, direction) >= 0:
                dx = inter[0] - point[0]
                dz = inter[2] - point[2]

                dist = np.hypot(dx, dz)
                distances.append(dist)
                directed_intersections.append(inter)

        if len(directed_intersections) > 0:
            i_min = np.argmin(distances)
            return directed_intersections[i_min]
        return None

    def _get_support_point_angle(self, support_face: BluemiraFace):
        bb = support_face.boundary[0].bounding_box
        z_down = bb.z_min
        z_up = bb.z_max

        distance = np.inf
        best_angle = None
        v1, v2, v3, v4 = None, None, None, None
        for z, sign in zip([z_up, z_down], [1, -1]):
            for angle in [0.5 * np.pi, 2 / 3 * np.pi, 1 / 3 * np.pi]:
                p_inters = []
                distances = []
                for x in [bb.x_min, bb.x_max]:
                    point = [x, 0, z]
                    p_inter = self._get_first_intersection(
                        point, sign * angle, self.tf_xz_keep_out_zone
                    )

                    if p_inter is not None:
                        d = np.hypot(point[0] - p_inter[0], point[2] - p_inter[2])
                        p_inters.append(p_inter)
                        distances.append(d)

                if len(p_inters) == 2:  # noqa: PLR2004
                    avg_distance = np.average(distances)
                    if avg_distance <= distance:
                        distance = avg_distance
                        v1 = np.array([bb.x_min, 0, z])
                        v2 = np.array([bb.x_max, 0, z])
                        v3 = p_inters[1]
                        v4 = p_inters[0]
                        best_angle = sign * angle

        if distance == np.inf:
            raise BuilderError("No intersections found!")

        return v1, v2, v3, v4, best_angle

    def _get_intersecting_wire(self, v1, v2, v3, v4, angle):
        # Add some offset to get one small wire when cutting
        v3 += 0.1 * np.array([np.cos(angle), 0, np.sin(angle)])
        v4 += 0.1 * np.array([np.cos(angle), 0, np.sin(angle)])

        cut_box = make_polygon([v1, v2, v3, v4], closed=True)

        return sorted(
            boolean_cut(self.tf_xz_keep_out_zone, cut_box), key=lambda wire: wire.length
        )[0]

    def _make_rib_profile(self, support_face):
        # Then, project sideways to find the minimum distance from a support point
        # to the TF coil
        v1, v2, v3, v4, angle = self._get_support_point_angle(support_face)

        # Get the intersection with the TF edge wire and use this for the rib profile
        intersection_wire = self._get_intersecting_wire(v1, v2, v3, v4, angle)

        # Make the closing wire, and make sure the polygon doesn't self-intersect
        v3 = intersection_wire.start_point().xyz.T[0]
        v4 = intersection_wire.end_point().xyz.T[0]

        inter1 = get_intersect(
            np.array([[v1[0], v3[0]], [v1[2], v3[2]]]),
            np.array([[v2[0], v4[0]], [v2[2], v4[2]]]),
        )
        if len(inter1[0]) > 0:
            v3, v4 = v4, v3

        closing_wire = make_polygon(
            {
                "x": [v3[0], v1[0], v2[0], v4[0]],
                "y": 0,
                "z": [v3[2], v1[2], v2[2], v4[2]],
            },
            closed=False,
        )
        rib_face = BluemiraFace(BluemiraWire([intersection_wire, closing_wire]))

        # Trim rib face if there is a collision
        result = boolean_cut(rib_face, BluemiraFace(self.tf_xz_keep_out_zone))

        if result:
            result.sort(key=lambda face: -face.area)
            rib_face = result[0]

        return rib_face

    def _make_ribs(self, width, support_face):
        xz_profile = self._make_rib_profile(support_face)
        # Calculate the rib gap width and make the ribs
        rib_list = []
        total_rib_tk = self.params.pf_s_n_plate.value * self.params.pf_s_tk_plate.value
        if total_rib_tk >= width:
            bluemira_warn(
                "PF coil support rib thickness and number exceed available thickness!"
                " You're getting a solid block instead"
            )
            gap_size = 0
            rib_block = extrude_shape(xz_profile, vec=(0, width, 0))
            rib_list.append(rib_block)
        else:
            gap_size = (width - total_rib_tk) / (self.params.pf_s_n_plate.value - 1)
            rib = extrude_shape(xz_profile, vec=(0, self.params.pf_s_tk_plate.value, 0))
            rib_list.append(rib)
            for _ in range(self.params.pf_s_n_plate.value - 1):
                rib = rib.deepcopy()
                rib.translate(vector=(0, self.params.pf_s_tk_plate.value + gap_size, 0))
                rib_list.append(rib)
        return rib_list

    def build_xyz(
        self,
    ) -> PhysicalComponent:
        """
        Build the x-y-z components of the PF coil support.
        """
        shape_list = []
        # First build the support block around the PF coil
        support_face = self._build_support_xs()
        # Trim support face is there is a collision
        support_face = boolean_cut(support_face, BluemiraFace(self.tf_xz_keep_out_zone))[
            0
        ]

        width = self.params.tf_wp_depth.value + 2 * self.params.tk_tf_side.value
        support_block = extrude_shape(support_face, vec=(0, width, 0))
        shape_list.append(support_block)

        # Make the rib x-z profile and ribs
        shape_list.extend(self._make_ribs(width, support_face))

        try:
            shape = boolean_fuse(shape_list)
        except GeometryError:
            bluemira_warn(
                "PFCoilSupportBuilder boolean_fuse failed, getting a BluemiraCompound"
                " instead of a BluemiraSolid, please check!"
            )
            shape = BluemiraCompound(shape_list)

        shape.translate(vector=(0, -0.5 * width, 0))
        component = PhysicalComponent(self.name, shape)
        apply_component_display_options(component, color=BLUE_PALETTE["TF"][2])
        return component


class StraightOISOptimisationProblem(OptimisationProblem):
    """
    Optimisation problem for a straight outer inter-coil structure

    Parameters
    ----------
    wire:
        Sub wire along which to place the OIS
    keep_out_zone:
        Region in which the OIS cannot be
    n_koz_discr:
        Number of discretisation points to use when checking the keep-out zone constraint
    """

    def __init__(
        self,
        wire: BluemiraWire,
        keep_out_zone: BluemiraFace,
        optimiser: Optional[_DeprecatedOptimiser] = None,
        n_koz_discr: int = 100,
    ):
        self.wire = wire
        self.n_koz_discr = n_koz_discr
        self.koz_points = (
            keep_out_zone.boundary[0].discretize(byedges=True, ndiscr=n_koz_discr).xz.T
        )
        if optimiser is not None:
            warnings.warn(
                "Use of StraightOISOptimisationProblem's 'optimiser' argument is "
                "deprecated and it will be removed in version 2.0.0.\n"
                "See "
                "https://bluemira.readthedocs.io/en/latest/optimisation/"
                "optimisation.html "
                "for documentation of the new optimisation module.",
                DeprecationWarning,
                stacklevel=2,
            )

    def objective(self, x: np.ndarray) -> float:
        """Objective function to maximise length."""
        return self.negative_length(x)

    def ineq_constraints(self) -> List[ConstraintT]:
        """The inequality constraints for the problem."""
        return [
            {
                "f_constraint": self.constrain_koz,
                "tolerance": np.full(self.n_koz_discr, 1e-6),
            },
            {
                "f_constraint": self.constrain_x,
                "df_constraint": self.df_constrain_x,
                "tolerance": np.array([1e-6]),
            },
        ]

    @staticmethod
    def bounds() -> Tuple[np.ndarray, np.ndarray]:
        """The optimisation parameter bounds."""
        return np.array([0, 0]), np.array([1, 1])

    @staticmethod
    def f_L_to_wire(wire: BluemiraWire, x_norm: List[float]):  # noqa: N802
        """
        Convert a pair of normalised L values to a wire
        """
        p1 = wire.value_at(x_norm[0])
        p2 = wire.value_at(x_norm[1])
        return make_polygon([p1, p2])

    @staticmethod
    def f_L_to_xz(wire: BluemiraWire, value: float) -> np.ndarray:
        """
        Convert a normalised L value to an x, z pair.
        """
        point = wire.value_at(value)
        return np.array([point[0], point[2]])

    def negative_length(self, x_norm: np.ndarray) -> float:
        """
        Calculate the negative length of the straight OIS

        Parameters
        ----------
        x_norm:
            Normalised solution vector

        Returns
        -------
        Negative length from the normalised solution vector
        """
        p1 = self.f_L_to_xz(self.wire, x_norm[0])
        p2 = self.f_L_to_xz(self.wire, x_norm[1])
        return -np.hypot(*(p2 - p1))

    def constrain_koz(self, x_norm: np.ndarray) -> np.ndarray:
        """
        Constrain the straight OIS to be outside a keep-out-zone

        Parameters
        ----------
        x_norm:
            Normalised solution vector

        Returns
        -------
        KOZ constraint array
        """
        straight_line = self.f_L_to_wire(self.wire, x_norm)
        straight_points = straight_line.discretize(ndiscr=self.n_koz_discr).xz.T
        return signed_distance_2D_polygon(straight_points, self.koz_points)

    @staticmethod
    def constrain_x(x_norm: np.ndarray) -> np.ndarray:
        """
        Constrain the second normalised value to be always greater than the first.
        """
        return x_norm[0] - x_norm[1]

    @staticmethod
    def df_constrain_x(x_norm: np.ndarray) -> np.ndarray:  # noqa: ARG004
        """
        Gradient of the constraint on  the solution vector
        """
        return np.array([1.0, -1.0])


@dataclass
class StraightOISDesignerParams(ParameterFrame):
    """
    Parameters for the StraightOISDesigner
    """

    tk_ois: Parameter[float]
    g_ois_tf_edge: Parameter[float]
    min_OIS_length: Parameter[float]


class StraightOISDesigner(Designer[List[BluemiraWire]]):
    """
    Design a set of straight length outer inter-coil structures.

    Parameters
    ----------
    params:
        ParameterFrame for the StraightOISDesigner
    build_config:
        Build config dictionary for the StraightOISDesigner
    tf_coil_xz_face:
        x-z face of the TF coil on the y=0 plane
    keep_out_zones:
        List of x-z keep_out_zone faces on the y=0 plane
    """

    param_cls = StraightOISDesignerParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame],
        build_config: Dict,
        tf_coil_xz_face: BluemiraFace,
        keep_out_zones: List[BluemiraFace],
    ):
        super().__init__(params, build_config)
        self.tf_face = tf_coil_xz_face
        self.keep_out_zones = keep_out_zones

    def run(self) -> List[BluemiraWire]:
        """
        Create and run the design optimisation problem.

        Returns
        -------
        A list of outer inter-coil structure wires on the y=0 plane.
        """
        inner_tf_wire = self.tf_face.boundary[1]
        koz_centreline = offset_wire(
            inner_tf_wire,
            self.params.g_ois_tf_edge.value,
            open_wire=False,
            join="arc",
        )
        ois_centreline = offset_wire(
            inner_tf_wire,
            2 * self.params.g_ois_tf_edge.value,
            open_wire=False,
            join="arc",
        )
        ois_regions = self._make_ois_regions(ois_centreline, koz_centreline)
        koz = self._make_ois_koz(koz_centreline)

        ois_wires = []
        for region in ois_regions:
            opt_problem = StraightOISOptimisationProblem(region, koz)
            result = opt_problem.optimise(
                x0=np.array([0.0, 1.0]),
                algorithm="COBYLA",
                opt_conditions={"ftol_rel": 1e-6, "max_eval": 1000},
            ).x
            p1 = region.value_at(result[0])
            p2 = region.value_at(result[1])
            wire = self._make_ois_wire(p1, p2)
            ois_wires.append(wire)
        return ois_wires

    def _make_ois_wire(self, p1, p2):
        """
        Make a rectangular wire from the two inner edge points
        """
        dx = p2[0] - p1[0]
        dz = p2[2] - p1[2]
        normal = np.array([dz, 0, -dx])
        normal /= np.linalg.norm(normal)
        tk = self.params.tk_ois.value
        p3 = p2 + tk * normal
        p4 = p1 + tk * normal
        return make_polygon([p1, p2, p3, p4], closed=True)

    def _make_ois_koz(self, koz_centreline):
        """
        Make the (fused) keep-out-zone for the outer inter-coil structures.
        """
        # Note we use the same offset to the exclusion zones as for the OIS
        # to the TF.
        koz_wires = [
            offset_wire(koz.boundary[0], self.params.g_ois_tf_edge.value)
            for koz in self.keep_out_zones
        ]
        koz_faces = [BluemiraFace(koz) for koz in koz_wires]

        return boolean_fuse([BluemiraFace(koz_centreline), *koz_faces])

    def _make_ois_regions(self, ois_centreline, koz_centreline):
        """
        Select regions that are viable for outer inter-coil structures
        """
        inner_wire = self.tf_face.boundary[1]
        # Drop the inboard (already connected by the vault)
        # Note we also drop the probable worst case of the edge corners of the OIS
        # colliding when swept.
        x_min = inner_wire.bounding_box.x_min + np.sqrt(2) * self.params.tk_ois.value
        z_min = self.tf_face.bounding_box.z_min - 0.1
        z_max = self.tf_face.bounding_box.z_max + 0.1
        inboard_cutter = BluemiraFace(
            make_polygon(
                {"x": [0, x_min, x_min, 0], "z": [z_min, z_min, z_max, z_max]},
                closed=True,
            )
        )
        cutter = BluemiraFace(koz_centreline)
        cutter = boolean_fuse([cutter, inboard_cutter, *self.keep_out_zones])

        ois_regions = boolean_cut(ois_centreline, cutter)

        # Drop regions that are too short for OIS
        big_ois_regions = []
        for region in ois_regions:
            length = np.sqrt(
                np.sum((region.start_point().xyz - region.end_point().xyz) ** 2)
            )
            if length > self.params.min_OIS_length.value:
                big_ois_regions.append(region)
        return big_ois_regions


@dataclass
class OISBuilderParams(ParameterFrame):
    """
    Outer intercoil structure parameters
    """

    n_TF: Parameter[int]
    tf_wp_depth: Parameter[float]
    tk_tf_side: Parameter[float]


class OISBuilder(Builder):
    """
    Outer intercoil structure builder
    """

    RIGHT_OIS = "TF OIS right"
    LEFT_OIS = "TF OIS left"
    OIS_XZ = "TF OIS"
    param_cls: Type[OISBuilderParams] = OISBuilderParams

    def __init__(
        self,
        params: Union[OISBuilderParams, Dict],
        build_config: Dict,
        ois_xz_profiles: Union[BluemiraWire, List[BluemiraWire]],
    ):
        super().__init__(params, build_config)
        if not isinstance(ois_xz_profiles, List):
            ois_xz_profiles = [ois_xz_profiles]
        self.ois_xz_profiles = ois_xz_profiles

    def build(self) -> Component:
        """
        Build the PF coil support component.
        """
        return self.component_tree(self.build_xz(), self.build_xy(), self.build_xyz())

    def build_xy(self):
        """
        Build the x-y component of the OIS
        """

    def build_xz(self):
        """
        Build the x-z component of the OIS
        """
        components = []
        for i, ois_profile in enumerate(self.ois_xz_profiles):
            face = BluemiraFace(ois_profile)
            component = PhysicalComponent(f"{self.OIS_XZ} {i}", face)
            apply_component_display_options(component, color=BLUE_PALETTE["TF"][2])
            components.append(component)
        return components

    def build_xyz(self):
        """
        Build the x-y-z component of the OIS
        """
        width = self.params.tf_wp_depth.value + 2 * self.params.tk_tf_side.value
        tf_angle = 2 * np.pi / self.params.n_TF.value
        centre_radius = 0.5 * width / np.tan(0.5 * tf_angle)
        direction = (-np.sin(0.5 * tf_angle), np.cos(0.5 * tf_angle), 0)
        half_plane = BluemiraPlane(base=(0, 0, 0), axis=direction)

        components = []
        for i, ois_profile in enumerate(self.ois_xz_profiles):
            ois_profile_1 = ois_profile.deepcopy()
            ois_profile_1.translate(vector=(0, 0.5 * width, 0))

            ois_profile_2 = ois_profile_1.deepcopy()
            ois_profile_2.rotate(
                base=(centre_radius, 0.5 * width, 0), degree=np.rad2deg(tf_angle)
            )

            # First we make the full OIS
            path = make_polygon(
                [ois_profile_1.center_of_mass, ois_profile_2.center_of_mass]
            )
            ois_right = sweep_shape([ois_profile_1, ois_profile_2], path)

            # Then we "chop" it in half, but without the boolean_cut operation
            # This is because I cba to write a project_shape function...
            ois_profile_mid = slice_shape(ois_right, half_plane)[0]

            path = make_polygon(
                [ois_profile_1.center_of_mass, ois_profile_mid.center_of_mass]
            )
            ois_right = sweep_shape([ois_profile_1, ois_profile_mid], path)
            ois_left = mirror_shape(ois_right, base=(0, 0, 0), direction=(0, 1, 0))

            right_component = PhysicalComponent(f"{self.RIGHT_OIS} {i+1}", ois_right)
            left_component = PhysicalComponent(f"{self.LEFT_OIS} {i+1}", ois_left)
            components.extend([left_component, right_component])

        for component in components:
            apply_component_display_options(component, color=BLUE_PALETTE["TF"][2])

        return components
