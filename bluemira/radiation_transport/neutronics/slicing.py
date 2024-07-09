# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Oversees the conversion from bluemira wires into pre-cells, then into csg."""

from __future__ import annotations

from itertools import chain, pairwise, starmap
from typing import TYPE_CHECKING

import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.codes import _freecadapi as cadapi
from bluemira.geometry.constants import EPS_FREECAD
from bluemira.geometry.coordinates import choose_direction, get_bisection_line
from bluemira.geometry.error import GeometryError
from bluemira.geometry.plane import (
    BluemiraPlane,
    calculate_plane_dir,
    x_plane,
    xz_plane_from_2_points,
    z_plane,
)
from bluemira.geometry.tools import get_wire_plane_intersect, make_polygon
from bluemira.radiation_transport.neutronics.constants import (
    DISCRETISATION_LEVEL,
    TOLERANCE_DEGREES,
)
from bluemira.radiation_transport.neutronics.make_pre_cell import (
    DivertorPreCell,
    DivertorPreCellArray,
    PreCell,
    PreCellArray,
)
from bluemira.radiation_transport.neutronics.wires import (
    CircleInfo,
    StraightLineInfo,
    WireInfo,
    WireInfoList,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import numpy.typing as npt

    from bluemira.geometry.wire import BluemiraWire


def cut_curve(
    cut_points: list[npt.NDArray],
    wire: BluemiraWire,
    discretisation_level: int,
    *,
    reverse: bool = False,
) -> Iterator[npt.NDArray[np.float64]]:
    """
    Generator to cut a closed BluemiraWire to size.
    Current implementation is to yield a parameter range for every sequential cut-point
    pair. Subject to change of implementation in the future when issue 3038 is fixed.

    Parameters
    ----------
    cut_points:
        A list of at least 3 points, presumed to be lying on the wire, that we want to
        cut the wire at.
    wire:
        The wire that we want to cut up into multiple pieces (at the cut_points).
    discretisation_level:
        We yield a list of points of len==discretisation_level, such that we can build a
        wire made of (discretisation_level-1) straight lines to approximate each segment.
    reversed:
        Whether we want the neutron spectrum to go in the increasing or the decreasing
        direction.

    Yields
    ------
    params_range:
        For each segment, yield the values of the parametric variable t such that
        `[wire.value_at(t) for t in params_range]`
        is a list of points (sufficient to built up a series of (discretisation_level-1)
        straight lines that approximate that segment)
    """
    cut_params = [wire.parameter_at(cp) for cp in cut_points]
    # for reference: cut_params has range:
    # [0, 1] for an open wire, [0, 1) for a closed wire.

    # determine whether we're going up or down in parameter (t) space.
    finite_difference = np.diff(cut_params)
    if len(finite_difference) <= 2:  # noqa: PLR2004
        raise GeometryError(
            "Too few points! I.e. discretisation_level parameter too low. "
            "Can't determine the cut direction!"
        )
    if (finite_difference <= 0).sum() <= 1:
        # strictly monotonically increasing except for 1 wrap-around point
        increasing = True
    elif (finite_difference >= 0).sum() <= 1:
        # strictly monotonically decreasing except for 1 wrap-around point
        increasing = False
    else:  # no discernible pattern in the increase/decrease
        raise GeometryError("Points are too disordered!")

    # generator function
    for alpha, beta in pairwise(cut_params):
        used_alpha = alpha
        if increasing:
            if alpha > beta:  # alpha is expected to be smaller than beta
                used_alpha -= 1.0
        elif alpha < beta:  # alpha is expected to be larger than beta.
            used_alpha += 1.0
        param_range = np.linspace(used_alpha, beta, discretisation_level) % 1.0
        yield param_range[::-1] if reverse else param_range


def check_and_breakdown_wire(wire: BluemiraWire) -> WireInfoList:
    """
    Raise GeometryError if the BluemiraWire has an unexpected data storage structure.
    Then, get only the key information (start/end points and tangent) of each segment of
    the wire.
    """
    wire_container = []

    def add_line(
        edge: cadapi.apiEdge,
        wire: BluemiraWire,
        start_vec: cadapi.apiVector | npt.NDArray,
        end_vec: cadapi.apiVector | npt.NDArray,
    ):
        """Function to record a line"""
        return WireInfo(
            StraightLineInfo(np.array(start_vec), np.array(end_vec)),
            [edge.tangentAt(edge.FirstParameter), edge.tangentAt(edge.LastParameter)],
            wire,
        )

    def add_circle(
        edge: cadapi.apiEdge,
        wire: BluemiraWire,
        start_vec: cadapi.apiVector | npt.NDArray,
        end_vec: cadapi.apiVector | npt.NDArray,
    ):
        """Function to record the arc of a circle."""
        return WireInfo(
            CircleInfo(
                np.array(start_vec),
                np.array(end_vec),
                np.array(edge.Curve.Center),
                edge.Curve.Radius,
            ),
            [edge.tangentAt(edge.FirstParameter), edge.tangentAt(edge.LastParameter)],
            wire,
        )

    for w_edge in wire.edges:
        if len(w_edge.boundary) != 1 or len(w_edge.boundary[0].OrderedEdges) != 1:
            raise GeometryError("Expected each boundary to contain only 1 curve!")
        edge = w_edge.boundary[0].OrderedEdges[0]

        # Create aliases for easier referring to variables.
        # The following line may become `edge.start_point(), edge.end_point()`
        # when PR # 3095 is merged
        current_start, current_end = edge.firstVertex().Point, edge.lastVertex().Point

        # Get the info about this segment of wire
        if isinstance(edge.Curve, cadapi.Part.BSplineCurve | cadapi.Part.BezierCurve):
            wire_container.extend(
                check_and_breakdown_wire(
                    make_polygon(w_edge.discretise(DISCRETISATION_LEVEL), closed=False)
                ).info_list
            )
            continue

        if isinstance(edge.Curve, cadapi.Part.Line | cadapi.Part.LineSegment):
            wire_info = add_line(edge, w_edge, current_start, current_end)

        elif isinstance(edge.Curve, cadapi.Part.ArcOfCircle | cadapi.Part.Circle):
            wire_info = add_circle(edge, w_edge, current_start, current_end)
        elif isinstance(edge.Curve, cadapi.Part.ArcOfEllipse | cadapi.Part.Ellipse):
            raise NotImplementedError("Conversion for ellipses are not available yet.")
            # TODO: implement this feature
        else:
            raise NotImplementedError(f"Conversion for {edge.Curve} not available yet.")

        # horrible hack to work around issue #3037, which is still an open issue:
        # wire segments are sometimes reversed for no apparent reason.
        if len(wire_container) == 0:
            wire_container.append(wire_info)
            continue
        distance_to_start = np.linalg.norm(
            wire_container[-1].key_points.end_point - wire_info.key_points.start_point
        )
        distance_to_end = np.linalg.norm(
            wire_container[-1].key_points.end_point - wire_info.key_points.end_point
        )
        if distance_to_end < distance_to_start:
            wire_info = wire_info.reverse()
        wire_container.append(wire_info)

    return WireInfoList(wire_container)


def turned_morethan_180(
    xyz_vector1: Sequence[float], xyz_vector2: Sequence[float], direction_sign: int
) -> bool:
    """
    Checked if one needs to rotate vector 1 by more than 180° in the specified direction
    by more than 180° to align with vector 2.

    Parameters
    ----------
    xyz_vector1, xyz_vector2: Sequence[float]
        xyz array of where the vector points.
    direction_sign: signed integer
        +1: evaluate rotation required in the counter-clockwise direction.
        -1: evaluate rotation required in the clockwise direction.
    """
    if xyz_vector1[1] != 0 or xyz_vector2[1] != 0:
        raise GeometryError("Tangent vector points out of plane!")
    angle1 = np.arctan2(xyz_vector1[2], xyz_vector1[0])
    angle2 = np.arctan2(xyz_vector2[2], xyz_vector2[0])
    if direction_sign == 1:
        if angle2 < angle1:
            angle2 += 2 * np.pi
        return (angle2 - angle1) >= np.pi
    if angle2 > angle1:
        angle2 -= 2 * np.pi
    return (angle1 - angle2) >= np.pi


def deviate_less_than(
    xyz_vector1: Sequence[float], xyz_vector2: Sequence[float], threshold_degrees: float
) -> bool:
    """Check if two vector's angles less than a certain threshold angle (in degrees)."""
    angle1 = np.arctan2(xyz_vector1[2], xyz_vector1[0])
    angle2 = np.arctan2(xyz_vector2[2], xyz_vector2[0])
    return np.rad2deg(abs(angle2 - angle1)) < threshold_degrees


def straight_lines_deviate_less_than(
    info1: WireInfo, info2: WireInfo, threshold_degrees: float
) -> bool:
    """
    Check that both lines are straight lines, then check if deviation is less than
    threshold_degrees or not
    """
    if not (
        isinstance(info1.key_points, StraightLineInfo)
        and isinstance(info2.key_points, StraightLineInfo)
    ):
        return False
    return deviate_less_than(info1.tangents[1], info2.tangents[0], threshold_degrees)


def break_wire_into_convex_chunks(
    wire: BluemiraWire, curvature_sign: int = -1
) -> list[WireInfoList]:
    """
    Break a wire up into several convex wires.
    Merge if they are almost collinear.

    Parameters
    ----------
    wire:
    curvature_sign:
        if it's -1: we allow each convex chunk's wire to turn right only.
        if it's 1: we allow each convex chunk's wire to turn left only.

    Returns
    -------
    convex_chunks:
        a list of WireInfos
    """
    wire_segments = list(check_and_breakdown_wire(wire))
    # initialising the first chunk
    convex_chunks, this_chunk = [], []
    chunk_start_tangent = wire_segments[0].tangents[0]

    def add_to_chunk(this_seg: WireInfo) -> None:
        """Add the info and wire into the current chunk and current wire."""
        if this_chunk and straight_lines_deviate_less_than(
            this_chunk[-1], this_seg, TOLERANCE_DEGREES
        ):
            # modify the previous line directly
            this_chunk[-1].key_points = StraightLineInfo(
                this_chunk[-1].key_points.start_point, this_seg.key_points.end_point
            )
            return
        this_chunk.append(WireInfo(this_seg.key_points, this_seg.tangents))

    for no, w_s in enumerate(wire_segments[1:]):
        add_to_chunk(wire_segments[no])
        prev_end_tangent = this_chunk[-1].tangents[-1]
        next_start_tangent = w_s.tangents[0]
        if deviate_less_than(
            this_chunk[-1].tangents[1], w_s.tangents[0], TOLERANCE_DEGREES
        ):
            continue
        interior_curve_turned_over_180 = turned_morethan_180(
            chunk_start_tangent, next_start_tangent, curvature_sign
        )
        if turned_morethan_180(prev_end_tangent, next_start_tangent, curvature_sign):
            convex_chunks.append(WireInfoList(this_chunk.copy()))
            this_chunk.clear()
            chunk_start_tangent = w_s.tangents[0]
        elif interior_curve_turned_over_180:
            # curled in on itself too much.
            bluemira_warn(
                "Divertor wire geometry possibly too extreme for program "
                "to handle. Check pre-cell visually by using the .plot_2d() methods "
                "on the relevant DivertorPreCell and DivertorPreCellArray."
            )

    add_to_chunk(wire_segments[-1])
    convex_chunks.append(WireInfoList(this_chunk.copy()))

    return convex_chunks


class PanelsAndExteriorCurve:
    """
    A collection of three objects, the first wall panels, the vacuum vessel interior
    curve, and the exterior curve.

    Parameters
    ----------
    panel_break_points:
        numpy array of shape==(N, 2), showing the RZ coordinates of joining points
        between adjacent first wall panels, running in the clockwise direction,
        from the inboard divertor to the top, then down to the outboard divertor.
    vv_interior:
        The BluemiraWire representing the inside surface of the vacuum vessel.
    vv_exterior:
        The BluemiraWire representing the outside surface of the vacuum vessel.
    """

    def __init__(
        self,
        panel_break_points: npt.NDArray[np.float64],
        vv_interior: BluemiraWire,
        vv_exterior: BluemiraWire,
    ):
        """Instantiate from a break point curve and the vv_interior.

        Parameters
        ----------
        panel_break_points:
            A series of 2D coordinate (of shape = (N+1, 3)) representing the N panels
            of the blanket. It (also) runs clockwise (inboard side to outboard side),
            same as vv_interior
        vv_interior:
            A BluemiraWire that runs clockwise, showing the vacuum vessel on the RHHP
            cross-section of the tokamak.
        """
        self.vv_interior = vv_interior
        self.vv_exterior = vv_exterior
        # shape = (N+1, 3)
        self.interior_panels = panel_break_points
        if len(self.interior_panels[0]) != 3 or np.ndim(self.interior_panels) != 2:  # noqa: PLR2004
            raise ValueError(
                "Expected an input np.ndarray of breakpoints of shape = "
                f"(N+1, 2). Instead received shape = {np.shape(panel_break_points)}."
            )

    def get_bisection_line(
        self, index: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate the bisection line that separates two panels at breakpoint[i].

        Parameters
        ----------
        index:
            the n-th breakpoint that we want the bisection line to project from.
            (N.B. There are N+1 breakpoints for N panels.)
            Thus this is the last point of the n-th panel.
        """
        p1, p2, p3 = self.interior_panels[index - 1 : index + 2, ::2]
        origin_2d, direction_2d = get_bisection_line(p1, p2, p3, p2)
        line_origin = np.insert(origin_2d, 1, 0, axis=-1)
        line_direction = np.insert(direction_2d, 1, 0, axis=-1)
        return line_origin, line_direction

    def calculate_cut_points(
        self,
        starting_cut: npt.NDArray[np.float64] | None,
        ending_cut: npt.NDArray[np.float64] | None,
        snap_to_horizontal_angle: float,
    ) -> tuple[list[npt.NDArray[np.float64]], ...]:
        """
        Cut the curves according to some specified criteria:
        In general, a line would be drawn from each panel break point outwards towards
        the curves. This would form the cut line.

        The space between two neighbouring cut line makes a PreCell.

        Usually these cut's angle is the bisection of the normal angles of its
        neighbouring panels, but in special rules applies when snap_to_horizontal_angle
        is set to a value >0.

        Since there is only one neighbouring panel at the start break point and end break
        point, starting_cut and end_cut must be specified. If not, they are chosen to be
        horizontal cuts.

        Parameters
        ----------
        snap_to_horizontal_angle:
            If the cutting plane is less than x degrees (°) away from horizontal,
            then snap it to horizontal.
            allowed range: [0, 90]
        starting_cut, ending_cut:
            Each is an ndarray with shape (2,), denoting the destination of the cut
            lines.

            The program cannot deduce what angle to cut the curves at these
            locations without extra user input. Therefore the user is able to use these
            options to specify the cut line's destination point for the first and final
            points respectively.

            For the first cut line, the cut line would start from interior_panels[0]
            and reach starting_cut.
            For the final cut line, the cut line would start from interior_panels[-1]
            and reach ending_cut.
            Both arguments have shape (2,) if given, representing the RZ coordinates.


        Returns
        -------
        vv_cut_points:
            cut points where the vacuum vessel interior wire is split
        exterior_cut_points:
            cut points where the vacuum vessel exterior wire is split
        """
        vv_cut_points, exterior_cut_points = [], []

        def add_cut_points(cutting_plane: BluemiraPlane, cut_direction: npt.NDArray):
            """
            Find where the cutting_plane intersect the self.vv_interior and
            self.vv_exterior; these points will eventually be used to cut up
            self.vv_interior and self.vv_exterior.

            N.B. These cut points must be sequentially added, i.e. they should
            follow the curves in the clockwise direction.
            """
            vv_cut_points.append(
                get_wire_plane_intersect(self.vv_interior, cutting_plane, cut_direction)
            )
            exterior_cut_points.append(
                get_wire_plane_intersect(self.vv_exterior, cutting_plane, cut_direction)
            )

        threshold_angle = np.deg2rad(snap_to_horizontal_angle)

        # initial cut point
        add_cut_points(
            *calculate_plane_dir(
                self.interior_panels[0], [starting_cut[0], 0, starting_cut[-1]]
            )
        )

        for i in range(1, len(self.interior_panels) - 1):
            origin, c_dir = self.get_bisection_line(i)

            if c_dir[0] == 0:
                plane = x_plane(self.interior_panels[i][0])  # vertical cut plane
            elif abs(np.arctan(c_dir[-1] / c_dir[0])) < threshold_angle:
                plane = z_plane(self.interior_panels[i][-1])  # horizontal cut plane
            else:
                plane = xz_plane_from_2_points(origin, origin + c_dir)
            add_cut_points(plane, c_dir)

        # final cut point
        add_cut_points(
            *calculate_plane_dir(
                self.interior_panels[-1], [ending_cut[0], 0, ending_cut[-1]]
            )
        )

        return vv_cut_points, exterior_cut_points

    def execute_curve_cut(
        self,
        discretisation_level: int,
        starting_cut: npt.NDArray[np.float64] | None = None,
        ending_cut: npt.NDArray[np.float64] | None = None,
        snap_to_horizontal_angle: float = 30.0,
    ) -> tuple[list[BluemiraWire], ...]:
        """
        Cut the exterior curve into a series and return them.
        This is the slowest part of the entire csg-creation process, because of the
        discretisation.

        Parameters
        ----------
        snap_to_horizontal_angle, starting_cut, ending_cut:
            See
            :meth:`~bluemira.radiation_transport.neutronics.slicing.PanelsAndExteriorCurve.calculate_cut_points`
        discretisation_level:
            how many points to use to approximate the curve.

        Returns
        -------
        vv_curve_segments:
            segments of the vacuum vessel interior curve forming each pre-cell's interior
            curve.
        exterior_curve_segments:
            segments of the vacuum vessel exterior curve forming each pre-cell's exterior
            curve.
        """
        # TODO: remove discretisation level when issue #3038 is fixed. The raw wire can
        # be used without discretisation then.
        vv_cut_points, exterior_cut_points = self.calculate_cut_points(
            starting_cut, ending_cut, snap_to_horizontal_angle
        )

        vv_curve_segments = [
            self.approximate_curve(
                self.vv_interior,
                t_range,
                label=f"vacuum vessel interior curve {i + 1}",
            )
            for i, t_range in enumerate(
                cut_curve(
                    vv_cut_points, self.vv_interior, discretisation_level, reverse=False
                )
            )
        ]

        exterior_curve_segments = [
            self.approximate_curve(
                self.vv_exterior,
                t_range,
                label=f"vacuum vessel exterior curve {i + 1}",
            )
            for i, t_range in enumerate(
                cut_curve(
                    exterior_cut_points,
                    self.vv_exterior,
                    discretisation_level,
                    reverse=False,
                )
            )
        ]

        return vv_curve_segments, exterior_curve_segments

    @staticmethod
    def approximate_curve(
        curve: BluemiraWire, param_range: npt.NDArray[np.float64], label=""
    ) -> BluemiraWire:
        """
        Given params_range (an iterable of floats of len n between 0 to 1), create n-1
        straight lines such that the n-th point corresponds to
        `curve.value_at(params_range[n])`.

        This implementation shall be replaced when issue #3038 gets resolved.
        """
        return make_polygon(
            np.array([curve.value_at(t) for t in param_range]).T,
            label=label,
            closed=False,  # we expect it to form an open curve.
        )

    def make_quadrilateral_pre_cell_array(
        self,
        discretisation_level: int = DISCRETISATION_LEVEL,
        starting_cut: npt.NDArray[np.float64] | None = None,
        ending_cut: npt.NDArray[np.float64] | None = None,
        snap_to_horizontal_angle: float = 30.0,
    ) -> PreCellArray:
        """
        Cut the exterior curve up, so that it would act as the exterior side of the
        quadrilateral pre-cell. Then, the panel would act as the interior side of the
        pre-cell. The two remaining sides are the counter-clockwise and clockwise walls.

        The vacuum vessel interior surface is sandwiched between them.
        """
        # make horizontal cuts if the starting and ending cuts aren't provided.
        if starting_cut is None:
            start_r = 0  # starting cut's final destination's r value
            starting_cut = np.array([start_r, self.interior_panels[0][-1]])
        if ending_cut is None:
            # ending cut's final destination's r value
            end_r = self.interior_panels[-1, 0] * 2
            ending_cut = np.array([end_r, self.interior_panels[-1][-1]])

        return PreCellArray([
            PreCell(
                make_polygon(self.interior_panels[i : i + 2][::-1].T, closed=False),
                vv_segment,
                exterior_segment,
            )
            for i, (vv_segment, exterior_segment) in enumerate(
                zip(
                    *self.execute_curve_cut(
                        discretisation_level,
                        starting_cut,
                        ending_cut,
                        snap_to_horizontal_angle,
                    ),
                    strict=True,
                )
            )
        ])


class DivertorWireAndExteriorCurve:
    """
    A class to store a wire with a vacuum vessel interior curve and an exterior curve.
    """

    def __init__(
        self,
        divertor_wire: BluemiraWire,
        vv_interior: BluemiraWire,
        vv_exterior: BluemiraWire,
    ):
        """
        Instantiate from a BluemiraWire of the divertor and a BluemiraWire of the
        exterior (outside of vacuum vessem) curve. Also save the BluemiraWire that goes
        between these two (inside of vacuum vessel).

        Parameters
        ----------
        divertor_wire
            A BluemiraWire that runs from the inboard side to the outboard side of the
            tokamak, representing the plasma facing surface of the divertor.
        vv_interior
            A BluemiraWire that runs clockwise, showing the inside of the vacuum vessel
            on the RHHP cross-section of the tokamak.
        vv_exterior
            A BluemiraWire that runs clockwise, showing the outside of the vacuum vessel
            on the RHHP cross-section of the tokamak.
        """
        self.convex_segments = break_wire_into_convex_chunks(divertor_wire)
        self.key_points = np.array([  # shape = (N+1, 3)
            *(seg.key_points[0] for seg in chain.from_iterable(self.convex_segments)),
            self.convex_segments[-1][-1].key_points[1],
        ])
        self.tangents = [  # shape = (N, 2, 3)
            seg.tangents for seg in chain.from_iterable(self.convex_segments)
        ]
        self.center_point = np.array([  # center of the divertor
            (self.key_points[0, 0] + self.key_points[-1, 0]) / 2,  # mean x
            self.key_points[:, -1].max(),  # highest z
        ])
        self.vv_interior = vv_interior
        self.vv_exterior = vv_exterior

    def get_projection_line(
        self, prev_idx: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Get the the line bisecting the x and the y, represented as origin and direction.

        Parameters
        ----------
        prev_idx
            two convex chunks are involved when making a bisection line. This index
            refers to the smaller of these two chunks' indices.

        Returns
        -------
        line_origin:
            a point on this line
        line_direction:
            a normal vector pointing in the direction of this line.
        """
        # Get the anchor point as an array of shape = (2,)
        anchor = self.convex_segments[prev_idx].end_point[::2]
        # force the directions to point outwards.
        unnormed_dir = anchor - self.center_point
        direct = unnormed_dir / np.linalg.norm(unnormed_dir)
        return np.insert(anchor, 1, 0, axis=-1), np.insert(direct, 1, 0, axis=-1)

    def calculate_cut_points(
        self,
        starting_cut: npt.NDArray[np.float64] | None,
        ending_cut: npt.NDArray[np.float64] | None,
    ) -> tuple[list[npt.NDArray[np.float64]], ...]:
        """
        Cut the curves up into N segments to match the N convex chunks of the
        divertor.
        The space between two neighbouring cut line makes a DivertorPreCell.

        Parameters
        ----------
        starting_cut, ending_cut:
            Each is an ndarray with shape (2,), denoting the destination of the cut
            lines.

            The program cannot deduce what angle to cut the curves at these
            locations without extra user input. Therefore the user is able to use these
            options to specify the cut line's destination point for the first and final
            points respectively.

            For the first cut line, the cut line would start from interior_panels[0] and
            reach starting_cut.
            For the final cut line, the cut line would start from interior_panels[-1] and
            reach ending_cut.
            Both arguments have shape (2,) if given, representing the RZ coordinates.


        Returns
        -------
        self.vv_cut_points
            cut points where the vacuum vessel interior wire is split
        self.exterior_cut_points
            cut points where the vacuum vessel exterior wire is split
        """
        vv_cut_points, exterior_cut_points = [], []

        def add_cut_points(cutting_plane: BluemiraPlane, cut_direction: npt.NDArray):
            """
            Find where the cutting_plane intersect the self.vv_interior and
            self.vv_exterior; these points will eventually be used to cut up
            self.vv_interior and self.vv_exterior.

            N.B. These cut points must be sequentially added, i.e. they should
            follow the curves in the clockwise direction.

            While identical to :meth:`~PanelsAndExteriorCurve.add_cut_points`,
            this can't be refactored away because they're specific to the class.
            """
            vv_cut_points.append(
                get_wire_plane_intersect(self.vv_interior, cutting_plane, cut_direction)
            )
            exterior_cut_points.append(
                get_wire_plane_intersect(self.vv_exterior, cutting_plane, cut_direction)
            )

        # initial cut point
        add_cut_points(
            *calculate_plane_dir(
                np.array(self.convex_segments[0][0].key_points[0]),
                [starting_cut[0], 0, starting_cut[-1]],
            )
        )

        for i in range(len(self.convex_segments) - 1):
            origin, c_dir = self.get_projection_line(i)
            plane = xz_plane_from_2_points(origin, origin + c_dir)
            add_cut_points(plane, c_dir)

        # final cut point
        add_cut_points(
            *calculate_plane_dir(
                np.array(self.convex_segments[-1][-1].key_points[1]),
                [ending_cut[0], 0, ending_cut[-1]],
            )
        )

        return vv_cut_points, exterior_cut_points

    def execute_curve_cut(
        self,
        discretisation_level: int,
        starting_cut: npt.NDArray[np.float64] | None,
        ending_cut: npt.NDArray[np.float64] | None,
    ) -> tuple[list[WireInfoList], ...]:
        """
        Cut the vacuum vessel curves into a series return these segments.

        (Obsolete)
        ----------
        Use the following table to decide whether the
        segments run clockwise or counter-clockwise.
        |                 | INCREASING = True | INCREASING = False |
        ------------------------------------------------------------
        | reverse = False |     clockwise     | counter-clockwise  |
        | reverse = False | counter-clockwise |     clockwise      |

        This is the slowest part of the entire csg-creation process, because of the
        discretisation.

        Parameters
        ----------
        starting_cut, ending_cut:
            See
            :meth:`~bluemira.radiation_transport.neutronics.slicing.DivertorWireAndExteriorCurve.calculate_cut_points`
        discretisation_level:
            how many points to use to approximate the curve.

        Returns
        -------
        vv_curve_segments

            list of WireInfoList describing the each pre-cell's vacuum vessel interior
            curve

        exterior_curve_segments

            list of WireInfoList describing the each pre-cell's vacuum vessel exterior
            curve
        """
        # TODO: remove discretisation level when issue #3038 is fixed. The raw wire can
        # be used without discretisation then.
        vv_cut_points, exterior_cut_points = self.calculate_cut_points(
            starting_cut, ending_cut
        )

        vv_curve_segments = [
            self.approximate_curve(self.vv_interior, t_range)
            for t_range in cut_curve(
                vv_cut_points, self.vv_interior, discretisation_level, reverse=True
            )
        ]

        exterior_curve_segments = [
            self.approximate_curve(self.vv_exterior, t_range)
            for t_range in cut_curve(
                exterior_cut_points, self.vv_exterior, discretisation_level, reverse=True
            )
        ]

        return vv_curve_segments, exterior_curve_segments

    @staticmethod
    def approximate_curve(
        curve: BluemiraWire, param_range: npt.NDArray[np.float64]
    ) -> WireInfoList:
        """
        Given params_range (an iterable of floats of len n between 0 to 1), create n-1
        straight lines such that the n-th point corresponds to
        `curve.value_at(params_range[n])`.

        This implementation shall be updated/replaced when issue #3038 gets resolved.
        """
        return WireInfoList(
            starmap(WireInfo.from_2P, pairwise([curve.value_at(t) for t in param_range]))
        )

    def make_divertor_pre_cell_array(
        self,
        starting_cut: npt.NDArray[np.float64] | None = None,
        ending_cut: npt.NDArray[np.float64] | None = None,
        discretisation_level: int = DISCRETISATION_LEVEL,
    ):
        """
        Cut the exterior curve up, so that would act as the exterior side of the
        quadrilateral pre-cell. Then, the panel would act as the interior side of the
        pre-cell. The two remaining sides are the counter-clockwise and clockwise walls.
        The vv_interior curve would be sandwiched between these two.

        Ordering: The cells should run from inoboard to outboard.

        Parameters
        ----------
        starting_cut:
            shape = (2,)
        ending_cut:
            shape = (2,)
        discretisation_level:
            integer: how many points to use to approximate each curve segment.
        """
        # deduced starting_cut and ending_cut from the divertor itself.
        if not starting_cut:
            first_point = np.array(self.convex_segments[0][0].key_points[0])[::2]
            tangent = np.array(self.convex_segments[0][0].tangents[0])[::2]
            tangent = choose_direction(tangent, self.center_point, first_point)
            starting_cut = first_point + tangent
        if not ending_cut:
            last_point = np.array(self.convex_segments[-1][-1].key_points[1])[::2]
            tangent = np.array(self.convex_segments[-1][-1].tangents[1])[::2]
            tangent = choose_direction(tangent, self.center_point, last_point)
            ending_cut = last_point + tangent

        pre_cell_list = []
        for i, (vv_segment, exterior_segment) in enumerate(
            zip(
                *self.execute_curve_cut(discretisation_level, starting_cut, ending_cut),
                strict=True,
            )
        ):
            interior_wire = self.convex_segments[i]
            # make a new line joining the start and end.
            # merge lines if collinear
            cw_line = (
                WireInfoList([interior_wire.pop(0)])
                if np.allclose(
                    exterior_segment.end_point,
                    interior_wire.start_point,
                    rtol=0,
                    atol=EPS_FREECAD,
                )
                else WireInfoList([
                    WireInfo.from_2P(
                        exterior_segment.end_point, interior_wire.start_point
                    )
                ])
            )

            while straight_lines_deviate_less_than(cw_line[-1], interior_wire[0], 0.5):
                cw_line.end_point = interior_wire.start_point
                interior_wire.pop(0)

            # merge lines if collinear
            ccw_line = (
                WireInfoList([interior_wire.pop(-1)])
                if np.allclose(
                    interior_wire.end_point,
                    exterior_segment.start_point,
                    rtol=0,
                    atol=EPS_FREECAD,
                )
                else WireInfoList([
                    WireInfo.from_2P(
                        interior_wire.end_point, exterior_segment.start_point
                    )
                ])
            )

            while straight_lines_deviate_less_than(interior_wire[-1], ccw_line[0], 0.5):
                ccw_line.start_point = interior_wire.end_point
                interior_wire.pop(-1)

            pre_cell_list.append(
                DivertorPreCell(interior_wire, vv_segment, exterior_segment)
            )

        return DivertorPreCellArray(pre_cell_list)
