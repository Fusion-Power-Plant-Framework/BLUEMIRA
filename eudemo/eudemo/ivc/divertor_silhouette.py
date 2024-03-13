# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Define builder for divertor
"""

from __future__ import annotations

import enum
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.base.reactor import ComponentManager
from bluemira.builders.divertor import DivertorBuilder
from bluemira.equilibria.find import find_flux_surface_through_point
from bluemira.equilibria.find_legs import LegFlux
from bluemira.geometry.tools import (
    interpolate_bspline,
    make_circle,
    make_polygon,
)
from bluemira.geometry.wire import BluemiraWire

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from bluemira.equilibria import Equilibrium


class Divertor(ComponentManager):
    """
    Wrapper around a divertor component tree.
    """

    def silhouette(self) -> BluemiraWire:
        """
        Returns
        -------
        :
            A wire representing the divertor poloidal silhouette.
        """
        return (
            self.component()
            .get_component("xz")
            .get_component(DivertorBuilder.BODY)
            .shape.boundary[0]
        )


@dataclass
class DivertorSilhouetteParams(ParameterFrame):
    """Parameters for running the `DivertorSilhouetteDesigner`."""

    div_type: Parameter[str]
    div_L2D_ib: Parameter[float]
    div_L2D_ob: Parameter[float]
    div_Ltarg_ib: Parameter[float]  # noqa: N815
    div_Ltarg_ob: Parameter[float]  # noqa: N815
    div_targ_angle_ib: Parameter[float]
    div_targ_angle_ob: Parameter[float]
    div_open: Parameter[bool]


class LegPosition(enum.Enum):
    """
    Enum classifying divertor/separatrix leg positions
    """

    INNER = enum.auto()
    OUTER = enum.auto()


class WireEndAxis(enum.Enum):
    """
    Enum for wire end axis
    """

    X = enum.auto()
    Z = enum.auto()


def get_separatrix_legs(
    equilibrium: Equilibrium,
) -> dict[LegPosition, list[BluemiraWire]]:
    """
    Find the separatrix legs for the given equilibrium.

    Returns
    -------
    :
        Legs for given position on separatrix
    """
    # A flag specifying which end of the plasma (i.e., upper or lower)
    # we want the legs from will need to be added
    legs = LegFlux(equilibrium).get_legs()
    return {
        LegPosition.INNER: [make_polygon(loop.xyz) for loop in legs["lower_inner"]],
        LegPosition.OUTER: [make_polygon(loop.xyz) for loop in legs["lower_outer"]],
    }


class DivertorSilhouetteDesigner(Designer[tuple[BluemiraWire, ...]]):
    """
    Designs the divertor silhouette to help design the divertor keep out zone

    Parameters
    ----------
    params:
        Divertor silhouette designer parameters
    equilibrium:
        The equilibrium to design around
    wall:
        wall boundary keep out zone (cut at divertor)

    """

    INNER_BAFFLE = "inner_baffle"
    OUTER_BAFFLE = "outer_baffle"
    DOME = "dome"
    INNER_TARGET = "inner_target"
    OUTER_TARGET = "outer_target"

    param_cls = DivertorSilhouetteParams
    params: DivertorSilhouetteParams

    def __init__(
        self,
        params: dict | ParameterFrame,
        equilibrium: Equilibrium,
        wall: BluemiraWire,
    ):
        super().__init__(params)
        if self.params.div_type.value == "DN":
            raise NotImplementedError("Double Null divertor not implemented")

        self.equilibrium = equilibrium
        self.x_limits = (wall.start_point().x[0], wall.end_point().x[0])
        self.z_limits = (wall.start_point().z[0], wall.end_point().z[0])
        self.leg_length = {
            LegPosition.INNER: self.params.div_L2D_ib,
            LegPosition.OUTER: self.params.div_L2D_ob,
        }
        self.separatrix_legs = get_separatrix_legs(self.equilibrium)

    def run(self) -> tuple[BluemiraWire, ...]:
        """
        Run method of DivertorSilhouetteDesigner


        Returns
        -------
        :
            Inner baffle
        :
            Inner target
        :
            Dome
        :
            Outer target
        :
            Outer baffle
        """
        # Build the targets for each separatrix leg
        inner_target = self.make_target(LegPosition.INNER, self.INNER_TARGET)
        outer_target = self.make_target(LegPosition.OUTER, self.OUTER_TARGET)

        # The inner target ends inside the private flux region (lower psi_norm)
        # The outer target ends outside the private flux region (higher psi_norm)
        inner_target_start, inner_target_end = self._get_wire_ends_by_psi(inner_target)
        outer_target_end, outer_target_start = self._get_wire_ends_by_psi(outer_target)

        # Build the dome based on target positions
        dome = self.make_dome(inner_target_end, outer_target_start, label=self.DOME)

        # Build the baffles
        inner_baffle = self.make_baffle(
            self.INNER_BAFFLE,
            target_baffle_join_point=inner_target_start,
            target_dome_join_point=inner_target_end,
        )
        outer_baffle = self.make_baffle(
            self.OUTER_BAFFLE,
            target_baffle_join_point=outer_target_end,
            target_dome_join_point=outer_target_start,
        )

        return inner_baffle, inner_target, dome, outer_target, outer_baffle

    def make_target(self, leg: LegPosition, label: str) -> BluemiraWire:
        """
        Make a divertor target for a the given leg.

        Returns
        -------
        :
            Divertor target
        """
        target_coords = self._make_angled_target(leg)
        return make_polygon(target_coords, label=label)

    def _make_angled_target(self, leg: LegPosition):
        sol = self._get_sols_for_leg(leg)[0]

        target_point = sol.value_at(distance=self.leg_length[leg].value)
        # use a point slightly further along the leg to get a vector
        # tangent to the sep. leg in the expect direction
        # i.e. towards the increasing leg length
        post_target_point = sol.value_at(distance=self.leg_length[leg].value + 0.1)

        a = post_target_point - target_point
        a_hat = a / np.linalg.norm(a)

        # ccw angle
        theta = (
            np.deg2rad(self.params.div_targ_angle_ib.value)
            if leg is LegPosition.INNER
            else np.deg2rad(self.params.div_targ_angle_ob.value)
        )

        rot_matrix = np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 0, 0],
            [np.sin(theta), 0, np.cos(theta)],
        ])  # ccw rotation about y-axis

        b_hat = rot_matrix @ a_hat

        target_half_length = (
            0.5 * self.params.div_Ltarg_ib.value
            if leg is LegPosition.INNER
            else 0.5 * self.params.div_Ltarg_ob.value
        )

        p1 = target_point - b_hat * target_half_length
        p2 = target_point + b_hat * target_half_length
        return np.array([p1, p2]).T

    def make_dome(self, start: np.ndarray, end: np.ndarray, label: str) -> BluemiraWire:
        """
        Make a dome between the two given points.

        Returns
        -------
        :
            Divertor dome

        Notes
        -----
        The dome shape follows a constant line of flux that is closest to the input
        coordinates.
        The nearest point on the flux surface to the start point and the end point are
        joined.
        The flux surface is picked based on the lowest z coordinate of the start and end
        point to ensure a continuous divertor shape is produced.
        """
        # Get the flux surface that crosses the through the start or end point.
        # We can use this surface to guide the shape of the dome.
        psi_start = self.equilibrium.psi(*(start if start[1] < end[1] else end))
        flux_surface = find_flux_surface_through_point(
            self.equilibrium.x,
            self.equilibrium.z,
            self.equilibrium.psi(),
            start[0],
            start[1],
            psi_start,
        )

        # Get the indices of the closest points on the flux surface to
        # the input start and end points
        start_coord = np.array([[start[0]], [start[1]]])  # [[x], [z]]
        end_coord = np.array([[end[0]], [end[1]]])
        idx = np.array([
            np.argmin(np.hypot(*(flux_surface - start_coord))),
            np.argmin(np.hypot(*(flux_surface - end_coord))),
        ])

        # Make sure the start and end are in the right order
        if idx[0] > idx[1]:
            idx = idx[::-1]
            dome_contour = flux_surface[:, idx[0] + 1 : idx[1]]
            dome_contour = dome_contour[:, ::-1]
        else:
            dome_contour = flux_surface[:, idx[0] + 1 : idx[1]]

        # Build the coords of the dome in 3D (all(y == 0))
        dome = np.zeros((3, dome_contour.shape[1] + 2))
        dome[(0, 2), 0] = start_coord.T
        dome[(0, 2), 1:-1] = dome_contour
        dome[(0, 2), -1] = end_coord.T

        return interpolate_bspline(dome, label=label)

    def make_baffle(
        self,
        label: str,
        target_baffle_join_point: np.ndarray,
        target_dome_join_point: np.ndarray,
    ) -> BluemiraWire:
        """
        Make a baffle that joins the target wire to the blanket.

        Parameters
        ----------
        label:
            The label to give the returned Component.
        target_baffle_join_point:
            The position (in x-z) where the target connects to the baffle.
        target_dome_join_point:
            The position (in x-z) where the target connects to the dome.

        Returns
        -------
        :
            The baffle shape.

        Raises
        ------
        ValueError
            Baffle internal and external radii are not equal
        """

        def grad_xz(p1: np.ndarray, p2: np.ndarray):
            x1, z1 = p1[0], p1[1]
            x2, z2 = p2[0], p2[1]
            if np.isclose(z1, z2):
                return 0
            if np.isclose(x1, x2):
                return np.inf
            return (z1 - z2) / (x1 - x2)

        idx_inner = np.argmin(self.x_limits)
        idx_outer = np.argmax(self.x_limits)

        blanket_join_point = (
            np.array([
                self.x_limits[idx_inner],
                self.z_limits[idx_inner],
            ])
            if label == self.INNER_BAFFLE
            else np.array([
                self.x_limits[idx_outer],
                self.z_limits[idx_outer],
            ])
        )
        target_gradient = grad_xz(target_baffle_join_point, target_dome_join_point)

        return self._make_circular_baffle(
            label, blanket_join_point, target_baffle_join_point, target_gradient
        )

    def _make_circular_baffle(
        self,
        label: str,
        blanket_join_point: np.ndarray,
        target_baffle_join_point: np.ndarray,
        target_gradient: float,
    ) -> BluemiraWire:
        """
        Make a circular baffle which is tangent to the target.

        Parameters
        ----------
        label:
            The label to give the returned Component.
        blanket_join_point:
            The position (in x-z) where the baffle
            joins to the blanket wall
        target_start_point:
            The position (in x-z) where the baffle
            is tangent and joins to the target.
        target_end_point:
            The position (in x-z) where the target
            joins to the dome.
        """
        bx, bz = blanket_join_point[0], blanket_join_point[1]
        tx, tz = target_baffle_join_point[0], target_baffle_join_point[1]
        mt = target_gradient

        def solve(l1: Tuple[float, ...], l2: Tuple[float, ...]):
            A = np.array([[l1[0], l1[1]], [l2[0], l2[1]]])
            b = np.array([l1[2], l2[2]])
            return np.linalg.solve(A, b)

        # Solve the two linear equations below, in the form Ax + Bz = C

        # This comes from solving for where a circle
        # will intersect both the target and blanket join points
        l1 = (
            2 * bx - 2 * tx,  # A
            2 * bz - 2 * tz,  # B
            (bx**2) + bz**2 - (tx**2 + tz**2),  # C
        )

        # These are co-effs for a line perpendicular to the target
        # (through the target join point), found using
        # the fact that the gradient of the line is the negative
        # reciprocal of the target gradient
        # the equ: x + mt*z = tx + mt*tz
        l2 = (
            0 if mt == np.inf else 1,  # A
            1 if mt == np.inf else mt,  # B
            tz if mt == np.inf else tx + mt * tz,  # C
        )
        arc_center_point = solve(l1, l2)
        ox, oz = arc_center_point[0], arc_center_point[1]

        if oz > bz:
            raise ValueError(
                "Arc center point is above the blanket join point, "
                f"make the target angle sharper: {label}"
            )

        deg_t = np.rad2deg(np.arctan2(tz - oz, tx - ox))
        deg_b = np.rad2deg(np.arctan2(bz - oz, bx - ox))

        start_angle = deg_t
        end_angle = deg_b
        if label == self.OUTER_BAFFLE:
            start_angle = deg_b
            end_angle = deg_t

        radius_t = float(np.linalg.norm(target_baffle_join_point - arc_center_point))
        radius_b = float(np.linalg.norm(blanket_join_point - arc_center_point))

        if not np.isclose(radius_b, radius_t):
            raise ValueError("radi must be equal")

        # make_circle_arc_3P would have been used but it put the arc center
        # somewhere weird so could not be used
        return make_circle(
            radius=radius_t,
            center=(ox, 0, oz),
            axis=(0, -1, 0),
            start_angle=start_angle,
            end_angle=end_angle,
            label=label,
        )

    def _get_sols_for_leg(
        self, leg: LegPosition, layers: Iterable[int] = (0, -1)
    ) -> list[BluemiraWire]:
        """
        Get the selected scrape-off-leg layers from the separatrix legs.

        Returns
        -------
        :
            Separatrix legs
        """
        return [self.separatrix_legs[leg][layer] for layer in layers]

    @staticmethod
    def _get_wire_end_with_smallest(wire: BluemiraWire, axis: str) -> np.ndarray:
        """
        Get the coordinates of the end of a wire with largest value in
        the given dimension

        Returns
        -------
        :
            Wire end point
        """
        return DivertorSilhouetteDesigner._get_wire_end(wire, axis, operator.lt)

    @staticmethod
    def _get_wire_end_with_largest(wire: BluemiraWire, axis: str) -> np.ndarray:
        """
        Get the coordinates of the end of a wire with largest value in
        the given dimension

        Returns
        -------
        :
            Wire end point
        """
        return DivertorSilhouetteDesigner._get_wire_end(wire, axis, operator.gt)

    def _get_wire_ends_by_psi(self, wire: BluemiraWire) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the coordinates of the ends of a wire where the end
        with higher psi is returned first
        """
        start_point = wire.start_point()
        end_point = wire.end_point()

        psi_start = self.equilibrium.psi(x=start_point.x[0], z=start_point.z[0])
        psi_end = self.equilibrium.psi(x=end_point.x[0], z=end_point.z[0])

        # todo, should be the opposite but I don't know how to use psi_norm
        # and the value at specific points
        if psi_start < psi_end:
            return start_point.xz.flatten(), end_point.xz.flatten()
        return end_point.xz.flatten(), start_point.xz.flatten()

    @staticmethod
    def _get_wire_end(wire: BluemiraWire, axis: str, comp: Callable) -> np.ndarray:
        """
        Get the coordinates of the end of a wire whose coordinate in the
        given axis satisfies the comparison function.

        Returns
        -------
        :
            Wire end point
        """
        axis = WireEndAxis[axis.upper()].name.lower()

        start_point = wire.start_point()
        end_point = wire.end_point()
        if comp(getattr(start_point, axis), getattr(end_point, axis)):
            return start_point.xz.flatten()
        return end_point.xz.flatten()
