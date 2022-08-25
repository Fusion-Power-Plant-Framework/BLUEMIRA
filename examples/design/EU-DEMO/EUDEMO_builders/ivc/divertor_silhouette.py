# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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
Define builder for divertor
"""
import enum
import operator
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import NewParameter as Parameter
from bluemira.base.parameter_frame import NewParameterFrame as ParameterFrame
from bluemira.equilibria import Equilibrium
from bluemira.equilibria.find import find_flux_surface_through_point, get_legs
from bluemira.geometry.tools import make_polygon
from bluemira.geometry.wire import BluemiraWire


@dataclass
class DivertorSilhouetteParams(ParameterFrame):
    """Parameters for running the `WallSolver`."""

    div_L2D_ib: Parameter[float]
    div_L2D_ob: Parameter[float]
    div_Ltarg: Parameter[float]  # noqa: N815
    div_open: Parameter[bool]


class LegPosition(enum.Enum):
    """
    Enum classifying divertor/separatrix leg positions
    """

    INNER = enum.auto()
    OUTER = enum.auto()


def get_separatrix_legs(
    equilibrium: Equilibrium,
) -> Dict[LegPosition, List[BluemiraWire]]:
    """
    Find the separatrix legs for the given equilibrium.
    """
    # A flag specifying which end of the plasma (i.e., upper or lower)
    # we want the legs from will need to be added
    legs = get_legs(equilibrium)
    separatrix_legs = {
        LegPosition.INNER: [make_polygon(loop.xyz) for loop in legs["lower_inner"]],
        LegPosition.OUTER: [make_polygon(loop.xyz) for loop in legs["lower_outer"]],
    }
    return separatrix_legs


class DivertorSilhouetteDesigner(Designer[BluemiraWire]):

    INNER_BAFFLE = "inner_baffle"
    OUTER_BAFFLE = "outer_baffle"
    DOME = "dome"
    INNER_TARGET = "inner_target"
    OUTER_TARGET = "outer_target"

    param_cls = DivertorSilhouetteParams

    def __init__(
        self,
        params: Dict[str, DivertorSilhouetteParams],
        equilibrium: Equilibrium,
        wall,
    ):
        super().__init__(params)
        self.equilibrium = equilibrium
        self.x_limits = (wall.start_point().x[0], wall.end_point().x[0])
        self.z_limits = (wall.start_point().z[0], wall.end_point().z[0])
        self.leg_length = {
            LegPosition.INNER: self.params.div_L2D_ib,
            LegPosition.OUTER: self.params.div_L2D_ob,
        }
        self.separatrix_legs = get_separatrix_legs(self.equilibrium)

    def run(self) -> Tuple[BluemiraWire]:
        # Build the targets for each separatrix leg
        inner_target = self.make_target(LegPosition.INNER, self.INNER_TARGET)
        outer_target = self.make_target(LegPosition.OUTER, self.OUTER_TARGET)

        # Build the dome based on target positions
        inner_target_end = self._get_wire_end_with_smallest(inner_target, "z")
        outer_target_start = self._get_wire_end_with_smallest(outer_target, "z")

        dome = self.make_dome(inner_target_end, outer_target_start, label=self.DOME)

        # Build the baffles
        idx_inner = np.argmin(self.x_limits)
        idx_outer = np.argmax(self.x_limits)

        inner_baffle = self.make_inner_baffle(
            inner_target, self.x_limits[idx_inner], self.z_limits[idx_inner]
        )
        outer_baffle = self.make_outer_baffle(
            outer_target, self.x_limits[idx_outer], self.z_limits[idx_outer]
        )

        return inner_target, outer_target, dome, inner_baffle, outer_baffle

    def make_target(self, leg: LegPosition, label: str) -> BluemiraWire:
        """
        Make a divertor target for a the given leg.
        """
        sols = self._get_sols_for_leg(leg)

        # Just use the first scrape-off layer for now
        point = sols[0].value_at(distance=self.leg_length[leg])

        # Create some vertical targets for now. Eventually the target
        # angle will be derived from the grazing-angle parameter
        target_length = self.params.div_Ltarg
        target_coords = np.array(
            [
                [point[0], point[0]],
                [point[1], point[1]],
                [point[2] - target_length / 2, point[2] + target_length / 2],
            ]
        )
        return make_polygon(target_coords, label=label)

    def make_dome(
        self, start: Sequence[float], end: Sequence[float], label: str
    ) -> BluemiraWire:
        """
        Make a dome between the two given points
        The dome shape follows a constant line of flux that is closest
        to the input start coordinate. Finally, the nearset point on the
        flux surface to the end point and the end point are joined.
        """
        # Get the flux surface that crosses the through the start point
        # We can use this surface to guide the shape of the dome
        psi_start = self.equilibrium.psi(*start)
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
        idx = np.array(
            [
                np.argmin(np.hypot(*(flux_surface - start_coord))),
                np.argmin(np.hypot(*(flux_surface - end_coord))),
            ]
        )

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

        return make_polygon(dome, label=label)

    def _make_baffle(
        self,
        label: str,
        start: Sequence[float],
        end: Sequence[float],
    ) -> BluemiraWire:
        """
        Make a baffle.
        The baffle shape is a straight line between the given start and
        end points.

        Parameters
        ----------
        label: str
            The label to give the returned Component.
        start: Sequence
            The position (in x-z) to start drawing the baffle from,
            e.g., the outside end of a target.
        end: Sequence
            The position (in x-z) to stop drawing the baffle, e.g., the
            postion to the upper part of the first wall.
        """
        return make_polygon(
            np.array([[start[0], end[0]], [0, 0], [start[1], end[1]]]), label=label
        )

    def make_inner_baffle(
        self,
        target: BluemiraWire,
        x_lim: float,
        z_lim: float,
    ) -> BluemiraWire:
        """
        Build the inner baffle to join with the given target.
        """
        if self.params.div_open:
            raise NotImplementedError("Open divertor baffles not yet supported")
        else:
            inner_target_start = self._get_wire_end_with_largest(target, "x")
        return self._make_baffle(
            label=self.INNER_BAFFLE,
            start=np.array([x_lim, z_lim]),
            end=inner_target_start,
        )

    def make_outer_baffle(
        self,
        target: BluemiraWire,
        x_lim: float,
        z_lim: float,
    ) -> BluemiraWire:
        """
        Build the outer baffle to join with the given target.
        """
        if self.params.div_open:
            raise NotImplementedError("Open divertor baffles not yet supported")
        else:
            outer_target_end = self._get_wire_end_with_largest(target, "x")
        return self._make_baffle(
            label=self.OUTER_BAFFLE,
            start=outer_target_end,
            end=np.array([x_lim, z_lim]),
        )

    def _get_sols_for_leg(
        self, leg: LegPosition, layers: Iterable[int] = (0, -1)
    ) -> List[BluemiraWire]:
        """
        Get the selected scrape-off-leg layers from the separatrix legs.
        """
        sols = []
        for layer in layers:
            sols.append(self.separatrix_legs[leg][layer])
        return sols

    @staticmethod
    def _get_wire_end_with_smallest(wire: BluemiraWire, axis: str) -> np.ndarray:
        """
        Get the coordinates of the end of a wire with largest value in
        the given dimension
        """
        return DivertorSilhouetteDesigner._get_wire_end(wire, axis, operator.lt)

    @staticmethod
    def _get_wire_end_with_largest(wire: BluemiraWire, axis: str) -> np.ndarray:
        """
        Get the coordinates of the end of a wire with largest value in
        the given dimension
        """
        return DivertorSilhouetteDesigner._get_wire_end(wire, axis, operator.gt)

    @staticmethod
    def _get_wire_end(wire: BluemiraWire, axis: str, comp: Callable) -> np.ndarray:
        """
        Get the coordinates of the end of a wire whose coordinate in the
        given axis satisfies the comparision function.
        """
        allowed_axes = ["x", "z"]
        if axis not in allowed_axes:
            raise ValueError(
                f"Unrecognised axis '{axis}'. Must be one of: {allowed_axes}."
            )

        start_point = wire.start_point()
        end_point = wire.end_point()
        if comp(getattr(start_point, axis), getattr(end_point, axis)):
            return start_point.xz.flatten()
        return end_point.xz.flatten()
