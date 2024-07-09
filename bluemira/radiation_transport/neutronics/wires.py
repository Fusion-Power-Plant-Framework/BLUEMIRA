# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Info about straight line wires and circles. Made to be simpler to modify than a whole
BluemiraWire.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy import typing as npt

from bluemira.geometry.error import GeometryError
from bluemira.geometry.tools import make_circle_arc_3P, make_polygon
from bluemira.geometry.wire import BluemiraWire

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class StraightLineInfo(NamedTuple):
    """Key information about a straight line"""

    start_point: Iterable[float]  # 3D coordinates
    end_point: Iterable[float]  # 3D coordinates

    def reverse(self) -> StraightLineInfo:
        """Flip the wire"""
        return StraightLineInfo(self.end_point, self.start_point)


class CircleInfo(NamedTuple):
    """Arc of a Circle, LESS THAN 180°"""

    start_point: Iterable[float]  # 3D coordinates
    end_point: Iterable[float]  # 3D coordinates
    center: Iterable[float]  # 3D coordinates
    radius: float  # scalar

    def reverse(self) -> CircleInfo:
        """Flip the wire"""
        return CircleInfo(self.end_point, self.start_point, self.center, self.radius)


@dataclass
class WireInfo:
    """
    A tuple to store:
    1. the key points about this wire (and what kind of wire this is)
    2. The tangent to that wire at the start and end
    3. A copy of the wire itself
    """

    # TODO: Perhaps implement more classes so it also work with splines? Or justify why
    # we don't need to. Or merge this invention into an existing issue?

    key_points: StraightLineInfo | CircleInfo  # 2 points of xyz/ CircleInfo
    tangents: Sequence[Iterable[float]]  # 2 normalised directional vectors xyz
    wire: BluemiraWire | None = None

    def reverse(self) -> WireInfo:
        """Flip the wire"""
        return type(self)(
            self.key_points.reverse(), [-t for t in self.tangents[::-1]], None
        )

    @classmethod
    def from_2P(  # noqa: N802
        cls, start_point: npt.NDArray[np.float64], end_point: npt.NDArray[np.float64]
    ) -> WireInfo:
        """
        Create the WireInfo for a straight line (i.e. one where the key_points is of
        instance StraightLineInfo) using only two points.
        """
        direction = np.array(end_point) - np.array(start_point)
        normed_dir = np.array(direction) / np.linalg.norm(direction)
        return cls(
            StraightLineInfo(np.array(start_point), np.array(end_point)),
            [normed_dir, normed_dir],
        )


class WireInfoList:
    """A class to store info about a series of wires"""

    def __init__(self, info_list: Iterable[WireInfo]):
        self.info_list = list(info_list)
        for i, (prev_wire, curr_wire) in enumerate(pairwise(self.info_list)):
            if not np.array_equal(
                prev_wire.key_points[1], curr_wire.key_points[0]
            ):  # cannot be mixed type wires: either all Coordinates or all NDArray.
                raise GeometryError(f"wire {i + 1} must start where the wire {i} stops.")

    def __len__(self) -> int:
        """Number of wire infos"""
        return len(self.info_list)

    def __getitem__(self, index_or_slice) -> list[WireInfo] | WireInfo:
        """Get a WireInfo"""
        return self.info_list[index_or_slice]

    def __repr__(self) -> str:
        """String representation"""
        return super().__repr__().replace(" at ", f" of {len(self)} WireInfo at ")

    def pop(self, index):
        """Pop one element"""
        return self.info_list.pop(index)

    def get_3D_coordinates(self) -> npt.NDArray:
        """
        Get all vertices (i.e. joining points between different segments) along the wire.
        shape = (N+1, 3)
        """
        # assume continuity, which is already enforced during initialisation, so we
        # should be fine.
        return np.array(
            [
                self.info_list[0].key_points[0],
                *(seg.key_points[1] for seg in self.info_list),
            ],
            dtype=float,
        )

    @property
    def start_point(self):
        """The start_point for the entire series of wires"""
        return self.info_list[0].key_points[0]

    @start_point.setter
    def start_point(self, new_start_point: npt.NDArray[np.float64]):
        """
        Set the start_point to somewhere new. Note this doesn't change the tangents.
        """
        old_kp = self.info_list[0].key_points
        self.info_list[0].key_points = type(old_kp)(new_start_point, *old_kp[1:])

    @property
    def end_point(self):
        """The end_point for the entire series of wires"""
        return self.info_list[-1].key_points[1]

    @end_point.setter
    def end_point(self, new_end_point):
        """Set the end_point to somewhere new. Note this doesn't change the tangents."""
        old_kp = self.info_list[-1].key_points
        self.info_list[0].key_points = type(old_kp)(
            old_kp[0], new_end_point, *old_kp[2:]
        )

    def reverse(self) -> WireInfoList:
        """Flip this list of wires"""
        return WireInfoList([info.reverse() for info in self.info_list[::-1]])

    def restore_to_wire(self) -> BluemiraWire:
        """Re-create a bluemira wire from a series of WireInfo."""
        wire_list = []
        for info in self:
            start_end = np.array(info.key_points[:2])
            if info.wire:
                # quick way to get the wire back without doing any computation is by
                # directly copying.
                wire_list.append(info.wire)
                continue
            if isinstance(info.key_points, StraightLineInfo):
                info.wire = make_polygon(start_end.T, closed=False)
                wire_list.append(info.wire)
            else:
                # given two points on the circumference, only makes the SHORTER of the
                # two possible arcs of the circle.
                chord_intersect = start_end.mean(axis=0)
                direction = chord_intersect - info.key_points.center
                normed_dir = direction / np.linalg.norm(direction)
                middle = info.key_points.center + info.key_points.radius * normed_dir
                info.wire = make_circle_arc_3P(start_end[0], middle, start_end[1])
                wire_list.append(info.wire)
        return BluemiraWire(wire_list)
