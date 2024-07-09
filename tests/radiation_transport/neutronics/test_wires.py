# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.error import GeometryError
from bluemira.radiation_transport.neutronics.wires import (
    CircleInfo,
    StraightLineInfo,
    WireInfo,
    WireInfoList,
)


@pytest.mark.parametrize(
    "start",
    "end",
    [
        (Coordinates([0, 0, 0]), Coordinates([1, 1, 1])),
        (np.array([0, 0, 0]), np.array([1, 1, 1])),
        (np.array([0, 0]), np.array([1, 1])),
    ],
)
def test_StraightLineInfo_reverse(start, end):
    """Test reversing a simple straight line segment"""
    sl = StraightLineInfo(start, end)
    rev_sl = sl.reverse()
    np.testing.assert_array_equal(sl.start_point, rev_sl.end_point)
    np.testing.assert_array_equal(sl.end_point, rev_sl.start_point)


@pytest.mark.parametrize(
    "start,end,center,radius",  # noqa: PT006
    [
        (
            Coordinates([0, 0, 0]),
            Coordinates([1, 0, 0]),
            Coordinates([0.5, 0.5, 0.0]),
            np.sqrt([0.25 + 0.25]),
        ),
        (
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0.5, 0.5, 0.0]),
            np.sqrt([0.25 + 0.25]),
        ),
        (
            np.array([0, 0]),
            np.array([1, 0]),
            np.array([0.5, 0.5]),
            np.sqrt([0.25 + 0.25]),
        ),
    ],
)
def test_CircleInfo_reverse(start, end, center, radius):
    """Test reversing a simple arc"""
    circ = CircleInfo(start, end, center, radius)
    rev_circ = circ.reverse()
    np.testing.assert_array_equal(circ.start_point, rev_circ.end_point)
    np.testing.assert_array_equal(circ.end_point, rev_circ.start_point)


def normalize(numpy_2d_3d_vector):
    """Short-hand to normalize the vector. Used only locally in this module."""
    return np.array(numpy_2d_3d_vector) / np.linalg.norm(np.array(numpy_2d_3d_vector))


ccw_rotation_2d = np.array([[0, -1], [1, 0]])
cw_rotation_3d = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])


class TestArcAndStraightLines:
    """Test various shapes made of straight lines and arcs jointed onto each other."""

    origin = np.array([0, 0])
    height_one = np.array([0, 1])
    offset_one = np.array([0.6, 1])
    offset_two = offset_one + height_one
    center = height_one + np.array([0, 0.5])

    question_mark = WireInfoList([
        WireInfo.from_2P([0.6, 0], offset_one),
        WireInfo(
            CircleInfo(offset_one, offset_two, center, np.sqrt(0.6**2 + 0.5**2)),
            [
                ccw_rotation_2d @ normalize(offset_one - center),
                ccw_rotation_2d @ normalize(offset_two - center),
            ],
        ),
    ])
    sickle_shape = WireInfoList([
        WireInfo.from_2P(origin, height_one),
        WireInfo.from_2P(height_one, offset_one),
        WireInfo(
            CircleInfo(offset_one, offset_two, center, np.sqrt(0.6**2 + 0.5**2)),
            [
                ccw_rotation_2d @ normalize(offset_one - center),
                ccw_rotation_2d @ normalize(offset_two - center),
            ],
        ),
    ])
    elbow_90_deg = WireInfoList([
        WireInfo.from_2P([1.0, -1.0], [1.0, 0.0]),
        WireInfo(
            CircleInfo(np.array([1.0, 0.0]), np.array([0.0, 1.0]), origin, 1.0),
            [np.array([0.0, 1.0]), np.array([-1.0, 0.0])],
        ),
        WireInfo.from_2P([0.0, 1.0], [-1.0, 1.0]),
    ])
    S_shape = WireInfoList([
        WireInfo(
            CircleInfo(
                np.array([-1.0, -1.0]), np.array([0.0, 0.0]), np.array([0.0, 1.0]), 1.0
            ),
            [np.array([0.0, 1.0]), np.array([1.0, 0.0])],
        ),
        WireInfo(
            CircleInfo(
                np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([0.0, -1.0]), 1.0
            ),
            [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        ),
    ])

    @pytest.mark.parametrize(
        "wire_list", [question_mark, sickle_shape, elbow_90_deg, S_shape]
    )
    def test_reverse(self, wire_list):
        """Check that reversing works on all these shapes."""
        start_point = wire_list.start_point
        end_point = wire_list.end_point
        rev_wire = wire_list.reverse()
        np.testing.assert_array_equal(wire_list.start_point, rev_wire.end_point)
        np.testing.assert_array_equal(wire_list.end_point, rev_wire.start_point)


@pytest.mark.parametrize(
    "wire1, wire2",  # noqa: PT006
    [
        (
            WireInfo.from_2P(Coordinates([0, 0, 0]), Coordinates([1, 0, 0])),
            WireInfo.from_2P(Coordinates([1, 0, 0]), Coordinates([1, 0, 1])),
        ),  # L-shape
        (
            WireInfo.from_2P(Coordinates([0, 0, 0]), Coordinates([1, 0, 0])),
            WireInfo.from_2P(Coordinates([1, 0, 1]), Coordinates([1, 0, 0])).reverse(),
        ),  # L-shape, reversed
        (
            WireInfo.from_2P(Coordinates([0, 0, 0]), Coordinates([0, 0, 1])),
            WireInfo.from_2P(Coordinates([0, 0, 1]), Coordinates([0, 0, -1])),
        ),  # backtrack
    ],
)
def test_wire(wire1, wire2):
    """Test wire made from Coordinates"""
    WireInfoList([wire1, wire2])


c_start = np.array([0, 0, 1])
c_end = np.array([0, 0, 0])
center = np.array([-0.1, 0.0, 0.5])


@pytest.mark.parametrize(
    "wire1, wire2",  # noqa: PT006
    [
        (
            WireInfo.from_2P(Coordinates([0, 0, 0]), Coordinates([1, 0, 0])),
            WireInfo.from_2P(Coordinates([-1, 0, 0]), Coordinates([1, 0, 0])),
        ),  # double over self, but second wire is reversed.
        (
            WireInfo.from_2P(Coordinates([0, 0, 1]), Coordinates([0, 0, 0])),
            WireInfo(
                CircleInfo(
                    Coordinates(c_start),
                    Coordinates(c_end),
                    Coordinates(center),
                    np.linalg.norm(c_end - center),
                ),
                [
                    cw_rotation_3d @ normalize(c_start - center),
                    cw_rotation_3d @ normalize(c_end - center),
                ],
            ),
        ),
    ],
)
def test_disjointed_wires(wire1, wire2):
    """Make sure that WireInfoList raises a Geometry error when the wire is broken/
    one segment is flipped so it's in the wrong order.
    """
    with pytest.raises(GeometryError):
        WireInfoList([wire1, wire2])
