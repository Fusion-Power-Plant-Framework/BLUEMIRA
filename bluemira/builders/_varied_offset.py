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
Module containing functions to generate variable offset curves
"""

import numpy as np

from bluemira.geometry.error import GeometryError
from bluemira.geometry.tools import find_clockwise_angle_2d, interpolate_bspline
from bluemira.geometry.wire import BluemiraWire


def varied_offset(
    wire: BluemiraWire,
    inboard_offset: float,
    outboard_offset: float,
    inboard_offset_degree: float,
    outboard_offset_degree: float,
    num_points: int = 200,
) -> BluemiraWire:
    """
    Create a new wire that offsets the given wire using a variable
    offset in the xz plane.

    All angles are measured from the negative x-direction (9 o'clock),
    centred at the center of mass of the wire.
    The offset will be 'inboard_offset' between the negative x-direction
    and 'inboard_offset_degree'. Between 'outboard_offset_degree' and
    the positive x-direction the offset will be 'outboard_offset'. Between
    those angles, the offset will linearly transition between the min
    and max.

    Parameters
    ----------
    wire:
        The wire to create the offset from. This should be convex in
        order to get a sensible, non-intersecting, offset.
    inboard_offset:
        The size of the offset on the inboard side.
    outboard_offset:
        The size of the offset on the outboard side.
    inboard_offset_degree:
        The angle at which the variable offset should begin, in degrees.
    outboard_offset_degree:
        The angle at which the variable offset should end, in degrees.
    num_points:
        The number of points to use in the discretization of the input
        wire.

    Returns
    -------
    New wire at a variable offset to the input.
    """
    _throw_if_inputs_invalid(wire, inboard_offset_degree, outboard_offset_degree)
    coordinates = wire.discretize(num_points, byedges=True)
    if not np.all(coordinates.normal_vector == [0, 1, 0]):
        raise GeometryError(
            "Cannot create a variable offset from a wire that is not xz planar."
        )
    wire_coords = coordinates.xz
    inboard_offset_degree = np.radians(inboard_offset_degree)
    outboard_offset_degree = np.radians(outboard_offset_degree)
    center_of_mass = wire.center_of_mass[[0, 2]].reshape((2, 1))

    ib_axis = np.array([-1, 0])
    angles = np.radians(find_clockwise_angle_2d(ib_axis, wire_coords - center_of_mass))
    # Sort angles so coordinates are always clockwise and normals point outward
    angles, wire_coords = _sort_coords_by_angle(angles, wire_coords)

    offsets = _calculate_offset_magnitudes(
        angles,
        inboard_offset_degree,
        outboard_offset_degree,
        inboard_offset,
        outboard_offset,
    )
    normals = _calculate_normals_2d(wire_coords)
    new_shape_coords = wire_coords + normals * offsets
    return _2d_coords_to_wire(new_shape_coords)


def _throw_if_inputs_invalid(wire, inboard_offset_degree, outboard_offset_degree):
    if not wire.is_closed():
        raise GeometryError(
            "Cannot create a variable offset from a wire that is not closed."
        )
    if not 0 < inboard_offset_degree < 180:  # noqa: PLR2004
        raise ValueError("Inboard offset angle must be in the range [0, 180].")
    if not 0 < outboard_offset_degree < 180:  # noqa: PLR2004
        raise ValueError("Outboard offset angle must be in the range [0, 180].")
    if inboard_offset_degree > outboard_offset_degree:
        raise ValueError(
            "Inboard offset angle must be less than outboard angle. "
            f"Found '{inboard_offset_degree}' and '{outboard_offset_degree}'."
        )


def _sort_coords_by_angle(angles: np.ndarray, coords: np.ndarray):
    """Sort the given angles and use that to re-order the coords."""
    angle_sort_idx = np.argsort(angles)
    return angles[angle_sort_idx], coords[:, angle_sort_idx]


def _calculate_offset_magnitudes(
    angles,
    inboard_offset_degree,
    outboard_offset_degree,
    inboard_offset,
    outboard_offset,
):
    """Calculate the magnitude of the offset at each angle."""
    offsets = np.empty_like(angles)
    # All angles less than inboard_offset_degree set to min offset
    constant_minor_offset_idxs = np.logical_or(
        angles < inboard_offset_degree, angles > (2 * np.pi - inboard_offset_degree)
    )
    offsets[constant_minor_offset_idxs] = inboard_offset

    # All angles greater than outboard_offset_degree set to max offset
    constant_major_offset_idxs = np.logical_and(
        angles > outboard_offset_degree, angles < 2 * np.pi - outboard_offset_degree
    )
    offsets[constant_major_offset_idxs] = outboard_offset

    variable_offset_idxs = np.logical_not(
        np.logical_or(constant_minor_offset_idxs, constant_major_offset_idxs)
    )
    offsets[variable_offset_idxs] = _calculate_variable_offset_magnitudes(
        angles[variable_offset_idxs],
        inboard_offset_degree,
        outboard_offset_degree,
        inboard_offset,
        outboard_offset,
    )
    return offsets


def _calculate_variable_offset_magnitudes(
    angles, start_angle, end_angle, inboard_offset, outboard_offset
):
    """
    Calculate the variable offset magnitude for each of the given angles.

    The offset increases linearly between start_angle and end_angle,
    between inboard_offset and outboard_offset.
    """
    angles[angles > np.pi] = 2 * np.pi - angles[angles > np.pi]
    angle_fraction = (angles - start_angle) / (end_angle - start_angle)
    return inboard_offset + angle_fraction * (outboard_offset - inboard_offset)


def _calculate_normals_2d(wire_coords: np.ndarray) -> np.ndarray:
    """
    Calculate the unit normals to the tangents at each of the given
    coordinates.

    Note that this applies an anti-clockwise rotation to the tangents,
    so to get an outward facing normal to a polygon, the coordinates
    should be ordered in the clockwise direction.
    """
    gradients = np.gradient(wire_coords, axis=1)
    normals = np.array([-gradients[1], gradients[0]])
    return normals / np.linalg.norm(normals, axis=0)


def _2d_coords_to_wire(coords_2d):
    """
    Build a wire from a 2D array of coordinates using a bspline.
    """
    coords_3d = np.zeros((3, coords_2d.shape[1]))
    coords_3d[(0, 2), :] = coords_2d
    return interpolate_bspline(coords_3d, closed=True)
