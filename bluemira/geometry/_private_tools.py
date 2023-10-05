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
A collection of private geometry tools for discretised geometry. Do not use these;
use primitive operations in geometry/tools.py instead.
"""

from functools import partial
from itertools import zip_longest
from typing import Any, Dict, List, Tuple

import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.codes import _freecadapi as cadapi
from bluemira.geometry.coordinates import (
    _validate_coordinates,
    get_angle_between_points,
    get_area,
    normal_vector,
    vector_intersect,
)
from bluemira.geometry.error import GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.wire import BluemiraWire
from bluemira.utilities.tools import flatten_iterable

# =============================================================================
# Errors
# =============================================================================


class MixedFaceAreaError(GeometryError):
    """
    An error to raise when the area of a mixed face does not give a good match to the
    area enclosed by the original coordinates.
    """


# =============================================================================
# Utility functions used exclusively in this file
# =============================================================================


def _segment_lengths(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Returns the length of each individual segment in a set of coordinates

    Parameters
    ----------
    x:
        x coordinates [m]
    y:
        y coordinates [m]
    z:
        z coordinates [m]

    Returns
    -------
    The array of the length of each individual segment in the coordinates
    """
    return np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)


def _side_vector(polygon_array: np.ndarray) -> np.ndarray:
    """
    Calculates the side vectors of an anti-clockwise polygon

    Parameters
    ----------
    polygon_array:
        The 2-D array of polygon point coordinates

    Returns
    -------
    sides:
        The 2-D array of the polygon side vectors
    """
    return polygon_array - np.roll(polygon_array, 1)


# =============================================================================
# Coordinate creation
# =============================================================================


def offset(
    x: np.ndarray, z: np.ndarray, offset_value: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a square-based offset of the coordinates (no splines). N-sized output

    Parameters
    ----------
    x:
        The x coordinate vector
    z:
        The x coordinate vector
    offset_value:
        The offset value [m]

    Returns
    -------
    xo:
        The x offset coordinates
    zo:
        The z offset coordinates
    """
    # check numpy arrays:
    x, z = np.array(x), np.array(z)
    # check closed:
    if (x[-2:] == x[:2]).all() and (z[-2:] == z[:2]).all():
        closed = True
    elif x[0] == x[-1] and z[0] == z[-1]:
        closed = True
        # Need to "double lock" it for closed curves
        x = np.append(x, x[1])
        z = np.append(z, z[1])
    else:
        closed = False
    p = np.array([np.array(x), np.array(z)])
    # Normal vectors for each side
    v = normal_vector(_side_vector(p))
    # Construct points offset
    off_p = np.column_stack(p + offset_value * v)
    off_p2 = np.column_stack(np.roll(p, 1) + offset_value * v)
    off_p = np.array([off_p[:, 0], off_p[:, 1]])
    off_p2 = np.array([off_p2[:, 0], off_p2[:, 1]])
    ox = np.empty((off_p2[0].size + off_p2[0].size,))
    oz = np.empty((off_p2[1].size + off_p2[1].size,))
    ox[0::2], ox[1::2] = off_p2[0], off_p[0]
    oz[0::2], oz[1::2] = off_p2[1], off_p[1]
    off_s = np.array([ox[2:], oz[2:]]).T
    pnts = [
        vector_intersect(off_s[i], off_s[i + 1], off_s[i + 3], off_s[i + 2])
        for i in range(len(off_s[:, 0]) - 2)[0::2]
    ]
    pnts.append(pnts[0])
    pnts = np.array(pnts)[:-1][::-1]  # sorted ccw nicely
    if closed:
        pnts = np.concatenate((pnts, [pnts[0]]))  # Closed
    else:  # Add end points
        pnts = np.concatenate((pnts, [off_s[0]]))
        pnts = np.concatenate(([off_s[-1]], pnts))
    # sorted ccw nicely - i know looks weird but.. leave us kids alone
    # drop nan values
    return pnts[~np.isnan(pnts).any(axis=1)][::-1].T


def make_circle_arc(
    radius: float,
    x_centre: float = 0,
    y_centre: float = 0,
    angle: float = 2 * np.pi,
    n_points: int = 200,
    start_angle: float = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make a circle arc of a specified radius and angle at a given location.

    Parameters
    ----------
    radius:
        The radius of the circle arc
    x_centre:
        The x coordinate of the circle arc centre
    y_centre:
        The y coordinate of the circle arc centre
    angle:
        The angle of the circle arc [radians]
    n_points:
        The number of points on the circle
    start_angle:
        The starting angle of the circle arc

    Returns
    -------
    x:
        The x coordinates of the circle arc
    y:
        The y coordinates of the circle arc
    """
    n = np.linspace(start_angle, start_angle + angle, n_points)
    x = x_centre + radius * np.cos(n)
    y = y_centre + radius * np.sin(n)
    if angle == 2 * np.pi:
        # Small number correction (close circle exactly)
        x[-1] = x[0]
        y[-1] = y[0]
    return x, y


# =============================================================================
# Coordinates conversion
# =============================================================================


def convert_coordinates_to_wire(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    label: str = "",
    method: str = "mixed",
    **kwargs: Dict[str, Any],
) -> BluemiraWire:
    """
    Converts the provided coordinates into a BluemiraWire using the specified method.

    Parameters
    ----------
    x:
        The x coordinates of points to be converted to a BluemiraWire object
    y:
        The y coordinates of points to be converted to a BluemiraWire object
    z:
        The z coordinates of points to be converted to a BluemiraWire object
    method:
        The conversion method to be used:

            - mixed (default): results in a mix of splines and polygons
            - polygon: pure polygon representation
            - spline: pure spline representation

    label:
        The label for the resulting BluemiraWire object
    kwargs:
        Any other arguments for the conversion method, see e.g. make_mixed_face

    Returns
    -------
    The resulting BluemiraWire from the conversion
    """
    method_map = {
        "mixed": make_mixed_wire,
        "polygon": partial(make_wire, spline=False),
        "spline": partial(make_wire, spline=True),
    }
    return method_map[method](x, y, z, label=label, **kwargs)


def convert_coordinates_to_face(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    method: str = "mixed",
    label: str = "",
    **kwargs: Dict[str, Any],
) -> BluemiraFace:
    """
    Converts the provided coordinates into a BluemiraFace using the specified method.

    Parameters
    ----------
    x:
        The x coordinates of points to be converted to a BluemiraFace object
    y:
        The y coordinates of points to be converted to a BluemiraFace object
    z:
        The z coordinates of points to be converted to a BluemiraFace object
    method: str
        The conversion method to be used:

            - mixed (default): results in a mix of splines and polygons
            - polygon: pure polygon representation
            - spline: pure spline representation

    label:
        The label for the resulting BluemiraFace object
    kwargs:
        Any other arguments for the conversion method, see e.g. make_mixed_face

    Returns
    -------
    The resulting BluemiraFace from the conversion
    """
    method_map = {
        "mixed": make_mixed_face,
        "polygon": partial(make_face, spline=False),
        "spline": partial(make_face, spline=True),
    }
    return method_map[method](x, y, z, label=label, **kwargs)


def make_mixed_wire(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    label: str = "",
    *,
    median_factor: float = 2.0,
    n_segments: int = 4,
    a_acute: float = 150.0,
    cleaning_atol: float = 1e-6,
    allow_fallback: bool = True,
    debug: bool = False,
) -> BluemiraWire:
    """
    Construct a BluemiraWire object from the provided coordinates using a combination of
    polygon and spline wires. Polygons are determined by having a median length larger
    than the threshold or an angle that is more acute than the threshold.

    Parameters
    ----------
    x:
        The x coordinates of points to be converted to a BluemiraWire object
    y:
        The y coordinates of points to be converted to a BluemiraWire object
    z:
        The z coordinates of points to be converted to a BluemiraWire object
    label:
        The label for the resulting BluemiraWire object

    Other Parameters
    ----------------
    median_factor:
        The factor of the median for which to filter segment lengths
        (below median_factor*median_length --> spline)
    n_segments:
        The minimum number of segments for a spline
    a_acute:
        The angle [degrees] between two consecutive segments deemed to be too
        acute to be fit with a spline.
    cleaning_atol:
        If a point lies within this distance [m] of the previous point then it will be
        treated as a duplicate and removed. This can stabilise the conversion in cases
        where the point density is too high for a wire to be constructed as a spline.
        By default this is set to 1e-6.
    allow_fallback:
        If True then a failed attempt to make a mixed wire will fall back to a polygon
        wire, else an exception will be raised. By default True.
    debug:
        Whether or not to print debugging information

    Returns
    -------
    The BluemiraWire of the mixed polygon/spline coordinates
    """
    mfm = MixedFaceMaker(
        x,
        y,
        z,
        label=label,
        median_factor=median_factor,
        n_segments=n_segments,
        a_acute=a_acute,
        cleaning_atol=cleaning_atol,
        debug=debug,
    )
    try:
        mfm.build()

    except RuntimeError as e:
        if allow_fallback:
            bluemira_warn(
                f"CAD: MixedFaceMaker failed with error {e} "
                "- falling back to a polygon wire."
            )
            return make_wire(x, y, z, label=label)
        raise

    return mfm.wire


def make_mixed_face(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    label: str = "",
    *,
    median_factor: float = 2.0,
    n_segments: int = 4,
    a_acute: float = 150.0,
    cleaning_atol: float = 1e-6,
    area_rtol: float = 5e-2,
    allow_fallback: bool = True,
    debug: bool = False,
) -> BluemiraFace:
    """
    Construct a BluemiraFace object from the provided coordinates using a combination of
    polygon and spline wires. Polygons are determined by having a median length larger
    than the threshold or an angle that is more acute than the threshold.

    Parameters
    ----------
    x:
        The x coordinates of points to be converted to a BluemiraFace object
    y:
        The y coordinates of points to be converted to a BluemiraFace object
    z:
        The z coordinates of points to be converted to a BluemiraFace object
    label:
        The label for the resulting BluemiraFace object

    Other Parameters
    ----------------
    median_factor:
        The factor of the median for which to filter segment lengths
        (below median_factor*median_length --> spline)
    n_segments:
        The minimum number of segments for a spline
    a_acute:
        The angle [degrees] between two consecutive segments deemed to be too
        acute to be fit with a spline.
    cleaning_atol:
        If a point lies within this distance [m] of the previous point then it will be
        treated as a duplicate and removed. This can stabilise the conversion in cases
        where the point density is too high for a wire to be constructed as a spline.
        By default this is set to 1e-6.
    area_rtol:
        If the area of the resulting face deviates by this relative value from the area
        enclosed by the provided coordinates then the conversion will fail and either
        fall back to a polygon-like face or raise an exception, depending on the setting
        of `allow_fallback`.
    allow_fallback:
        If True then a failed attempt to make a mixed face will fall back to a polygon
        wire, else an exception will be raised. By default True.
    debug:
        Whether or not to print debugging information

    Returns
    -------
    The BluemiraFace of the mixed polygon/spline coordinates
    """
    mfm = MixedFaceMaker(
        x,
        y,
        z,
        label=label,
        median_factor=median_factor,
        n_segments=n_segments,
        a_acute=a_acute,
        cleaning_atol=cleaning_atol,
        debug=debug,
    )
    try:
        mfm.build()

    except RuntimeError as e:
        if allow_fallback:
            bluemira_warn(
                f"CAD: MixedFaceMaker failed with error {e} "
                "- falling back to a polygon face."
            )
            return make_face(x, y, z, label=label)
        raise

    # Sometimes there won't be a RuntimeError, and you get a free SIGSEGV for your
    # troubles.
    face_area = mfm.face.area
    coords_area = get_area(x, y, z)
    if np.isclose(coords_area, face_area, rtol=area_rtol):
        return mfm.face
    if allow_fallback:
        bluemira_warn(
            f"CAD: MixedFaceMaker resulted in a face with area {face_area} "
            f"but the provided coordinates enclosed an area of {coords_area} "
            "- falling back to a polygon face."
        )
        return make_face(x, y, z, label=label)
    raise MixedFaceAreaError(
        f"MixedFaceMaker resulted in a face with area {face_area} "
        f"but the provided coordinates enclosed an area of {coords_area}."
    )


def make_wire(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, label: str = "", spline: bool = False
) -> BluemiraWire:
    """
    Makes a wire from a set of coordinates.

    Parameters
    ----------
    x:
        The x coordinates of points to be converted to a BluemiraWire object
    y:
        The y coordinates of points to be converted to a BluemiraWire object
    z:
        The z coordinates of points to be converted to a BluemiraWire object
    label:
        The label for the resulting BluemiraWire object
    spline:
        If True then creates the BluemiraWire using a Bezier spline curve, by default
        False

    Returns
    -------
    The BluemiraWire bound by the coordinates
    """
    wire_func = cadapi.interpolate_bspline if spline else cadapi.make_polygon
    return BluemiraWire(wire_func(np.array([x, y, z]).T), label=label)


def make_face(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, label: str = "", spline: bool = False
) -> BluemiraFace:
    """
    Makes a face from a set of coordinates.

    Parameters
    ----------
    x:
        The x coordinates of points to be converted to a BluemiraFace object
    y:
        The y coordinates of points to be converted to a BluemiraFace object
    z:
        The z coordinates of points to be converted to a BluemiraFace object
    label:
        The label for the resulting BluemiraFace object
    spline:
        If True then creates the BluemiraFace using a Bezier spline curve, by default
        False

    Returns
    -------
    The BluemiraFace bound by the coordinates
    """
    wire = make_wire(x, y, z, label=label, spline=spline)
    return BluemiraFace(wire, label=label)


class MixedFaceMaker:
    """
    Utility class for the creation of Faces that combine splines and polygons.

    Polygons are detected by median length and turning angle.

    Parameters
    ----------
    x:
        The x coordinates of points to be converted to a BluemiraFace object
    y:
        The y coordinates of points to be converted to a BluemiraFace object
    z:
        The z coordinates of points to be converted to a BluemiraFace object
    label:
        The label for the resulting BluemiraFace object

    Other Parameters
    ----------------
    median_factor:
        The factor of the median for which to filter segment lengths
        (below median_factor*median_length --> spline)
    n_segments:
        The minimum number of segments for a spline
    a_acute:
        The angle [degrees] between two consecutive segments deemed to be too
        acute to be fit with a spline.
    cleaning_atol:
        If a point lies within this distance [m] of the previous point then it will be
        treated as a duplicate and removed. This can stabilise the conversion in cases
        where the point density is too high for a wire to be constructed as a spline.
        By default this is set to 1e-6.
    debug:
        Whether or not to print debugging information
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        label: str = "",
        *,
        median_factor: float = 2.0,
        n_segments: int = 4,
        a_acute: float = 150.0,
        cleaning_atol: float = 1e-6,
        debug: bool = False,
    ):
        _validate_coordinates(x, y, z)
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.z = np.copy(z)
        self.num_points = len(x)

        self.label = label

        self.median_factor = median_factor
        self.n_segments = n_segments
        self.a_acute = a_acute
        self.cleaning_atol = cleaning_atol
        self.debug = debug

        # Constructors
        self.edges = None
        self.wire = None
        self.face = None
        self.polygon_loops = None
        self.spline_loops = None
        self.flag_spline_first = None
        self._debugger = None

    def build(self):
        """
        Carry out the MixedFaceMaker sequence to make a Face
        """
        # Get the vertices of polygon-like segments
        p_vertices = self._find_polygon_vertices()

        if len(p_vertices) > 0:
            # identify sequences of polygon indices
            p_sequences = self._get_polygon_sequences(p_vertices)

            if (
                len(p_sequences) == 1
                and p_sequences[0][0] == 0
                and p_sequences[0][-1] == len(p_vertices) - 1
            ):
                # All vertices are pure polygon-like so just make the wire
                self.wires = make_wire(self.x, self.y, self.z, spline=False)
            else:
                # Get the (negative) of the polygon sequences to get spline sequences
                s_sequences = self._get_spline_sequences(p_sequences)

                if self.debug:
                    print("p_sequences :", p_sequences)
                    print("s_sequences :", s_sequences)

                # Make coordinates for all the segments
                self._make_subcoordinates(p_sequences, s_sequences)

                # Make the wires for each of the sub-coordinates, and daisychain them
                self._make_subwires()
        else:
            # All vertices are pure spline-like so just make the wire
            self.wires = make_wire(self.x, self.y, self.z, spline=True)

        # Finally, make the OCC face from the wire formed from the boundary wires
        self._make_wire()
        self.face = BluemiraFace(self.wire, label=self.label)

    def _find_polygon_vertices(self) -> np.ndarray:
        """
        Finds all vertices in the Coordinates which belong to polygon-like edges

        Returns
        -------
        The vertices of the coordinates which are polygon-like (dtype=int)
        """
        seg_lengths = _segment_lengths(self.x, self.y, self.z)
        median = np.median(seg_lengths)

        long_indices = np.nonzero(seg_lengths > self.median_factor * median)[0]

        # find sharp angle indices
        angles = np.zeros(len(self.x) - 2)
        for i in range(len(self.x) - 2):
            angles[i] = get_angle_between_points(
                [self.x[i], self.y[i], self.z[i]],
                [self.x[i + 1], self.y[i + 1], self.z[i + 1]],
                [self.x[i + 2], self.y[i + 2], self.z[i + 2]],
            )
        if (
            self.x[0] == self.x[-1]
            and self.y[0] == self.y[-1]
            and self.z[0] == self.z[-1]
        ):
            # Get the angle over the closed joint
            join_angle = get_angle_between_points(
                [self.x[-2], self.y[-2], self.z[-2]],
                [self.x[0], self.y[0], self.z[0]],
                [self.x[1], self.y[1], self.z[1]],
            )
            angles = np.append(angles, join_angle)

        angles = np.rad2deg(angles)
        sharp_indices = np.nonzero((angles <= self.a_acute) & (angles != 0))[0]
        # Convert angle numbering to segment numbering (both segments of angle)
        sharp_edge_indices = []
        for index in sharp_indices:
            sharp_edges = [index + 1, index + 2]
            sharp_edge_indices.extend(sharp_edges)
        sharp_edge_indices = np.array(sharp_edge_indices)

        # build ordered set of polygon edge indices
        indices = np.unique(np.append(long_indices, sharp_edge_indices))

        # build ordered set of polygon vertex indices
        vertices = []
        for index in indices:
            if index == self.num_points:
                # If it is the last index, do not overshoot
                vertices.extend([index])
            else:
                vertices.extend([index, index + 1])
        return np.unique(np.array(vertices, dtype=int))

    def _get_polygon_sequences(self, vertices: np.ndarray) -> List[List[float]]:
        """
        Gets the sequences of polygon segments

        Parameters
        ----------
        vertices:
            The vertices of the lcoordinates which are polygon-like

        Returns
        -------
        The list of start and end tuples of the polygon segments
        list([start, end], [start, end])
        """
        sequences = []

        if len(vertices) == 0:
            return sequences

        start = vertices[0]
        for i, vertex in enumerate(vertices[:-1]):
            delta = vertices[i + 1] - vertex

            if i == len(vertices) - 2:
                # end of coordinates clean-up
                end = vertices[i + 1]
                sequences.append([start, end])
                break

            if delta <= self.n_segments:
                # Spline would be too short, so stitch polygons together
                continue
            end = vertex
            sequences.append([start, end])
            start = vertices[i + 1]  # reset start index

        if not sequences:
            raise GeometryError("Not a good candidate for a mixed face ==> spline")

        if (
            len(sequences) == 1
            and sequences[0][0] == 0
            and sequences[0][1] == len(vertices) - 1
        ):
            # Shape is a pure polygon
            return sequences

        # Now check the start and end of the coordinates, to see if a polygon segment
        # bridges the join
        first_p_vertex = sequences[0][0]
        last_p_vertex = sequences[-1][1]

        if (
            first_p_vertex <= self.n_segments
            and self.num_points - last_p_vertex <= self.n_segments
        ):
            start_offset = self.n_segments - first_p_vertex
            end_offset = (self.num_points - last_p_vertex) + self.n_segments
            total = start_offset + end_offset
            if total <= self.n_segments:
                start = sequences[-1][0]
                end = sequences[0][1]
                # Remove first sequence
                sequences = sequences[1:]
                # Replace last sequence with bridged sequence
                sequences[-1] = [start, end]

        last_p_vertex = sequences[-1][1]
        # There is a small spline section at the end of the coordinates, that
        # needs to be bridged
        if self.num_points - last_p_vertex <= self.n_segments and sequences[0][0] == 0:
            # There is no bridge -> take action
            start = sequences[-1][0]
            end = sequences[0][1]
            sequences = sequences[1:]
            sequences[-1] = [start, end]

        return sequences

    def _get_spline_sequences(self, polygon_sequences: np.ndarray) -> List[List[float]]:
        """
        Gets the sequences of spline segments

        Parameters
        ----------
        polygon_sequences:
            The list of start and end tuples of the polygon segments

        Returns
        -------
        The list of start and end tuples of the spline segments
        list([start, end], [start, end])
        """
        spline_sequences = []

        # Catch the start, if polygon doesn't start at zero, and there is no
        # bridge
        last = polygon_sequences[-1]
        if last[0] > last[1]:  # there is a polygon bridge
            pass  # Don't add a spline at the start
        else:
            # Check that the first polygon segment doesn't start at zero
            first = polygon_sequences[0]
            if first[0] == 0:
                pass
            else:  # It doesn't start at zero and there is no bridge: catch
                spline_sequences.append([0, first[0]])

        for i, seq in enumerate(polygon_sequences[:-1]):
            start = seq[1]
            end = polygon_sequences[i + 1][0]
            spline_sequences.append([start, end])

        # Catch the end, if polygon doesn't end at end
        if last[1] == self.num_points:
            # NOTE: if this is true, there can't be a polygon bridge
            pass
        elif last[0] > last[1]:  # there is a polygon bridge
            spline_sequences.append([last[1], polygon_sequences[0][0]])
        else:
            spline_sequences.append([last[1], self.num_points])

        # Check if we need to make a spline bridge
        spline_first = spline_sequences[0][0]
        spline_last = spline_sequences[-1][1]
        if (spline_first == 0) and (spline_last == self.num_points):
            # Make a spline bridge
            start = spline_sequences[-1][0]
            end = spline_sequences[0][1]
            spline_sequences = spline_sequences[1:]
            spline_sequences[-1] = [start, end]

        if spline_sequences[0][0] == 0:
            self.flag_spline_first = True
        else:
            self.flag_spline_first = False

        return spline_sequences

    def _clean_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """
        Clean the provided coordinates by removing any values that are closer than the
        instance's cleaning_atol value.

        Parameters
        ----------
        coords:
            3D array of coordinates to be cleaned.

        Returns
        -------
        3D array of cleaned coordinates.
        """
        mask = ~np.isclose(_segment_lengths(*coords), 0, atol=self.cleaning_atol)
        mask = np.insert(mask, 0, True)
        return coords[:, mask]

    def _make_subcoordinates(
        self, polygon_sequences: np.ndarray, spline_sequences: np.ndarray
    ):
        polygon_coords = []
        spline_coords = []

        for sequence, s_coords in [
            [polygon_sequences, polygon_coords],
            [spline_sequences, spline_coords],
        ]:
            for seg in sequence:
                if seg[0] > seg[1]:
                    # There is a bridge
                    coords = np.hstack(
                        (
                            np.array(
                                [self.x[seg[0] :], self.y[seg[0] :], self.z[seg[0] :]]
                            ),
                            np.array(
                                [
                                    self.x[0 : seg[1] + 1],
                                    self.y[0 : seg[1] + 1],
                                    self.z[0 : seg[1] + 1],
                                ]
                            ),
                        )
                    )
                else:
                    coords = np.array(
                        [
                            self.x[seg[0] : seg[1] + 1],
                            self.y[seg[0] : seg[1] + 1],
                            self.z[seg[0] : seg[1] + 1],
                        ]
                    )
                clean_coords = self._clean_coordinates(coords)
                if all(shape >= 2 for shape in clean_coords.shape):  # noqa: PLR2004
                    s_coords.append(clean_coords)

        self.spline_coords = spline_coords
        self.polygon_coords = polygon_coords

    def _make_subwires(self):
        # First daisy-chain correctly...
        coords_order = []
        if self.flag_spline_first:
            set1, set2 = self.spline_coords, self.polygon_coords
        else:
            set2, set1 = self.spline_coords, self.polygon_coords
        for i, (a, b) in enumerate(zip_longest(set1, set2)):
            if a is not None:
                coords_order.append(set1[i])
            if b is not None:
                coords_order.append(set2[i])

        for i, coords in enumerate(coords_order[:-1]):
            if not (coords[:, -1] == coords_order[i + 1][:, 0]).all():
                coords_order[i + 1] = coords_order[i + 1][:, ::-1]
                if i == 0 and not (coords[:, -1] == coords_order[i + 1][:, 0]).all():
                    coords = coords[:, ::-1]  # noqa: PLW2901
                    if not (coords[:, -1] == coords_order[i + 1][:, 0]).all():
                        coords_order[i + 1] = coords_order[i + 1][:, ::-1]

        if self.flag_spline_first:
            set1 = [
                make_wire(*spline_coord, spline=True)
                for spline_coord in self.spline_coords
            ]
            set2 = [
                make_wire(*polygon_coord, spline=False)
                for polygon_coord in self.polygon_coords
            ]
        else:
            set2 = [
                make_wire(*spline_coord, spline=True)
                for spline_coord in self.spline_coords
            ]
            set1 = [
                make_wire(*polygon_coord, spline=False)
                for polygon_coord in self.polygon_coords
            ]

        wires = []
        for a, b in zip_longest(set1, set2):
            if a is not None:
                wires.append(a)

            if b is not None:
                wires.append(b)

        self.wires = list(flatten_iterable(wires))
        self._debugger = coords_order

    def _make_wire(self):
        self.wire = BluemiraWire(self.wires)

    def _make_face(self):
        self.face = BluemiraFace(self.wire)
