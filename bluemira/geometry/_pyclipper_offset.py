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
Discretised offset operations used in case of failure in primitive offsetting.
"""

from copy import deepcopy
from typing import List, Tuple

import numpy as np
from pyclipper import (
    ET_CLOSEDPOLYGON,
    ET_OPENROUND,
    ET_OPENSQUARE,
    JT_MITER,
    JT_ROUND,
    JT_SQUARE,
    PolyTreeToPaths,
    PyPolyNode,
    PyclipperOffset,
    scale_from_clipper,
    scale_to_clipper,
)

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.geometry.coordinates import Coordinates, rotation_matrix_v1v2
from bluemira.geometry.error import GeometryError

__all__ = ["offset_clipper"]

# =============================================================================
# Pyclipper utilities
# =============================================================================


def coordinates_to_pyclippath(coordinates: Coordinates) -> np.ndarray:
    """
    Transforms a bluemira Coordinates object into a Path for use in pyclipper

    Parameters
    ----------
    coordinates:
        The Coordinates to be used in pyclipper

    Returns
    -------
    The vertex polygon path formatting required by pyclipper
    """
    return scale_to_clipper(coordinates.xz.T)


def pyclippath_to_coordinates(path: np.ndarray) -> Coordinates:
    """
    Transforms a pyclipper path into a bluemira Coordinates object

    Parameters
    ----------
    path:
        The vertex polygon path formatting used in pyclipper

    Returns
    -------
    The Coordinates from the path object
    """
    p2 = scale_from_clipper(np.array(path).T)
    return Coordinates({"x": p2[0], "y": 0, "z": p2[1]})


def pyclippolytree_to_coordinates(polytree: List[np.ndarray]) -> List[Coordinates]:
    """
    Converts a ClipperLib PolyTree into a list of Coordinates

    Parameters
    ----------
    polytree:
        The polytree to convert to Coordinates
    """
    paths = PolyTreeToPaths(polytree)
    return [pyclippath_to_coordinates(path) for path in paths]


class PyclipperMixin:
    """
    Mixin class for typical pyclipper operations and processing
    """

    name = NotImplemented

    def perform(self):
        """
        Perform the pyclipper operation
        """
        raise NotImplementedError

    def raise_warning(self):
        """
        Raise a warning if None is to be returned.
        """
        bluemira_warn(f"{self.name} operation on 2-D polygons returning None.\n")

    def handle_solution(self, solution: Tuple[np.ndarray]) -> List[Coordinates]:
        """
        Handles the output of the Pyclipper.Execute(*) algorithms, turning them
        into Coordaintes objects. NOTE: These are closed by default.

        Parameters
        ----------
        solution:
            The tuple of tuple of tuple of path vertices

        Returns
        -------
        The list of Coordinates objects produced by the pyclipper operations
        """
        if not solution:
            self.raise_warning()
            return None
        coords = []
        if isinstance(solution, PyPolyNode):
            coords = pyclippolytree_to_coordinates(solution)
        else:
            for path in solution:
                c = pyclippath_to_coordinates(path)
                c.close()
                coords.append(c)

        # Sort open coordinates by length
        return sorted(coords, key=lambda x: -x.length)


# =============================================================================
# Offset operations
# =============================================================================


class OffsetOperationManager(PyclipperMixin):
    """
    Abstract base class for offset operations

    Parameters
    ----------
    coordinates:
        The Coordinates upon which to perform the offset operation
    """

    method = NotImplemented
    closed_method = ET_CLOSEDPOLYGON
    open_method = NotImplementedError

    def __init__(self, coordinates: Coordinates):
        self.tool = PyclipperOffset()
        path = coordinates_to_pyclippath(coordinates)
        self._scale = self._calculate_scale(path, coordinates)  # Store scale

        co_method = self.closed_method if coordinates.closed else self.open_method

        self.tool.AddPath(path, self.method, co_method)

    def perform(self, delta: float):
        """
        Perform the offset operation.

        Parameters
        ----------
        delta:
            The value of the offset [m]. Positive for increasing size, negative for
            decreasing
        """
        delta = int(round(delta * self._scale))  # approximation
        solution = self.tool.Execute(delta)
        return self.handle_solution(solution)

    @staticmethod
    def _calculate_scale(path: np.ndarray, coordinates: Coordinates):
        """
        Calculate the pyclipper scaling to integers
        """
        # Find the first non-zero dimension (low number of iterations)
        for i in range(len(path) - 1):
            if path[i][0] != 0:
                return path[i][0] / coordinates.x[i]
            if path[i][1] != 0:
                return path[i][1] / coordinates.z[i]
        return None


class RoundOffset(OffsetOperationManager):
    """
    Offset class for rounded offsets.
    """

    name = "Round Offset"
    method = JT_ROUND
    open_method = ET_OPENROUND


class SquareOffset(OffsetOperationManager):
    """
    Offset class for squared offsets.
    """

    name = "Square Offset"
    method = JT_SQUARE
    open_method = ET_OPENSQUARE


class MiterOffset(OffsetOperationManager):
    """
    Offset class for mitered offsets.
    """

    name = "Miter Offset"
    method = JT_MITER
    open_method = ET_OPENROUND

    def __init__(self, coordinates: Coordinates, miter_limit: float = 2.0):
        super().__init__(coordinates)

        self.tool.MiterLimit = miter_limit


def offset_clipper(
    coordinates: Coordinates,
    delta: float,
    method: str = "square",
    miter_limit: float = 2.0,
) -> Coordinates:
    """
    Carries out an offset operation on the Coordinates using the ClipperLib library.
    Only supports closed Coordinates.

    Parameters
    ----------
    coordinates:
        The Coordinates upon which to perform the offset operation
    delta:
        The value of the offset [m]. Positive for increasing size, negative for
        decreasing
    method:
        The type of offset to perform ['square', 'round', 'miter']
    miter_limit:
        The ratio of delta to use when mitering acute corners. Only used if
        method == 'miter'

    Returns
    -------
    The offset Coordinates result

    Raises
    ------
    GeometryError:
        If the Coordinates are not planar
        If the Coordinates are not closed
    """
    if not coordinates.is_planar:
        raise GeometryError("Cannot offset non-planar coordinates.")

    if not coordinates.closed:
        raise GeometryError("Open Coordinates are not supported by offset_clipper.")

    # Transform coordinates to x-z plane
    coordinates = deepcopy(coordinates)
    com = coordinates.center_of_mass

    t_coordinates = transform_coordinates_to_xz(
        coordinates, -np.array(com), (0.0, 1.0, 0.0)
    )

    if method == "square":
        tool = SquareOffset(t_coordinates)
    elif method == "round":
        bluemira_warn("I don't know why, but this is very slow...")
        tool = RoundOffset(t_coordinates)
    elif method == "miter":
        tool = MiterOffset(t_coordinates, miter_limit=miter_limit)
    else:
        raise GeometryError(
            "Please choose an offset method from:\n round \n square \n miter"
        )

    result = tool.perform(delta)
    if result is None:
        raise GeometryError(
            f"Offset operation with delta={delta} resulted in no geometry."
        )

    if len(result) > 1:
        bluemira_warn(
            f"Offset operation with delta={delta} has produced multiple 'islands'; only"
            " returning the biggest one!"
        )

    result = result[0]

    # Transform offset coordinates back to original plane
    return transform_coordinates_to_original(result, com, coordinates.normal_vector)


def transform_coordinates_to_xz(
    coordinates: Coordinates, base: np.ndarray, direction: np.ndarray
) -> Coordinates:
    """
    Rotate coordinates to the x-z plane.
    """
    coordinates.translate(base)
    if abs(coordinates.normal_vector[1]) == 1.0:  # noqa: PLR2004
        return coordinates

    r = rotation_matrix_v1v2(coordinates.normal_vector, np.array(direction))
    x, y, z = r.T @ coordinates

    return Coordinates({"x": x, "y": y, "z": z})


def transform_coordinates_to_original(
    coordinates: Coordinates, base: np.ndarray, original_normal: np.ndarray
) -> Coordinates:
    """
    Rotate coordinates back to original plane
    """
    r = rotation_matrix_v1v2(coordinates.normal_vector, np.array(original_normal))
    x, y, z = r.T @ coordinates
    coordinates = Coordinates({"x": x, "y": y, "z": z})
    coordinates.translate(base)
    return coordinates
