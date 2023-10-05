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
A collection of tools used for position interpolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple, Union

if TYPE_CHECKING:
    from bluemira.geometry.wire import BluemiraWire

import abc

import numpy as np
from scipy.spatial import ConvexHull

from bluemira.base.constants import EPS
from bluemira.geometry.constants import VERY_BIG
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import slice_shape
from bluemira.utilities.error import PositionerError
from bluemira.utilities.tools import is_num


class XZGeometryInterpolator(abc.ABC):
    """
    Abstract base class for 2-D x-z geometry interpolation to normalised [0, 1] space.

    By convention, normalised x-z space is oriented counter-clockwise w.r.t. [0, 1, 0].

    Parameters
    ----------
    geometry:
        Geometry to interpolate with
    """

    def __init__(self, geometry: BluemiraWire):
        self.geometry = geometry

    def _get_xz_coordinates(self):
        """
        Get discretised x-z coordinates of the geometry.
        """
        coordinates = self.geometry.discretize(
            byedges=True, dl=self.geometry.length / 1000
        )
        coordinates.set_ccw([0, 1, 0])
        return coordinates.xz

    @abc.abstractmethod
    def to_xz(
        self, l_value: Union[float, np.ndarray]
    ) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
        """
        Convert parametric-space 'L' values to physical x-z space.
        """

    @abc.abstractmethod
    def to_L(
        self, x: Union[float, np.ndarray], z: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Convert physical x-z space values to parametric-space 'L' values.
        """

    @abc.abstractproperty
    def dimension(self) -> int:
        """
        The dimension of the parametric space
        """


class PathInterpolator(XZGeometryInterpolator):
    """
    Sets up an x-z path for a point to move along.

    The path is treated as flat in the x-z plane.
    """

    def to_xz(
        self, l_values: Union[float, np.ndarray]
    ) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
        """
        Convert parametric-space 'L' values to physical x-z space.
        """
        l_values = np.clip(l_values, 0.0, 1.0)
        if is_num(l_values):
            return self.geometry.value_at(alpha=l_values)[[0, 2]]

        x, z = np.zeros(len(l_values)), np.zeros(len(l_values))
        for i, lv in enumerate(l_values):
            x[i], z[i] = self.geometry.value_at(alpha=lv)[[0, 2]]

        return x, z

    def to_L(
        self, x: Union[float, np.ndarray], z: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Convert physical x-z space values to parametric-space 'L' values.
        """
        if is_num(x):
            return self.geometry.parameter_at([x, 0, z], tolerance=VERY_BIG)

        l_values = np.zeros(len(x))
        for i, (xi, zi) in enumerate(zip(x, z)):
            l_values[i] = self.geometry.parameter_at([xi, 0, zi], tolerance=VERY_BIG)
        return l_values

    @property
    def dimension(self) -> int:
        """
        Dimension of the parametric space of the PathInterpolator
        """
        return 1


class RegionInterpolator(XZGeometryInterpolator):
    """
    Sets up an x-z region for a point to move within.

    The region is treated as a flat x-z surface.

    The normalisation occurs by cutting the shape in two axes and
    normalising over the cut length within the region.

    Currently this is limited to convex polygons.

    Generalisation to all polygons is possible but unimplemented
    and possibly quite slow when converting from normalised to real coordinates.

    When the point position provided is outside the given region the point will
    be moved to the closest edge of the region.

    The mapping from outside to the edge of the region is not strictly defined.
    The only certainty is that the point will be moved into the region.

    Parameters
    ----------
    geometry:
        Region to interpolate within
    """

    def __init__(self, geometry: BluemiraWire):
        super().__init__(geometry)
        self._check_geometry_feasibility(geometry)
        self.z_min = geometry.bounding_box.z_min
        self.z_max = geometry.bounding_box.z_max

    def _check_geometry_feasibility(self, geometry: BluemiraWire):
        """
        Checks the provided region is convex.

        This is a current limitation of RegionInterpolator
        not providing a 'smooth' interpolation surface.

        Parameters
        ----------
        geometry:
            Region to check

        Raises
        ------
        PositionerError
            When geometry is not a convex
        """
        if not self.geometry.is_closed:
            raise PositionerError("RegionInterpolator can only handle closed wires.")

        xz_coordinates = self._get_xz_coordinates()
        hull = ConvexHull(xz_coordinates.T)
        # Yes, the "area" of a 2-D scipy ConvexHull is its perimeter...
        if not np.allclose(hull.area, geometry.length, atol=EPS):
            raise PositionerError(
                "RegionInterpolator can only handle convex geometries. Perimeter"
                " difference between convex hull and geometry:"
                f" {hull.area - geometry.length}"
            )

    def to_xz(
        self, l_values: Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]
    ) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
        """
        Convert parametric-space 'L' values to physical x-z space.

        Parameters
        ----------
        l_values:
            Coordinates in normalised space

        Returns
        -------
        x:
            x coordinate in real space
        z:
            z coordinate in real space

        Raises
        ------
        GeometryError
            When loop is not a Convex Hull

        """
        l_0, l_1 = l_values
        z = self.z_min + (self.z_max - self.z_min) * l_1

        plane = BluemiraPlane.from_3_points([0, 0, z], [1, 0, z], [0, 1, z])

        intersect = slice_shape(self.geometry, plane)
        if len(intersect) == 1:
            x = intersect[0][0]
        elif len(intersect) == 2:  # noqa: PLR2004
            x_min, x_max = sorted([intersect[0][0], intersect[1][0]])
            x = x_min + (x_max - x_min) * l_0
        else:
            raise PositionerError(
                "Unexpected number of intersections in x-z conversion."
            )

        return x, z

    def to_L(
        self, x: Union[float, np.ndarray], z: Union[float, np.ndarray]
    ) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
        """
        Convert physical x-z space values to parametric-space 'L' values.

        Parameters
        ----------
        x:
            x coordinate in real space
        z:
            z coordinate in real space

        Returns
        -------
        l_1:
            Coordinate 1 in normalised space
        l_2:
            Coordinate 2 in normalised space

        Raises
        ------
        GeometryError
            When loop is not a Convex Hull

        """
        l_1 = (z - self.z_min) / (self.z_max - self.z_min)
        l_1 = np.clip(l_1, 0.0, 1.0)

        plane = BluemiraPlane.from_3_points([x, 0, z], [x + 1, 0, z], [x, 1, z])
        intersect = slice_shape(self.geometry, plane)

        return self._intersect_filter(x, l_1, intersect)

    def _intersect_filter(
        self, x: float, l_1: float, intersect: BluemiraPlane
    ) -> Tuple[float, float]:
        """
        Checks where points are based on number of intersections
        with a plane. Should initially be called with a plane involving z.

        No intersection could mean above 1 edge therefore a plane in xy
        is checked before recalling this function.
        If there is one intersection point we are on an edge (either bottom or top),
        if there is two intersection points we are in the region,
        otherwise the region is not a convex hull.

        Parameters
        ----------
        x:
            x coordinate
        l_1:
            Normalised z coordinate
        intersect:
            A plane through xz

        Returns
        -------
        l_1:
            Coordinate 1 in normalised space
        l_2:
            Coordinate 2 in normalised space

        Raises
        ------
        PositionerError
            When geometry is not a convex
        """
        if intersect is None:
            plane = BluemiraPlane.from_3_points([x, 0, 0], [x + 1, 0, 0], [x, 1, 0])
            intersect = slice_shape(self.geometry, plane)
            l_0, l_1 = self._intersect_filter(
                x, l_1, [False] if intersect is None else intersect
            )
        elif len(intersect) == 2:  # noqa: PLR2004
            x_min, x_max = sorted([intersect[0][0], intersect[1][0]])
            l_0 = np.clip((x - x_min) / (x_max - x_min), 0.0, 1.0)
        elif len(intersect) == 1:
            l_0 = float(l_1 == 1.0)  # noqa: PLR2004
        else:
            raise PositionerError("Unexpected number of intersections in L conversion.")
        return l_0, l_1

    @property
    def dimension(self):
        """
        Dimension of the parametric space of the RegionInterpolator
        """
        return 2


class PositionMapper:
    """
    Positioning tool for use in optimisation

    Parameters
    ----------
    interpolators:
        The ordered list of geometry interpolators
    """

    def __init__(self, interpolators: Dict[str, XZGeometryInterpolator]):
        self.interpolators = interpolators

    def _check_length(self, thing):
        """
        Check that something is the same length as the number of available interpolators.
        """
        if len(thing) != len(self.interpolators):
            raise PositionerError(
                f"Object of length: {len(thing)} not of length {len(self.interpolators)}"
            )

    def _vector_to_list(self, l_values):
        """
        Convert a vector of l_values into a ragged list if necessary
        """
        list_values = []
        for i, interpolator in enumerate(self.interpolators.values()):
            values = l_values[i : i + interpolator.dimension]
            list_values.append(values)
        return list_values

    def to_xz(self, l_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a set of parametric-space values to physical x-z coordinates.

        Parameters
        ----------
        l_values:
            The set of parametric-space values to convert

        Returns
        -------
        x:
            Array of x coordinates
        z:
            Array of z coordinates
        """
        l_values = self._vector_to_list(l_values)
        self._check_length(l_values)
        return np.array(
            [
                tool.to_xz(l_values[i])
                for i, tool in enumerate(self.interpolators.values())
            ]
        ).T

    def to_xz_dict(self, l_values: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Convert a set of parametric space values to physical coordinates in a dictionary
        form.

        Parameters
        ----------
        l_values:
            The set of parametric-space values to convert

        Returns
        -------
        Dictionary of x-z values corresponding to each interpolator
        """
        l_values = self._vector_to_list(l_values)
        self._check_length(l_values)
        xz_dict = {}
        for i, (key, tool) in enumerate(self.interpolators.items()):
            xz_dict[key] = tool.to_xz(l_values[i])
        return xz_dict

    def to_L(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Convert a set of physical x-z coordinates to parametric-space values.

        Parameters
        ----------
        x:
            The x coordinates to convert
        z:
            The z coordinates to convert

        Returns
        -------
        l_values:
            The set of parametric-space values
        """
        self._check_length(x)
        self._check_length(z)
        l_values = np.zeros(self.dimension)
        for i, tool in enumerate(self.interpolators.values()):
            l_values[i : i + tool.dimension] = tool.to_L(x[i], z[i])
        return np.array(l_values)

    @property
    def dimension(self) -> int:
        """
        The total dimension of the parametric space
        """
        return sum([interp.dimension for interp in self.interpolators.values()])
