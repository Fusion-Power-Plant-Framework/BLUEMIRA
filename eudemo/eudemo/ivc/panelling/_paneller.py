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
from typing import Optional, Union

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from eudemo.ivc.panelling._pivot_string import make_pivoted_string


class Paneller:
    """
    Provides functions for generating panelling along the outside of a boundary.

    Parameters
    ----------
    boundary_points
        The points defining the boundary along which to build the panels.
        This should have shape (2, N), where N is the number of points.
    """

    def __init__(
        self,
        boundary_points: np.ndarray,
        max_angle: float,
        dx_min: float,
        fix_num_panels: Optional[int] = None,
    ):
        self.max_angle = max_angle
        self.dx_min = dx_min
        self.boundary = LengthNormBoundary(boundary_points)

        if fix_num_panels:
            self.x0 = np.linspace(0, 1, fix_num_panels)[1:-1]
        else:
            # Build the initial guess of our panels, these points are the
            # coordinates of where the panels tangent the boundary
            _, idx = make_pivoted_string(
                boundary_points.T,
                max_angle=max_angle,
                dx_min=dx_min,
            )
            self.x0: np.ndarray = self.boundary.length_norm[idx][1:-1]

    @property
    def n_panels(self) -> int:
        """The number of panels defined by this paneller."""
        return len(self.x0) + 2

    def joints(self, dists: np.ndarray) -> np.ndarray:
        """
        Calculate panel joint coordinates from panel-boundary tangent points.

        Parameters
        ----------
        dists
            The normalised distances along the boundary at which there
            are panel-boundary tangent points.
        """
        # Add the start and end panel joints at distances 0 & 1
        dists = np.hstack((0, dists, 1))
        points = np.vstack((self.boundary.x(dists), self.boundary.z(dists)))
        tangents = np.vstack(
            (self.boundary.x_tangent(dists), self.boundary.z_tangent(dists))
        )
        # This could potentially be vectorized, if we need a speed up.
        joints = np.zeros((2, len(dists) + 1))
        joints[:, 0] = points[:, 0]
        joints[:, -1] = points[:, -1]
        for i in range(joints.shape[1] - 2):
            joints[:, i + 1] = vector_intersect(
                points[:, i],
                points[:, i] + tangents[:, i],
                points[:, i + 1],
                points[:, i + 1] + tangents[:, i + 1],
            )
        return joints

    # Typically, in a panelling optimisation loop, we'll call, at least,
    # 'length' and 'angles', which both call out to 'joints'.
    # For a potential optimisation we could cache the result of 'joints'.
    def length(self, dists: np.ndarray) -> float:
        """The cumulative length of the panels."""
        return self.panel_lengths(dists).sum()

    def angles(self, dists: np.ndarray) -> np.ndarray:
        """
        Return the angles of rotation between each set of adjacent panels.

        Note that this is the tail-tail angle between the panel vectors,
        not the head-tail angles.
        """
        joints = self.joints(dists)
        line_vectors = np.diff(joints, axis=1)
        magnitudes = np.linalg.norm(line_vectors, axis=0)
        dots = (line_vectors[:, :-1] * line_vectors[:, 1:]).sum(axis=0)
        dots /= magnitudes[:-1] * magnitudes[1:]
        return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)), out=dots)

    def panel_lengths(self, dists: np.ndarray) -> np.ndarray:
        """Return the lengths of each panel."""
        panel_vecs = np.diff(self.joints(dists))
        return np.hypot(panel_vecs[0], panel_vecs[1])


class LengthNormBoundary:
    """Class to represent a wire interpolated over the normalised distance along it."""

    def __init__(self, boundary_points: np.ndarray):
        self.length_norm = norm_lengths(boundary_points)
        self._x_spline = InterpolatedUnivariateSpline(
            self.length_norm, boundary_points[0]
        )
        self._z_spline = InterpolatedUnivariateSpline(
            self.length_norm, boundary_points[1]
        )

        self.tangent_norm = norm_tangents(boundary_points)
        self._x_tangent_spline = InterpolatedUnivariateSpline(
            self.length_norm, self.tangent_norm[0]
        )
        self._z_tangent_spline = InterpolatedUnivariateSpline(
            self.length_norm, self.tangent_norm[1]
        )

    def x(self, dist: Union[float, np.ndarray]) -> np.ndarray:
        """Find x at the given normalised distance along the boundary."""
        return self._x_spline(dist)

    def z(self, dist: Union[float, np.ndarray]) -> np.ndarray:
        """Find z at the given normalised distance along the boundary."""
        return self._z_spline(dist)

    def x_tangent(self, dist: Union[float, np.ndarray]) -> np.ndarray:
        """Find x at the tangent vector a given distance along the boundary."""
        return self._x_tangent_spline(dist)

    def z_tangent(self, dist: Union[float, np.ndarray]) -> np.ndarray:
        """Find z at the tangent vector a given distance along the boundary."""
        return self._z_tangent_spline(dist)


def norm_lengths(points: np.ndarray) -> np.ndarray:
    """
    Calculate the cumulative normalized lengths between each 2D point.

    Parameters
    ----------
    points
        A numpy array of points, shape should be (2, N).
    """
    dists = np.diff(points, axis=1)
    sq_dists = np.square(dists, out=dists)
    summed_dists = np.sum(sq_dists, axis=0)
    sqrt_dists = np.sqrt(summed_dists, out=summed_dists)
    cumulative_sum = np.cumsum(summed_dists, out=sqrt_dists)
    return np.hstack((0, cumulative_sum / cumulative_sum[-1]))


def norm_tangents(points: np.ndarray) -> np.ndarray:
    """
    Calculate the normalised tangent vector at each of the given points.

    Parameters
    ----------
    points
        Array of coordinates. This must have shape (2, N), where N is
        the number of points.

    Returns
    -------
    tangents
        The normalised vector of tangents.
    """
    grad = np.gradient(points, axis=1)
    magnitudes = np.hypot(grad[0], grad[1])
    return np.divide(grad, magnitudes, out=grad)


def vector_intersect(p1, p2, p3, p4):
    """
    Find the point of intersection between two vectors defined by the given points.

    Parameters
    ----------
    p1: np.ndarray(2)
        The first point on the first vector
    p2: np.ndarray(2)
        The second point on the first vector
    p3: np.ndarray(2)
        The first point on the second vector
    p4: np.ndarray(2)
        The second point on the second vector

    Returns
    -------
    p_inter: np.ndarray(2)
        The point of the intersection between the two vectors
    """
    da = p2 - p1
    db = p4 - p3

    if np.isclose(np.cross(da, db), 0):  # vectors parallel
        point = p2
    else:
        dp = p1 - p3
        dap = normal_vector(da)
        denom = np.dot(dap, db)
        num = np.dot(dap, dp)
        point = num / denom.astype(float) * db + p3
    return point


def normal_vector(side_vectors):
    """
    Anti-clockwise

    Parameters
    ----------
    side_vectors: np.array(N, 2)
        The side vectors of a polygon

    Returns
    -------
    a: np.array(2, N)
        The array of 2-D normal vectors of each side of a polygon
    """
    a = -np.array([-side_vectors[1], side_vectors[0]]) / np.sqrt(
        side_vectors[0] ** 2 + side_vectors[1] ** 2
    )
    a[np.isnan(a)] = 0
    return a
