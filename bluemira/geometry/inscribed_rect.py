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
Function to find inscribed rectangle.

In contained file because loop module imports geomtools and geombase modules
"""
from copy import deepcopy
from typing import List, Tuple

import numpy as np
from scipy.spatial.distance import pdist

from bluemira.geometry.coordinates import (
    Coordinates,
    coords_plane_intersect,
    get_intersect,
    in_polygon,
)
from bluemira.geometry.plane import BluemiraPlane

__all__ = ["inscribed_rect_in_poly"]


def inscribed_rect_in_poly(
    x_poly: np.ndarray,
    z_poly: np.ndarray,
    x_point: float,
    z_point: float,
    aspectratio: float = 1.0,
    *,
    convex: bool = True,
    rtol: float = 1e-06,
    atol: float = 1e-08,
) -> Tuple[float, float]:
    """
    Find largest inscribed rectangle in a given polygon.

    Parameters
    ----------
    x_poly:
        x coordinates of the polygon
    z_poly:
        z coordinates of the polygon
    x_point:
        x coordinate of the centroid of the
    z_point:
        z coordinate of the centroid of the rectangle
    aspectratio:
        aspect ratio of rectangle
    convex:
        treat the loop as convex default:True
    rtol:
        The relative tolerance parameter (see Notes)
    atol:
        The absolute tolerance parameter (see Notes)

    Returns
    -------
    dx:
        half width of inscribed rectangle
    dz:
        half height of inscribed rectangle

    Notes
    -----
    See notes of https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
    for an explanation of relative and absolute tolerances.
    The tolerances only affects non convex loops in certain complex situations.
    Setting either value to a very small value could cause the function to hang.
    """
    coordinates = Coordinates({"x": x_poly, "z": z_poly})
    if not coordinates.closed:
        coordinates.close()

    if not in_polygon(x_point, z_point, coordinates.xz.T, include_edges=False):
        return 0, 0

    x, z = x_point, z_point

    angle_r = np.rad2deg(np.arctan(1 / aspectratio))

    # Set up "Union Jack" intersection Planes
    xx = Coordinates([[x, 0, z], [x + 1, 0, z], [x, 1, z], [0, 0, 0]])
    zz = Coordinates([[x, 0, z], [x, 0, z + 1], [x, 1, z], [0, 0, 0]])

    xo, rot_p = [x, 0, z], [0, 1, 0]

    xx_plane = BluemiraPlane.from_3_points(*xx.points[:3])
    zz_plane = BluemiraPlane.from_3_points(*zz.points[:3])

    xx_rot = deepcopy(xx)
    xx_rot.rotate(base=xo, direction=rot_p, degree=angle_r)
    xz_plane = BluemiraPlane.from_3_points(*xx_rot.points[:3])

    zz_rot = deepcopy(xx)
    zz_rot.rotate(base=xo, direction=rot_p, degree=-angle_r)
    zx_plane = BluemiraPlane.from_3_points(*zz_rot.points[:3])

    # Set up distance calculation
    getdxdz = _GetDxDz(
        coordinates,
        [x_point, z_point],
        aspectratio,
        convex,
        [xx_plane, zz_plane, xz_plane, zx_plane],
    )

    dx, dz = getdxdz()

    if convex or all(
        not i.size for i in get_intersect(_rect(x, z, dx, dz).xz, coordinates.xz)
    ):
        return dx, dz

    left, right = 0, dx

    while not np.isclose(right, left, rtol=rtol, atol=atol):
        dx = (left + right) / 2
        dz = dx / aspectratio

        if all(
            not i.size for i in get_intersect(_rect(x, z, dx, dz).xz, coordinates.xz)
        ):
            left = dx  # increase the dx
        else:
            right = dx  # decrease the dx
    return dx, dz


class _GetDxDz:
    """
    Calculate dx and dz of nearest edge intersection.

    Parameters
    ----------
    coords:
        Region coordinates
    point:
        central point of rectangle
    aspectratio:
        Aspect ratio of rectangle
    convex:
        convex region boolean
    planes:
        list of intersection planes
    """

    def __init__(
        self,
        coords: Coordinates,
        point: Tuple[float, float],
        aspectratio: float,
        convex: bool,
        planes: List[BluemiraPlane],
    ):
        self.vec_arr_x = np.zeros((9, 2))
        self.vec_arr_x[0] = point

        self.point = point
        self.coords = coords
        self.planes = planes

        self.n_p = 2 * len(planes)

        self.aspectratio = aspectratio

        self.elements = np.arange(1, self.n_p + 1)[0::2]

        self.check = self.approx if convex else self.precise

    def approx(self, n: int, lpi: np.ndarray):
        """
        Approximate nearest intersection (for convex shapes).
        """
        self.vec_arr_x[n : n + 2] = lpi[0, [0, 2]], lpi[-1, [0, 2]]

    def precise(self, n, lpi):
        """
        Precise nearest intersection.
        """
        for i2, i in enumerate(lpi[:-1], start=1):
            int_s1, int_s2 = i[[0, 2]], lpi[i2, [0, 2]]
            # if point between intersection points
            if np.allclose(
                np.linalg.norm(int_s1 - self.point)
                + np.linalg.norm(self.point - int_s2),
                np.linalg.norm(int_s1 - int_s2),
            ):
                self.vec_arr_x[n : n + 2] = int_s1, int_s2
                break

    def __call__(self) -> Tuple[float, float]:
        """
        Get dx and dz.

        Returns
        -------
        dx:
            maximum width/2 of rectangle
        dz:
            maximum height/2 of rectangle
        """
        for n, plane in zip(self.elements, self.planes):
            lpi = coords_plane_intersect(self.coords, plane)
            self.check(n, lpi)

        self.vec_arr_z = self.vec_arr_x.copy()
        self.vec_arr_x[:, 1] = self.point[1]
        self.vec_arr_z[:, 0] = self.point[0]

        dist = np.array(
            [
                pdist(self.vec_arr_x, "euclidean")[: self.n_p],
                pdist(self.vec_arr_z, "euclidean")[: self.n_p],
            ]
        )

        # Find minimum distance
        # this is loopy indexing TODO cleanup
        # a1= [xz], a2= intersections, a3=staight/diagonal
        dist = dist.reshape((2, -1, 2), order="F")
        amin = np.argmin(np.sum(dist, axis=0), axis=0)

        (dx1, dz1), (dx2, dz2) = dist[:, amin[0], 0], dist[:, amin[1], 1]
        if dx1 == 0:
            dx1 = dz1 * self.aspectratio
        elif dz1 == 0:
            dz1 = dx1 / self.aspectratio

        return (dx2, dz2) if dx2 < dx1 or dz2 < dz1 else (dx1, dz1)


def _rect(x: float, z: float, dx: float, dz: float) -> Coordinates:
    """
    Helper function to create a rectangular loop at a given point.

    Parameters
    ----------
    x: float
        central x coordinate
    z: float
        central z coordinate
    dx: float
        width/2 of rectangle
    dz: float
        height/2 of rectangle

    Returns
    -------
    Rectangular closed set of coordinates
    """
    return Coordinates(
        {
            "x": x + np.array([-dx, dx, dx, -dx, -dx]),
            "z": z + np.array([-dz, -dz, dz, dz, -dz]),
        }
    )
