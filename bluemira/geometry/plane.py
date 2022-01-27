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
Wrapper for FreeCAD Plane (Placement) objects
"""

from __future__ import annotations

import numpy as np

import bluemira.codes._freecadapi as cadapi
from bluemira.geometry.error import GeometryError


class BluemiraPlane:
    """
    Bluemira Plane class.

    Parameters
    ----------
    base: Iterable
        Base vector of the plane
    axis: Iterable
        Axis vector of the plane
    angle: float
        Angle of the plane
    label: str
        Label of the plane

    Notes
    -----
    The BluemiraPlane wraps a Placement, and not a Plane. The reasons for this will
    become clear in future. Placements and Planes should be interchangeable.

    Angle is rotation around the axis which is taken from the base.
    Usually the shape has a starting point which is not its centre
    """

    def __init__(
        self, base=[0.0, 0.0, 0.0], axis=[0.0, 0.0, 1.0], angle=0.0, label: str = ""
    ):
        self._shape = cadapi.make_plane(base, axis, angle)
        self.label = label

    @classmethod
    def from_3_points(cls, point_1, point_2, point_3, label: str = ""):
        """
        Instantiate a BluemiraPlane from three points.

        Parameters
        ----------
        point_1: Iterable
            First point
        point_2: Iterable
            Second Point
        point_3: Iterable
            Third point
        label: str
            Label of the plane
        """
        p1 = np.array(point_1)
        p2 = np.array(point_2)
        p3 = np.array(point_3)
        v1, v2 = p3 - p1, p2 - p1
        v3 = np.cross(v2, v1)
        if np.all(v3 == 0):
            raise GeometryError("Cannot make a BluemiraPlane from co-linear points.")

        normal = v3 / np.sqrt(v3.dot(v3))
        return cls(point_1, normal, 0.0, label=label)

    @classmethod
    def from_matrix(cls, matrix, label=""):
        """
        Instantiate a BluemiraPlane from a 4 x 4 matrix

        Parameters
        ----------
        matrix: np.ndarray
            4 x 4 matrix from which to make the placement
        label: str
            Label of the plane
        """
        obj = cls.__new__(cls)
        for i in range(3):
            row = matrix[i, :3]
            matrix[i, :3] = row / np.linalg.norm(row)
        matrix[-1, :] = [0, 0, 0, 1]
        obj._shape = cadapi.make_plane_from_matrix(matrix)
        obj.label = label
        return obj

    @property
    def base(self):
        """Plane's base vector"""
        return cadapi.vector_to_list(self._shape.Base)

    @base.setter
    def base(self, value):
        """
        Set a new plane base

        Parameters
        ----------
        value: Iterable
        """
        self._shape.Base = cadapi.Base.Vector(value)

    @property
    def axis(self):
        """Plane's rotation matrix"""
        return self._shape.Rotation.Axis

    @axis.setter
    def axis(self, value):
        """
        Set a new plane axis

        Parameters
        ----------
        value: Iterable
        """
        self._shape.Axis = cadapi.Base.Vector(value)

    @property
    def angle(self):
        """Plane's angle"""
        return np.rad2deg(self._shape.Rotation.Angle)

    @angle.setter
    def angle(self, value):
        """
        Set a new plane angle

        Parameters
        ----------
        value: float
            angle value in degree
        """
        self._shape.Angle = value

    def to_matrix(self):
        """Returns a matrix (quaternion) representing the Plane's transformation"""
        return np.array(self._shape.Matrix.A).reshape(4, 4)

    def move(self, vector):
        """Moves the Plane along the given vector"""
        cadapi.move_plane(self._shape, vector)

    def __repr__(self):  # noqa D105
        new = []
        new.append(f"([{type(self).__name__}] = Label: {self.label}")
        new.append(f" base: {self.base}")
        new.append(f" axis: {self.axis}")
        new.append(f" angle: {self.angle}")
        new.append(")")
        return ", ".join(new)

    def copy(self, label=None):
        """Make a copy of the BluemiraPlane"""
        plane_copy = BluemiraPlane(self.base, self.axis, self.angle)
        if label is not None:
            plane_copy.label = label
        else:
            plane_copy.label = self.label
        return plane_copy

    def deepcopy(self, label=None):
        """Make a deepcopy of the BluemiraPlane"""
        return self.copy()
