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
Wrapper for FreeCAD Plane objects
"""

from __future__ import annotations

import bluemira.geometry._freecadapi as _freecadapi


class BluemiraPlane:
    """Bluemira Plane class."""

    def __init__(
        self, base=[0.0, 0.0, 0.0], axis=[0.0, 0.0, 1.0], angle=0.0, label: str = ""
    ):
        self._shape = _freecadapi.make_plane(base, axis, angle)
        self.label = label

    @property
    def base(self):
        """Plane's base vector"""
        return _freecadapi.vector_to_list(self._shape.Base)

    @base.setter
    def base(self, value):
        """
        Set a new plane base

        Parameters
        ----------
        value: Iterable
        """
        self._shape.Base = _freecadapi.Base.Vector(value)

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
        self._shape.Axis = _freecadapi.Base.Vector(value)

    @property
    def angle(self):
        """Plane's rotation matrix"""
        return self._shape.Rotation.Angle

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
        return self._shape.toMatrix()

    def move(self, vector):
        """Moves the Plane along the given vector"""
        _freecadapi.move_plane(self._shape, vector)
