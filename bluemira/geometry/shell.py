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
Wrapper for FreeCAD Part.Face objects
"""

from __future__ import annotations

from typing import List, Tuple

# import from freecad
import bluemira.codes._freecadapi as cadapi
from bluemira.geometry.base import BluemiraGeo

# import from bluemira
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.wire import BluemiraWire

__all__ = ["BluemiraShell"]


class BluemiraShell(BluemiraGeo):
    """
    Bluemira Shell class.

    Parameters
    ----------
    boundary:
        List of faces from which to make the BluemiraShell
    label:
        Label to assign to the shell
    """

    def __init__(self, boundary: List[BluemiraFace], label: str = ""):
        boundary_classes = [BluemiraFace]
        super().__init__(boundary, label, boundary_classes)

    def _create_shell(self, check_reverse: bool = True):
        """Creation of the shell"""
        faces = [f._create_face(check_reverse=True) for f in self.boundary]
        shell = cadapi.apiShell(faces)

        if check_reverse:
            return self._check_reverse(shell)
        return shell

    def _create_shape(self):
        """Part.Shell: shape of the object as a primitive shell"""
        return self._create_shell()

    @classmethod
    def _create(cls, obj: cadapi.apiShell, label=""):
        if isinstance(obj, cadapi.apiShell):
            faces = obj.Faces
            bmfaces = [BluemiraFace._create(face) for face in faces]

            bmshell = BluemiraShell(None, label=label)
            bmshell._set_shape(obj)
            bmshell._set_boundary(bmfaces, False)
            bmshell._orientation = obj.Orientation
            return bmshell
        raise TypeError(
            f"Only Part.Shell objects can be used to create a {cls} instance"
        )

    @property
    def vertexes(self) -> Coordinates:
        """
        The vertexes of the shell.
        """
        return Coordinates(cadapi.vertexes(self.shape))

    @property
    def edges(self) -> Tuple[BluemiraWire]:
        """
        The edges of the shell.
        """
        return tuple([BluemiraWire(cadapi.apiWire(o)) for o in cadapi.edges(self.shape)])

    @property
    def wires(self) -> Tuple[BluemiraWire]:
        """
        The wires of the shell.
        """
        return tuple([BluemiraWire(o) for o in cadapi.wires(self.shape)])

    @property
    def faces(self) -> Tuple[BluemiraFace]:
        """
        The faces of the shell.
        """
        return tuple([BluemiraFace._create(o) for o in cadapi.faces(self.shape)])

    @property
    def shells(self) -> tuple:
        """
        The shells of the shell. By definition a tuple of itself.
        """
        return (self,)

    @property
    def solids(self) -> tuple:
        """
        The solids of the shell. By definition an empty tuple.
        """
        return ()
