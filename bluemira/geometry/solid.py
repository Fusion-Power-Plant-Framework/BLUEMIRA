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

# import from bluemira
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.error import DisjointedSolidError, GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.shell import BluemiraShell
from bluemira.geometry.wire import BluemiraWire

__all__ = ["BluemiraSolid"]


class BluemiraSolid(BluemiraGeo):
    """
    Bluemira Solid class.

    Parameters
    ----------
    boundary:
        List of shells from  which to make the BluemiraSolid
    label:
        Label to assign to the solid
    """

    def __init__(self, boundary: List[BluemiraShell], label: str = ""):
        boundary_classes = [BluemiraShell]
        super().__init__(boundary, label, boundary_classes)

    def _create_solid(self, check_reverse: bool = True):
        """Creation of the solid"""
        new_shell = self.boundary[0]._create_shell(check_reverse=False)
        solid = cadapi.apiSolid(new_shell)

        if len(self.boundary) > 1:
            shell_holes = [cadapi.apiSolid(s.shape) for s in self.boundary[1:]]
            solid = solid.cut(shell_holes)
            if len(solid.Solids) == 1:
                solid = solid.Solids[0]
            else:
                raise DisjointedSolidError("Disjointed solids are not accepted.")

        if check_reverse:
            return self._check_reverse(cadapi.apiSolid(solid))
        return solid

    def _create_shape(self):
        """Part.Solid: shape of the object as a single solid"""
        return self._create_solid()

    @classmethod
    def _create(cls, obj: cadapi.apiSolid, label: str = ""):
        if isinstance(obj, cadapi.apiSolid):
            if len(obj.Solids) > 1:
                raise DisjointedSolidError("Disjointed solids are not accepted.")

            if not obj.isValid():
                # cadapi.save_as_STP(obj, "object_not_valid")
                raise GeometryError(f"Solid {obj} is not valid.")

            bm_shells = [BluemiraShell._create(shell) for shell in obj.Shells]

            # create an empty BluemiraSolid
            bmsolid = cls(None, label=label)
            # assign shape, boundary, and orientation
            bmsolid._set_shape(obj)
            bmsolid._boundary = bm_shells
            bmsolid._orientation = obj.Orientation
            return bmsolid

        raise TypeError(
            f"Only Part.Solid objects can be used to create a {cls} instance"
        )

    @property
    def vertexes(self) -> Coordinates:
        """
        The vertexes of the solid.
        """
        return Coordinates(cadapi.vertexes(self.shape))

    @property
    def edges(self) -> Tuple[BluemiraWire]:
        """
        The edges of the solid.
        """
        return tuple([BluemiraWire(cadapi.apiWire(o)) for o in cadapi.edges(self.shape)])

    @property
    def wires(self) -> Tuple[BluemiraWire]:
        """
        The wires of the solid.
        """
        return tuple([BluemiraWire(o) for o in cadapi.wires(self.shape)])

    @property
    def faces(self) -> Tuple[BluemiraFace]:
        """
        The faces of the solid.
        """
        return tuple([BluemiraFace._create(o) for o in cadapi.faces(self.shape)])

    @property
    def shells(self) -> Tuple[BluemiraShell]:
        """
        The shells of the solid.
        """
        return tuple([BluemiraShell._create(o) for o in cadapi.shells(self.shape)])

    @property
    def solids(self) -> Tuple[BluemiraSolid]:
        """
        The solids of the solid. By definition a tuple of itself.
        """
        return (self,)
