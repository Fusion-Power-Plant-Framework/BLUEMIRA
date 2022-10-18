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
Wrapper for FreeCAD Part.Face objects
"""

from __future__ import annotations

# import from freecad
import bluemira.codes._freecadapi as cadapi

# import from bluemira
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.error import DisjointedSolid
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.shell import BluemiraShell
from bluemira.geometry.wire import BluemiraWire

__all__ = ["BluemiraSolid"]


class BluemiraSolid(BluemiraGeo):
    """Bluemira Solid class."""

    def __init__(self, boundary, label: str = ""):
        boundary_classes = [BluemiraShell]
        super().__init__(boundary, label, boundary_classes)

    def _create_solid(self, check_reverse=True):
        """Creation of the solid"""
        new_shell = self.boundary[0]._create_shell(check_reverse=False)
        solid = cadapi.apiSolid(new_shell)

        if len(self.boundary) > 1:
            shell_holes = [cadapi.apiSolid(s._shape) for s in self.boundary[1:]]
            solid = solid.cut(shell_holes)
            if len(solid.Solids) == 1:
                solid = solid.Solids[0]
            else:
                raise DisjointedSolid("Disjointed solids are not accepted.")

        if check_reverse:
            return self._check_reverse(cadapi.apiSolid(solid))
        else:
            return solid

    def create_shape(self):
        """Part.Solid: shape of the object as a single solid"""
        return self._create_solid()

    @classmethod
    def _create(cls, obj: cadapi.apiSolid, label=""):
        if isinstance(obj, cadapi.apiSolid):
            orientation = obj.Orientation

            if len(obj.Solids) > 1:
                raise DisjointedSolid("Disjointed solids are not accepted.")

            bm_shells = []
            for shell in obj.Shells:
                bm_shells.append(BluemiraShell._create(shell))

            bmsolid = cls(bm_shells, label=label)
            bmsolid._orientation = orientation
            return bmsolid

        raise TypeError(
            f"Only Part.Solid objects can be used to create a {cls} instance"
        )

    @property
    def vertexes(self):
        """
        The vertexes of the solid.
        """
        return Coordinates(cadapi.vertexes(self.shape))

    @property
    def edges(self):
        """
        The edges of the solid.
        """
        return [BluemiraWire(cadapi.apiWire(o)) for o in cadapi.edges(self.shape)]

    @property
    def wires(self):
        """
        The wires of the solid.
        """
        return [BluemiraWire(o) for o in cadapi.wires(self.shape)]

    @property
    def faces(self):
        """
        The faces of the solid.
        """
        return [BluemiraFace(o) for o in cadapi.faces(self.shape)]

    @property
    def shells(self):
        """
        The shells of the solid.
        """
        return [BluemiraShell(o) for o in cadapi.shells(self.shape)]

    @property
    def solids(self):
        """
        The solids of the solid. By definition a list of itself.
        """
        return [self]

    @property
    def shape_boundary(self):
        """
        The boundaries of the solid.
        """
        return self.shells
