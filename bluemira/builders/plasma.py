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
Plasma builder.
"""

from typing import Dict

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.designer import Designer
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_circle, revolve_shape
from bluemira.geometry.wire import BluemiraWire


class Plasma:
    def __init__(self, component_tree: Component):
        self._component_tree = component_tree

    def component(self) -> Component:
        """Return the plasma component tree."""
        return self._component_tree

    def lcfs(self) -> BluemiraWire:
        """Return a wire representing the last-closed flux surface."""
        return (
            self._component_tree.get_component("xz")
            .get_component(PlasmaBuilder.LCFS)
            .shape.boundary[0]
        )


class PlasmaBuilder(Builder):
    """
    Builder for a poloidally symmetric plasma.
    """

    LCFS = "LCFS"

    # This builder has no parameters
    param_cls = None

    def __init__(
        self,
        build_config: Dict,
        designer: Designer[BluemiraWire],
    ):
        super().__init__(None, build_config, designer)

    def build(self) -> Plasma:
        xz_lcfs = self.designer.run()
        component = self._build_component_tree(xz_lcfs)
        return Plasma(component)

    def build_xz(self, lcfs: BluemiraWire) -> PhysicalComponent:
        face = BluemiraFace(lcfs, self.name)
        component = PhysicalComponent(self.LCFS, face)
        component.plot_options.face_options["color"] = BLUE_PALETTE["PL"]
        return component

    def build_xy(self, lcfs: BluemiraWire) -> PhysicalComponent:
        inner = make_circle(lcfs.bounding_box.x_min, axis=[0, 1, 0])
        outer = make_circle(lcfs.bounding_box.x_max, axis=[0, 1, 0])
        face = BluemiraFace([outer, inner], self.name)
        component = PhysicalComponent(self.LCFS, face)
        component.plot_options.face_options["color"] = BLUE_PALETTE["PL"]
        return component

    def build_xyz(self, lcfs: BluemiraWire, degree: float = 360.0) -> PhysicalComponent:
        shell = revolve_shape(lcfs, direction=(0, 0, 1), degree=degree)
        component = PhysicalComponent(self.LCFS, shell)
        component.display_cad_options.color = BLUE_PALETTE["PL"]
        return component

    def _build_component_tree(self, lcfs: BluemiraWire) -> Component:
        component = Component(self.name)
        component.add_child(Component("xz", children=[self.build_xz(lcfs)]))
        component.add_child(Component("xy", children=[self.build_xy(lcfs)]))
        component.add_child(Component("xyz", children=[self.build_xyz(lcfs)]))
        return component
