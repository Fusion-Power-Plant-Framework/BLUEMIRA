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
EUDEMO component manager classes
"""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Iterable, List, Tuple

if TYPE_CHECKING:
    from bluemira.geometry.solid import BluemiraSolid
    from bluemira.geometry.wire import BluemiraWire

from bluemira.base.builder import ComponentManager
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.tools import _timing
from bluemira.builders.cryostat import CryostatBuilder
from bluemira.builders.radiation_shield import RadiationShieldBuilder
from bluemira.builders.thermal_shield import CryostatTSBuilder, VVTSBuilder
from bluemira.builders.tools import apply_component_display_options
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.tools import boolean_cut, boolean_fuse
from bluemira.materials import Void
from eudemo.maintenance.duct_connection import pipe_pipe_join
from eudemo.tools import make_2d_view_components


class VacuumVesselThermalShield(ComponentManager):
    """
    Wrapper around a VVTS component tree.
    """

    def xz_boundary(self) -> BluemiraWire:
        """Return a wire representing the VVTS poloidal silhouette."""
        return (
            self.component()
            .get_component("xz")
            .get_component(VVTSBuilder.VVTS)
            .shape.boundary[0]
        )


class CryostatThermalShield(ComponentManager):
    """
    Wrapper around a VVTS component tree.
    """

    def xz_boundary(self) -> BluemiraWire:
        """Return a wire representing the VVTS poloidal silhouette."""
        return (
            self.component()
            .get_component("xz")
            .get_component(CryostatTSBuilder.CRYO_TS)
            .shape.boundary[0]
        )


class OrphanerMixin:
    """
    Component orphanage mixin class
    """

    @staticmethod
    def _orphan_old_components(components):
        """
        Orphan and delete previous components
        """
        if not isinstance(components, Iterable):
            components = [components]
        for comp in components:
            for view in ["xyz", "xz", "xy"]:
                comp_view = comp.get_component(view)
                comp_view.parent = None
                del comp_view


class PlugManagerMixin(OrphanerMixin):
    """
    Mixin class for miscellaneous plug component integration utilities.
    """

    @staticmethod
    def _make_2d_views(parent, solid_comp, plug_comps, angle, color, plug_color):
        for view in ["xz", "xy"]:
            solid_comps = make_2d_view_components(
                view, azimuthal_angle=angle, components=[solid_comp]
            )[0]
            for solid in solid_comps:
                apply_component_display_options(solid, color=color)

            view_plug_comps = make_2d_view_components(
                view, azimuthal_angle=angle, components=plug_comps
            )
            view_plug_comps = [item for row in view_plug_comps for item in row]

            for plug in view_plug_comps:
                apply_component_display_options(plug, plug_color)

            view_comps = solid_comps + view_plug_comps

            Component(view, children=view_comps, parent=parent)

    def _add_plugs(
        self,
        plug_component: Component,
        n_TF: int,
        name: str,
        color_list: List[Tuple[float, float, float]],
    ):
        comp = plug_component.get_component("xyz")
        void_shapes = []
        plugs = []
        for child in comp.children:
            if "voidspace" in child.name:
                void_shapes.append(child.shape)
            else:
                plugs.append(child)

        component = self.component()
        xyz_shape = (
            component.get_component("xyz")
            .get_component("Sector 1")
            .get_component(name)
            .shape
        )

        xyz_shape = boolean_cut(xyz_shape, void_shapes)[0]
        xyz_comp = PhysicalComponent(name, xyz_shape)
        apply_component_display_options(xyz_comp, color=color_list[0])
        self._orphan_old_components(component)

        new_components = [xyz_comp, *plugs]

        Component(
            "xyz",
            parent=component,
            children=[Component("Sector 1", children=new_components)],
        )

        angle = 180 / n_TF
        self._make_2d_views(
            component,
            xyz_comp,
            plugs,
            angle,
            color_list[0],
            color_list[1],
        )


class PortManagerMixin(OrphanerMixin, abc.ABC):
    """
    Mixin class for miscellaneous port component integration utilities.
    """

    def __init__(self, *args, verbose: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_ports = _timing(
            self.add_ports,
            "Built in",
            f"Adding ports to {type(self).__name__}",
            debug_info_str=not verbose,
        )

    @staticmethod
    def _make_2d_views(
        parent, solid_comp, void_comp, angle, color, void_color=(0, 0, 0)
    ):
        for view in ["xz", "xy"]:
            solid_comps, void_comps = make_2d_view_components(
                view, azimuthal_angle=angle, components=[solid_comp, void_comp]
            )
            for solid in solid_comps:
                apply_component_display_options(solid, color=color)
            for void in void_comps:
                apply_component_display_options(void, color=void_color)
            view_comps = solid_comps + void_comps

            Component(view, children=view_comps, parent=parent)

    @abc.abstractmethod
    def add_ports(self, ports: List[Component], n_TF: int):
        """Add ports to Component"""


class ThermalShield(PortManagerMixin, ComponentManager):
    """
    Wrapper around a Thermal Shield component tree.
    """

    def vacuum_vessel_thermal_shield(self) -> Component:
        """
        Get the vacuum vessel thermal shield component
        """
        return self.component().get_component("VVTS")

    def cryostat_thermal_shield(self) -> Component:
        """
        Get the cryostat thermal shield component
        """
        return self.component().get_component("CryostatTS")

    @staticmethod
    def _join_ports_to_vvts(
        ports: List[Component],
        vvts_target_shape: BluemiraSolid,
        vvts_target_void: BluemiraSolid,
    ) -> Tuple[BluemiraSolid, BluemiraSolid, List[BluemiraSolid]]:
        if isinstance(ports, Component):
            ports = [ports]

        tool_voids = []
        new_shape_pieces = []
        for port in ports:
            port_xyz = port.get_component("xyz")
            tool_shape = port_xyz.get_component(port.name).shape
            tool_void = port_xyz.get_component(port.name + " voidspace").shape
            tool_voids.append(tool_void)
            result_pieces = pipe_pipe_join(
                vvts_target_shape, vvts_target_void, tool_shape, tool_void
            )
            # Assume the body is the biggest piece
            result_pieces.sort(key=lambda solid: -solid.volume)
            vvts_target_shape = result_pieces[0]
            new_shape_pieces.extend(result_pieces[1:])

        final_shape = boolean_fuse([vvts_target_shape, *new_shape_pieces])
        final_void = boolean_fuse([vvts_target_void, *tool_voids])
        return final_shape, final_void, tool_voids

    def add_ports(self, ports: List[Component], n_TF: int):
        """
        Add ports to the thermal shield
        """
        vvts = self.vacuum_vessel_thermal_shield()
        cts = self.cryostat_thermal_shield()

        vvts_xyz = vvts.get_component("xyz")
        vv_xyz = vvts_xyz.get_component("Sector 1")
        vvts_target_name = f"{VVTSBuilder.VVTS} 1"
        vvts_void_name = f"{VVTSBuilder.VOID} 1"
        vvts_target_shape = vv_xyz.get_component(vvts_target_name).shape
        vvts_target_void = vv_xyz.get_component(vvts_void_name).shape

        cts_xyz = cts.get_component("xyz")
        cr_xyz = cts_xyz.get_component("Sector 1")
        cts_target_name = f"{CryostatTSBuilder.CRYO_TS} 1"
        cts_void_name = f"{CryostatTSBuilder.VOID} 1"
        cts_target_shape = cr_xyz.get_component(cts_target_name).shape
        cts_target_void = cr_xyz.get_component(cts_void_name).shape

        final_shape, final_void, tool_voids = self._join_ports_to_vvts(
            ports, vvts_target_shape, vvts_target_void
        )

        temp = boolean_cut(final_shape, cts_target_shape)
        temp.sort(key=lambda solid: -solid.volume)
        final_shape = temp[0]

        cts_target_shape = boolean_cut(cts_target_shape, tool_voids)[0]

        vvts_sector_body = PhysicalComponent(vvts_target_name, final_shape)
        vvts_sector_void = PhysicalComponent(
            vvts_void_name, final_void, material=Void("vacuum")
        )

        cts_sector_body = PhysicalComponent(cts_target_name, cts_target_shape)
        cts_sector_void = PhysicalComponent(
            cts_void_name, cts_target_void, material=Void("vacuum")
        )

        self._orphan_old_components([vvts, cts])
        self._create_new_components(
            vvts_sector_body, vvts_sector_void, cts_sector_body, cts_sector_void, n_TF
        )

    def _create_new_components(
        self,
        vvts_sector_body,
        vvts_sector_void,
        cts_sector_body,
        cts_sector_void,
        n_TF: int,
    ):
        vvts = self.vacuum_vessel_thermal_shield()
        cts = self.cryostat_thermal_shield()
        angle = 180 / n_TF
        apply_component_display_options(vvts_sector_body, color=BLUE_PALETTE["TS"][0])
        apply_component_display_options(vvts_sector_void, color=(0, 0, 0))
        apply_component_display_options(cts_sector_body, color=BLUE_PALETTE["TS"][0])
        apply_component_display_options(cts_sector_void, color=(0, 0, 0))
        Component(
            "xyz",
            children=[
                Component("Sector 1", children=[vvts_sector_body, vvts_sector_void])
            ],
            parent=vvts,
        )
        Component(
            "xyz",
            children=[
                Component("Sector 1", children=[cts_sector_body, cts_sector_void])
            ],
            parent=cts,
        )

        self._make_2d_views(
            vvts,
            vvts_sector_body,
            vvts_sector_void,
            angle,
            BLUE_PALETTE["TS"][0],
            void_color=(0, 0, 0),
        )

        self._make_2d_views(
            cts,
            cts_sector_body,
            cts_sector_void,
            angle,
            BLUE_PALETTE["TS"][0],
            void_color=(0, 0, 0),
        )


class Cryostat(PlugManagerMixin, ComponentManager):
    """
    Wrapper around a Cryostat component tree.
    """

    def xz_boundary(self) -> BluemiraWire:
        """Return a wire representing the Cryostat poloidal silhouette."""
        return (
            self.component()
            .get_component("xz")
            .get_component(CryostatBuilder.CRYO)
            .shape.boundary[0]
        )

    def add_plugs(self, plug_component: Component, n_TF: int):
        """
        Add plugs to the cryostat component.
        """
        self._add_plugs(
            plug_component, n_TF, f"{CryostatBuilder.CRYO} 1", BLUE_PALETTE["CR"]
        )


class RadiationShield(PlugManagerMixin, ComponentManager):
    """
    Wrapper around a RadiationShield component tree.
    """

    def xz_boundary(self) -> BluemiraWire:
        """Return a wire representing the RadiationShield poloidal silhouette."""
        return (
            self.component()
            .get_component("xz")
            .get_component(RadiationShieldBuilder.BODY)
            .shape.boundary[0]
        )

    def add_plugs(self, plug_component: Component, n_TF: int):
        """
        Add plugs to the radiation shield component.
        """
        self._add_plugs(
            plug_component, n_TF, f"{RadiationShieldBuilder.BODY} 1", BLUE_PALETTE["RS"]
        )


class CoilStructures(ComponentManager):
    """
    Wrapper around the coil structures component tree
    """
