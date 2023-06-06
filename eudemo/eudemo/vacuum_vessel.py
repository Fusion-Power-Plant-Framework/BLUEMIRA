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
Builder for making a parameterised EU-DEMO vacuum vessel.
"""
from dataclasses import dataclass
from typing import Dict, List, Type, Union

from bluemira.base.builder import Builder, ComponentManager
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.builders.tools import (
    apply_component_display_options,
    build_sectioned_xy,
    build_sectioned_xyz,
    circular_pattern_component,
    get_n_sectors,
    varied_offset,
)
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import _offset_wire_discretised
from bluemira.geometry.wire import BluemiraWire
from bluemira.materials.cache import Void
from eudemo.maintenance.duct_connection import pipe_pipe_join


class VacuumVessel(ComponentManager):
    """
    Wrapper around a Vacuum Vessel component tree.
    """

    def xz_boundary(self) -> BluemiraWire:
        """Return a wire giving the vessel's boundary in the xz plane."""
        return (
            self.component()
            .get_component("xz")
            .get_component(VacuumVesselBuilder.BODY)
            .shape.boundary[0]
        )

    def add_ports(self, n_TF: int, port: Union[Component, List[Component]]):
        """ """
        if isinstance(port, Component):
            port = [port]
        degree = 0.0
        sector_degree, n_sectors = get_n_sectors(n_TF, degree)
        component = self.component()
        xyz = component.get_component("xyz")
        vv_xyz = xyz.get_component("Sector 1")
        target_void = vv_xyz.get_component("Vessel voidspace 1").shape
        target_shape = vv_xyz.get_component("Body 1").shape
        xyz.parent = None
        del xyz
        port_xyz = port.get_component("xyz")
        tool_shape = port_xyz.get_component(port.name).shape
        tool_void = port_xyz.get_component(port.name + " voidspace").shape
        shape, void = pipe_pipe_join(target_shape, target_void, tool_shape, tool_void)
        sector_body = PhysicalComponent(VacuumVesselBuilder.BODY, shape)
        sector_void = PhysicalComponent(
            VacuumVesselBuilder.VOID, void, material=Void("vacuum")
        )
        Component("xyz", children=[sector_body, sector_void], parent=component)


@dataclass
class VacuumVesselBuilderParams(ParameterFrame):
    """
    Vacuum Vessel builder parameters
    """

    n_TF: Parameter[int]
    r_vv_ib_in: Parameter[float]
    r_vv_ob_in: Parameter[float]
    tk_vv_in: Parameter[float]
    tk_vv_out: Parameter[float]
    g_vv_bb: Parameter[float]
    vv_in_off_deg: Parameter[float]
    vv_out_off_deg: Parameter[float]


class VacuumVesselBuilder(Builder):
    """
    Vacuum Vessel builder
    """

    VV = "VV"
    BODY = "Body"
    VOID = "Vessel voidspace"
    param_cls: Type[VacuumVesselBuilderParams] = VacuumVesselBuilderParams

    def __init__(
        self,
        params: Union[ParameterFrame, Dict],
        build_config: Dict,
        ivc_koz: BluemiraWire,
    ):
        super().__init__(params, build_config)
        self.ivc_koz = ivc_koz

    def build(self) -> Component:
        """
        Build the vacuum vessel component.
        """
        xz_vv, xz_vacuum = self.build_xz()
        vv_face = xz_vv.get_component_properties("shape")
        vacuum_face = xz_vacuum.get_component_properties("shape")

        return self.component_tree(
            xz=[xz_vv, xz_vacuum],
            xy=self.build_xy(vv_face),
            xyz=self.build_xyz(vv_face, vacuum_face, degree=0),
        )

    def build_xz(
        self,
    ) -> PhysicalComponent:
        """
        Build the x-z components of the vacuum vessel.
        """
        inner_vv = _offset_wire_discretised(
            self.ivc_koz,
            self.params.g_vv_bb.value,
            join="arc",
            open_wire=False,
            ndiscr=600,
        )

        outer_vv = varied_offset(
            inner_vv,
            self.params.tk_vv_in.value,
            self.params.tk_vv_out.value,
            self.params.vv_in_off_deg.value,
            self.params.vv_out_off_deg.value,
            num_points=300,
        )
        face = BluemiraFace([outer_vv, inner_vv])

        body = PhysicalComponent(self.BODY, face)
        vacuum = PhysicalComponent(
            self.VOID, BluemiraFace(inner_vv), material=Void("vacuum")
        )
        apply_component_display_options(body, color=BLUE_PALETTE[self.VV][0])
        apply_component_display_options(vacuum, color=(0, 0, 0))

        return body, vacuum

    def build_xy(self, vv_face: BluemiraFace) -> List[PhysicalComponent]:
        """
        Build the x-y components of the vacuum vessel.
        """
        return build_sectioned_xy(vv_face, BLUE_PALETTE[self.VV][0])

    def build_xyz(
        self, vv_face: BluemiraFace, vacuum_face: BluemiraFace, degree: float = 360.0
    ) -> PhysicalComponent:
        """
        Build the x-y-z components of the vacuum vessel.
        """
        return build_sectioned_xyz(
            [vv_face, vacuum_face],
            [self.BODY, self.VOID],
            self.params.n_TF.value,
            [BLUE_PALETTE[self.VV][0], (0, 0, 0)],
            degree,
            material=[None, Void("vacuum")],
        )
