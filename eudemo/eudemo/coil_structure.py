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
Coil structure stuff
"""
from dataclasses import dataclass

from bluemira.base.components import Component
from bluemira.base.parameter_frame import Parameter, ParameterFrame, make_parameter_frame
from bluemira.builders.coil_supports import (
    ITERGravitySupportBuilder,
    OISBuilder,
    PFCoilSupportBuilder,
    StraightOISDesigner,
)


@dataclass
class CoilStructuresParameters(ParameterFrame):
    """
    Parameters for the coil structures
    """

    n_TF: Parameter[int]
    tf_wp_depth: Parameter[float]
    tk_tf_side: Parameter[float]
    tf_wp_width: Parameter[float]

    # OIS
    tk_ois: Parameter[float]
    g_ois_tf_edge: Parameter[float]
    min_OIS_length: Parameter[float]

    # PF
    pf_s_tk_plate: Parameter[float]
    pf_s_n_plate: Parameter[int]
    pf_s_g: Parameter[float]

    # GS
    x_g_support: Parameter[float]
    z_gs: Parameter[float]
    tf_gs_tk_plate: Parameter[float]
    tf_gs_g_plate: Parameter[float]
    tf_gs_base_depth: Parameter[float]


def build_coil_structures_component(
    params, build_config, tf_coil_xz_face, pf_coil_xz_wires, pf_coil_keep_out_zones
):
    """
    Build the coil structures super-component.
    """
    params = make_parameter_frame(params, CoilStructuresParameters)
    ois_designer = StraightOISDesigner(
        params, build_config, tf_coil_xz_face, pf_coil_keep_out_zones
    )
    ois_xz_profiles = ois_designer.run()
    ois_builder = OISBuilder(params, build_config, ois_xz_profiles)
    ois_component = ois_builder.build()

    tf_koz = tf_coil_xz_face.boundary[0]
    support_components = []
    for i, pf_coil in enumerate(pf_coil_xz_wires):
        bc = {**build_config, "support_number": str(i)}
        pf_support_builder = PFCoilSupportBuilder(params, bc, tf_koz, pf_coil)
        support_components.append(pf_support_builder.build())

    pf_support_component = Component("PF supports", children=support_components)

    gs_builder = ITERGravitySupportBuilder(params, build_config, tf_koz)
    gs_component = gs_builder.build()

    return Component(
        "Coil Structures", children=[ois_component, pf_support_component, gs_component]
    )
