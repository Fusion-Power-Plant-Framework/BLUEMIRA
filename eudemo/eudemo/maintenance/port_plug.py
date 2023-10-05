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
Port plugs
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Type, Union

if TYPE_CHECKING:
    from bluemira.geometry.solid import BluemiraSolid

from dataclasses import dataclass

import numpy as np

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.builders.tools import apply_component_display_options
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import boolean_fuse, extrude_shape, offset_wire
from bluemira.geometry.wire import BluemiraWire
from bluemira.materials import Void


def make_castellated_plug(
    face: BluemiraFace,
    vec: Tuple[float, float, float],
    length: float,
    offsets: Union[float, Iterable],
    distances: Optional[Iterable] = None,
    n_castellations: Optional[int] = None,
) -> BluemiraSolid:
    """
    Make a castellated port plug.

    Parameters
    ----------
    face:
        starting profile to be castellated
    vec:
        unit vector along which to extrude
    length:
        total length of castellated BluemiraSolid in vec direction
    offsets:
        castellations offset(s) for each position
    distances:
        (optional) parameter for manually spaced castellations
    n_castellations:
        (optional) parameter for equally spaced castellations

    Returns
    -------
    BluemiraSolid of a castellated port plug
    """
    # Normalise vec
    vec = np.array(vec) / np.linalg.norm(vec)

    # Check/Set-up distances iterable
    if not ((n_castellations is None) or (n_castellations == 0)):
        interval = length / (n_castellations + 1)
        dist_iter = [interval * i for i in range(1, n_castellations + 1)]
    elif distances is not None:
        dist_iter = distances
    else:
        raise ValueError("Both distance and n_castellations parameters are None")

    # Check/Set-up offsets iterable
    if type(offsets) is float:  # noqa: E721
        off_iter = [offsets] * len(dist_iter)
    elif len(offsets) == len(dist_iter):
        off_iter = offsets
    else:
        raise ValueError("Length of offsets doesn't match distances/n_cast")

    parameter_array = list(zip(dist_iter, off_iter))
    parameter_array.append((length, 0.0))

    base = face
    sections = []
    _prev_dist = 0
    for dist, off in parameter_array:
        ext_vec = vec * (dist - _prev_dist)
        sections.append(extrude_shape(base, ext_vec))
        base.translate(ext_vec)
        base = BluemiraFace(offset_wire(BluemiraWire(base.wires), off))
        _prev_dist = dist

    return boolean_fuse(sections)


def make_onion_layer_plug_void(
    outer_profiles: Iterable[BluemiraWire],
    target_profile: BluemiraFace,
    thickness: float,
    offset: float,
    gap: float,
    tk_castellation: float,
    n_castellations: int,
    n_TF: int,
) -> Tuple[List[BluemiraSolid], List[BluemiraSolid]]:
    """
    Make geometries for castellated port plugs and voids for all ports.

    Parameters
    ----------
    outer_profiles:
        Outer profiles of the inner port plug imprint
    target_profile:
        Face onto which to project the port
    thickness:
        Thickness of the part through which the port opening must go through
    offset:
        Offset value from the outer profile
    tk_castellation:
        Thickness of the castellation offsets
    n_castellations:
        Number of castellations in the port plugs
    n_TF:
        Number of TF coils

    Returns
    -------
    plugs:
        BluemiraSolids of the port plugs
    voids:
        BluemiraSolids of the voids to cut away from the target
    """
    degree = 180 / n_TF
    x_max = target_profile.bounding_box.x_max
    z_max = target_profile.bounding_box.z_max
    plugs = []
    voids = []
    for wire in outer_profiles:
        bb = wire.bounding_box
        dx = abs(bb.x_max - x_max)
        dz = abs(bb.z_max - z_max)
        if dx < dz:
            # Horizontal connection
            dy = 0.5 * abs(bb.y_max - bb.y_min) + offset
            radius = np.sqrt((x_max - thickness) ** 2 - dy**2)
            length = x_max - radius
            vector = (radius - bb.x_max, 0, 0)

        else:
            # Vertical connection
            length = thickness
            vector = (0, 0, dz - thickness)

        wire.translate(vector)
        void_wire = offset_wire(wire, gap)

        plug = make_castellated_plug(
            BluemiraFace(wire),
            vector,
            length,
            offsets=tk_castellation,
            n_castellations=n_castellations,
        )
        void = make_castellated_plug(
            BluemiraFace(void_wire),
            vector,
            length,
            offsets=tk_castellation,
            n_castellations=n_castellations,
        )

        plug.rotate(degree=degree)
        void.rotate(degree=degree)
        plugs.append(plug)
        voids.append(void)
    return plugs, voids


@dataclass
class CryostatPortPlugBuilderParams(ParameterFrame):
    """
    Cryostat port plug builder parameters
    """

    # Global
    n_TF: Parameter[int]
    tk_cr_vv: Parameter[float]
    g_cr_ts: Parameter[float]

    # Local
    g_plug: Parameter[float]
    tk_castellation: Parameter[float]
    n_plug_castellations: Parameter[int]


class CryostatPortPlugBuilder(Builder):
    """
    Cryostat port plug builder.
    """

    param_cls: Type[CryostatPortPlugBuilderParams] = CryostatPortPlugBuilderParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame, CryostatPortPlugBuilderParams],
        build_config: Optional[Dict],
        outer_profiles: Iterable[BluemiraWire],
        cryostat_xz_boundary: BluemiraFace,
    ):
        super().__init__(params, build_config)
        self.outer_profiles = outer_profiles
        self.cryostat_xz_boundary = cryostat_xz_boundary

    def build(self) -> Component:
        """Build the Cryostat port plugs"""
        return self.component_tree(
            xz=None,
            xy=None,
            xyz=self.build_xyz(),
        )

    def build_xyz(self) -> List[PhysicalComponent]:
        """
        Build the 3D representation of the Cryostat port plugs
        """
        plugs, voids = make_onion_layer_plug_void(
            self.outer_profiles,
            self.cryostat_xz_boundary,
            self.params.tk_cr_vv.value,
            self.params.g_cr_ts.value,
            self.params.g_plug.value,
            self.params.tk_castellation.value,
            self.params.n_plug_castellations.value,
            self.params.n_TF.value,
        )

        plug_comps, void_comps = [], []
        for i, (plug, void) in enumerate(zip(plugs, voids)):
            plug = PhysicalComponent(f"{self.name} {i}", plug)  # noqa: PLW2901
            void = PhysicalComponent(  # noqa: PLW2901
                f"{self.name} {i} voidspace", void, material=Void("air")
            )
            apply_component_display_options(plug, BLUE_PALETTE["CR"][1])
            apply_component_display_options(void, (0, 0, 0))
            plug_comps.append(plug)
            void_comps.append(void)

        return plug_comps + void_comps


@dataclass
class RadiationPortPlugBuilderParams(ParameterFrame):
    """
    Radiation shield port plug builder parameters
    """

    # Global
    n_TF: Parameter[int]
    tk_rs: Parameter[float]
    g_cr_rs: Parameter[float]

    # Local
    g_plug: Parameter[float]
    tk_castellation: Parameter[float]
    n_plug_castellations: Parameter[int]


class RadiationPortPlugBuilder(Builder):
    """
    Radiation shield port plug builder.
    """

    param_cls: Type[RadiationPortPlugBuilderParams] = RadiationPortPlugBuilderParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame, RadiationPortPlugBuilderParams],
        build_config: Optional[Dict],
        outer_profiles: Iterable[BluemiraWire],
        radiation_xz_boundary: BluemiraFace,
    ):
        super().__init__(params, build_config)
        self.outer_profiles = outer_profiles
        self.radiation_xz_boundary = radiation_xz_boundary

    def build(self) -> Component:
        """Build the radiation shield port plugs"""
        return self.component_tree(
            xz=None,
            xy=None,
            xyz=self.build_xyz(),
        )

    def build_xyz(self) -> List[PhysicalComponent]:
        """
        Build the 3D representation of the radiation shield port plugs
        """
        plugs, voids = make_onion_layer_plug_void(
            self.outer_profiles,
            self.radiation_xz_boundary,
            self.params.tk_rs.value,
            self.params.g_cr_rs.value,
            self.params.g_plug.value,
            self.params.tk_castellation.value,
            self.params.n_plug_castellations.value,
            self.params.n_TF.value,
        )
        plug_comps, void_comps = [], []
        for i, (plug, void) in enumerate(zip(plugs, voids)):
            plug = PhysicalComponent(f"{self.name} {i}", plug)  # noqa: PLW2901
            void = PhysicalComponent(  # noqa: PLW2901
                f"{self.name} {i} voidspace", void, material=Void("air")
            )
            apply_component_display_options(plug, BLUE_PALETTE["RS"][1])
            apply_component_display_options(void, (0, 0, 0))
            plug_comps.append(plug)
            void_comps.append(void)

        return plug_comps + void_comps
