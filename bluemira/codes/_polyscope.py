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
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

import numpy as np
import polyscope as ps
from matplotlib import colors

import bluemira.codes._freecadapi as cadapi
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.utilities.tools import ColourDescriptor

if TYPE_CHECKING:
    from bluemira.display.palettes import ColorPalette


@dataclass
class DefaultDisplayOptions:
    """Polyscope default display options"""

    colour: ColourDescriptor = ColourDescriptor()
    transparency: float = 0.0
    material: str = "wax"
    tesselation: float = 0.05
    wires_on: bool = False
    wire_radius: float = 0.001
    smooth: bool = True

    @property
    def color(self) -> str:
        """See colour"""
        return self.colour

    @color.setter
    def color(self, value: Union[str, Tuple[float, float, float], ColorPalette]):
        """See colour"""
        self.colour = value


def show_cad(
    parts: Union[cadapi.apiShape, List[cadapi.apiShape]],
    part_options: List[Dict],
    labels: List[str],
    **kwargs,
):
    """
    The implementation of the display API for FreeCAD parts.

    Parameters
    ----------
    parts:
        The parts to display.
    part_options:
        The options to use to display the parts.
    labels:
        Labels to use for each part object
    **kwargs:
        options passed to polyscope
    """
    if part_options is None:
        part_options = [None]

    if None in part_options:
        part_options = [
            DefaultDisplayOptions() if o is None else o for o in part_options
        ]

    transparency = "none"
    for opt in part_options:
        if not np.isclose(opt["transparency"], 0):
            transparency = "pretty"
            break

    polyscope_setup(
        up_direction=kwargs.get("up_direction", "z_up"),
        fps=kwargs.get("fps", 60),
        aa=kwargs.get("aa", 1),
        transparency=transparency,
        render_passes=kwargs.get("render_passes", 3),
        gplane=kwargs.get("gplane", "none"),
    )

    add_features(labels, parts, part_options)

    ps.show()


def polyscope_setup(
    up_direction: str = "z_up",
    fps: int = 60,
    aa: int = 1,
    transparency: str = "pretty",
    render_passes: int = 2,
    gplane: str = "none",
):
    """
    Setup Polyscope default scene

    Parameters
    ----------
    up_direction:
        'x_up' The positive X-axis is up.
        'neg_x_up' The negative X-axis is up.
        'y_up' The positive Y-axis is up.
        'neg_y_up' The negative Y-axis is up.
        'z_up' The positive Z-axis is up.
        'neg_z_up' The negative Z-axis is up.
    fps:
        maximum frames per second of viewer (-1 == infinite)
    aa:
        anti aliasing amount, 1 is off, 2 is usually enough
    transparency:
        the transparency mode (none, simple, pretty)
    render_passes:
        for transparent shapes how many render passes to undertake
    gplane:
        the ground plane mode (none, tile, tile_reflection, shadon_only)
    """
    _init_polyscope()

    ps.set_max_fps(fps)
    ps.set_SSAA_factor(aa)
    ps.set_transparency_mode(transparency)
    if transparency != "none":
        ps.set_transparency_render_passes(render_passes)
    ps.set_ground_plane_mode(gplane)
    ps.set_up_dir(up_direction)

    ps.remove_all_structures()


@functools.lru_cache(maxsize=1)
def _init_polyscope():
    """
    Initialise polyscope (just once)
    """
    bluemira_warn(
        "Polyscope is not a NURBS based viewer."
        " Some features may appear subtly different to their CAD representation"
    )
    ps.set_program_name("Bluemira Display")
    ps.init()


def add_features(
    labels: List[str],
    parts: Union[cadapi.apiShape, List[cadapi.apiShape]],
    options: Union[Dict, List[Dict]],
) -> Tuple[List[ps.SurfaceMesh], List[ps.CurveNetwork]]:
    """
    Grab meshes of all parts to be displayed by Polyscope

    Parameters
    ----------
    parts:
        parts to be displayed
    options:
        display options

    Returns
    -------
    Registered Polyspline surface meshes

    """
    meshes = []
    curves = []

    # loop over every face adding their meshes to polyscope
    for shape_i, (label, part, option) in enumerate(
        zip(labels, parts, options),
    ):
        verts, faces = cadapi.collect_verts_faces(part, option["tesselation"])

        if not (verts is None or faces is None):
            m = ps.register_surface_mesh(
                clean_name(label, str(shape_i)),
                verts,
                faces,
                smooth_shade=option["smooth"],
                color=colors.to_rgb(option["colour"]),
                transparency=1 - option["transparency"],
                material=option["material"],
            )
            meshes.append(m)

        if option["wires_on"] or (verts is None or faces is None):
            verts, edges = cadapi.collect_wires(part, Deflection=0.01)
            c = ps.register_curve_network(
                clean_name(label, f"{shape_i}_wire"),
                verts,
                edges,
                radius=option["wire_radius"],
                color=colors.to_rgb(option["colour"]),
                transparency=1 - option["transparency"],
                material=option["material"],
            )
            curves.append(c)

    return meshes, curves


def clean_name(label: str, index_label: str) -> str:
    """
    Cleans or creates name.
    Polyscope doesn't like hashes in names,
    repeat names overwrite existing component.

    Parameters
    ----------
    label:
        name to be cleaned
    index_label:
        if name is empty -> {index_label}: NO LABEL

    Returns
    -------
    name
    """
    label = label.replace("#", "_")
    index_label = index_label.replace("#", "_")
    if len(label) == 0 or label == "_":
        return f"{index_label}: NO LABEL"
    return f"{index_label}: {label}"
