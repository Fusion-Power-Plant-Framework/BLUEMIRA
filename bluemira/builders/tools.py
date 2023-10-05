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
A collection of tools used in the EU-DEMO design.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from bluemira.display.palettes import ColorPalette

if TYPE_CHECKING:
    from collections.abc import Iterable

    from bluemira.geometry.solid import BluemiraSolid
    from bluemira.geometry.wire import BluemiraWire
    from bluemira.materials.material import SerialisedMaterial

import numpy as np
from anytree import PreOrderIter

import bluemira.base.components as bm_comp
import bluemira.geometry as bm_geo
from bluemira.base.components import PhysicalComponent
from bluemira.base.constants import EPS
from bluemira.base.error import BuilderError, ComponentError
from bluemira.builders._varied_offset import varied_offset
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import (
    boolean_cut,
    boolean_fuse,
    circular_pattern,
    extrude_shape,
    make_circle,
    make_polygon,
    revolve_shape,
    slice_shape,
    sweep_shape,
)

__all__ = [
    "apply_component_display_options",
    "get_n_sectors",
    "circular_pattern_component",
    "pattern_revolved_silhouette",
    "pattern_lofted_silhouette",
    "varied_offset",
    "find_xy_plane_radii",
    "make_circular_xy_ring",
    "build_sectioned_xy",
    "build_sectioned_xyz",
]


def apply_component_display_options(
    phys_component: PhysicalComponent,
    color: Union[Iterable, ColorPalette],
    transparency: Optional[float] = None,
):
    """
    Apply color and transparency to a PhysicalComponent for both plotting and CAD.
    """
    if isinstance(color, ColorPalette):
        color = color.as_hex()
    phys_component.plot_options.face_options["color"] = color
    phys_component.display_cad_options.color = color
    if transparency:
        phys_component.plot_options.face_options["alpha"] = transparency
        phys_component.display_cad_options.transparency = transparency


def get_n_sectors(no_obj: int, degree: float = 360) -> Tuple[float, int]:
    """
    Get sector count and angle size for a given number of degrees of the reactor.

    Parameters
    ----------
    no_obj:
        total number of components (eg TF coils)
    degree:
        angle to view of reactor

    Returns
    -------
    sector_degree:
        number of degrees per sector
    n_sectors:
        number of sectors
    """
    sector_degree = 360 / no_obj
    n_sectors = max(1, int(degree // int(sector_degree)))
    return sector_degree, n_sectors


def circular_pattern_component(
    component: Union[bm_comp.Component, List[bm_comp.Component]],
    n_children: int,
    parent_prefix: str = "Sector",
    *,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    degree: float = 360.0,
):
    """
    Pattern the provided Component equally spaced around a circle n_children times.

    The resulting components are assigned to a set of common parent Components having
    a name with the structure "{parent_prefix} {idx}", where idx runs from 1 to
    n_children. The Components produced under each parent are named according to the
    original Component with the corresponding idx value appended.

    Parameters
    ----------
    component:
        The original Component to use as the template for copying around the circle.
    n_children:
        The number of children to produce around the circle.
    parent_prefix:
        The prefix to provide to the new parent component, having a name of the form
        "{parent_prefix} {idx}", by default "Sector".
    origin:
        The origin of the circle to pattern around, by default (0., 0., 0.).
    direction:
        The surface normal of the circle to pattern around, by default (0., 0., 1.) i.e.
        the positive z axis, resulting in a counter clockwise circle in the x-y plane.
    degree:
        The angular extent of the patterning in degrees, by default 360.
    """
    component = [component] if isinstance(component, bm_comp.Component) else component
    sectors = [bm_comp.Component(f"{parent_prefix}") for _ in range(n_children)]
    # build sector trees by assigning copies of each component to sec. parents
    for c in component:
        for parent_sc in sectors:
            c.copy(parent_sc)

    sector_tree_indexs = [list(PreOrderIter(sc)) for sc in sectors]

    # to keep naming convention
    for sec_i, sector_index in enumerate(sector_tree_indexs):
        for comp in sector_index:
            comp.name = f"{comp.name} {sec_i + 1}"

    faux_sec_comp = bm_comp.Component(f"{parent_prefix} X")
    faux_sec_comp.children = component

    for search_index_i, comp in enumerate(PreOrderIter(faux_sec_comp)):
        if isinstance(comp, bm_comp.PhysicalComponent):
            shapes = bm_geo.tools.circular_pattern(
                comp.shape,
                n_shapes=n_children,
                origin=origin,
                direction=direction,
                degree=degree,
            )
            # assign each shape to each sector at index search_index_i
            # which should be the copy of the PhysicalComponent
            for sector_index, shape in zip(sector_tree_indexs, shapes):
                phy_comp = sector_index[search_index_i]
                if not isinstance(phy_comp, bm_comp.PhysicalComponent):
                    raise ComponentError(
                        "Could not find corresponding PhysicalComponent in "
                        f"sector index: {sector_index}, "
                        f"with search index: {search_index_i}"
                    )
                phy_comp.shape = shape

    return sectors


def pattern_revolved_silhouette(
    face: BluemiraFace, n_seg_p_sector: int, n_sectors: int, gap: float
) -> List[BluemiraSolid]:
    """
    Pattern a silhouette with revolutions about the z-axis, inter-spaced with parallel
    gaps between solids.

    Parameters
    ----------
    face:
        x-z silhouette of the geometry to revolve and pattern
    n_seg_p_sector:
        Number of segments per sector
    n_sectors:
        Number of sectors
    gap:
        Absolute distance between segments (parallel)

    Returns
    -------
    List of solids for each segment (ordered anti-clockwise)
    """
    sector_degree = 360 / n_sectors

    if gap <= 0.0:  # noqa: PLR2004
        # No gaps; just touching solids
        segment_degree = sector_degree / n_seg_p_sector
        shape = revolve_shape(
            face, base=(0, 0, 0), direction=(0, 0, 1), degree=segment_degree
        )
        shapes = circular_pattern(
            shape, origin=(0, 0, 0), degree=sector_degree, n_shapes=n_seg_p_sector
        )
    else:
        volume = revolve_shape(
            face, base=(0, 0, 0), direction=(0, 0, 1), degree=sector_degree
        )
        gaps = _generate_gap_volumes(face, n_seg_p_sector, n_sectors, gap)
        shapes = boolean_cut(volume, gaps)
    return _order_shapes_anticlockwise(shapes)


def pattern_lofted_silhouette(
    face: BluemiraFace, n_seg_p_sector: int, n_sectors: int, gap: float
) -> List[BluemiraSolid]:
    """
    Pattern a silhouette with lofts about the z-axis, inter-spaced with parallel
    gaps between solids.

    Parameters
    ----------
    face:
        x-z silhouette of the geometry to loft and pattern
    n_seg_p_sector:
        Number of segments per sector
    n_sectors:
        Number of sectors
    gap:
        Absolute distance between segments (parallel)

    Returns
    -------
    List of solids for each segment (ordered anti-clockwise)
    """
    sector_degree = 360 / n_sectors

    degree = sector_degree * (1 + 1 / n_seg_p_sector)
    faces = circular_pattern(
        face,
        origin=(0, 0, 0),
        direction=(0, 0, 1),
        degree=degree,
        n_shapes=n_seg_p_sector + 1,
    )
    shapes = []
    for i, r_face in enumerate(faces[:-1]):
        com_1 = r_face.center_of_mass
        com_2 = faces[i + 1].center_of_mass

        wire = make_polygon(
            {
                "x": [com_1[0], com_2[0]],
                "y": [com_1[1], com_2[1]],
                "z": [com_1[2], com_2[2]],
            }
        )
        volume = sweep_shape([r_face.boundary[0], faces[i + 1].boundary[0]], wire)
        shapes.append(volume)

    if gap > 0.0:  # noqa: PLR2004
        full_volume = boolean_fuse(shapes) if len(shapes) > 1 else shapes[0]

        gaps = _generate_gap_volumes(face, n_seg_p_sector, n_sectors, gap)
        shapes = boolean_cut(full_volume, gaps)

    return _order_shapes_anticlockwise(shapes)


def _generate_gap_volumes(face, n_seg_p_sector, n_sectors, gap):
    """
    Generate the gap volumes
    """
    bb = face.bounding_box
    delta = 1.0
    x = np.array(
        [bb.x_min - delta, bb.x_max + delta, bb.x_max + delta, bb.x_min - delta]
    )
    z = np.array(
        [bb.z_min - delta, bb.z_min - delta, bb.z_max + delta, bb.z_max + delta]
    )
    poly = make_polygon({"x": x, "y": 0, "z": z}, closed=True)
    bb_face = BluemiraFace(poly)
    bb_face.translate((0, -0.5 * gap, 0))
    gap_volume = extrude_shape(bb_face, (0, gap, 0))
    degree = 360 / n_sectors
    degree += degree / n_seg_p_sector
    return circular_pattern(gap_volume, degree=degree, n_shapes=n_seg_p_sector + 1)


def _order_shapes_anticlockwise(shapes):
    """
    Order shapes anti-clockwise about (0, 0, 1) by center of mass
    """
    x, y = np.zeros(len(shapes)), np.zeros(len(shapes))

    for i, shape in enumerate(shapes):
        com = shape.center_of_mass
        x[i] = com[0]
        y[i] = com[1]

    r = np.hypot(x, y)
    angles = np.where(y > 0, np.arccos(x / r), 2 * np.pi - np.arccos(x / r))
    indices = np.argsort(angles)
    return list(np.array(shapes)[indices])


def find_xy_plane_radii(wire: BluemiraWire, plane: BluemiraPlane) -> List[float]:
    """
    Get the radial coordinates of a wire's intersection points with a plane.

    Parameters
    ----------
    wire:
        Wire to get the radii for in the plane
    plane:
        Plane to slice with

    Returns
    -------
    The radii of intersections, sorted from smallest to largest
    """
    intersections = slice_shape(wire, plane)
    return sorted(intersections[:, 0])


def make_circular_xy_ring(r_inner: float, r_outer: float) -> BluemiraFace:
    """
    Make a circular annulus in the x-y plane (z=0)
    """
    centre = (0, 0, 0)
    axis = (0, 0, 1)
    if np.isclose(r_inner, r_outer, rtol=0, atol=2 * EPS):
        raise BuilderError(f"Cannot make an annulus where r_inner = r_outer = {r_inner}")

    if r_inner > r_outer:
        r_inner, r_outer = r_outer, r_inner

    inner = make_circle(r_inner, center=centre, axis=axis)
    outer = make_circle(r_outer, center=centre, axis=axis)
    return BluemiraFace([outer, inner])


def build_sectioned_xy(
    face: BluemiraFace,
    plot_colour: Tuple[float],
    material: Optional[SerialisedMaterial] = None,
) -> List[PhysicalComponent]:
    """
    Build the x-y components of sectioned component

    Parameters
    ----------
    face:
        xz face to build xy component
    plot_colour:
        colour tuple for component
    material:
        Optional material to apply to physical component

    Returns
    -------
    List of PhysicalComponents with colours applied
    """
    xy_plane = BluemiraPlane.from_3_points([0, 0, 0], [1, 0, 0], [1, 1, 0])

    r_ib_out, r_ob_out = find_xy_plane_radii(face.boundary[0], xy_plane)
    r_ib_in, r_ob_in = find_xy_plane_radii(face.boundary[1], xy_plane)

    sections = []
    for name, r_in, r_out in [
        ["inboard", r_ib_in, r_ib_out],
        ["outboard", r_ob_in, r_ob_out],
    ]:
        board = make_circular_xy_ring(r_in, r_out)
        section = PhysicalComponent(name, board, material=material)
        apply_component_display_options(section, color=plot_colour)
        sections.append(section)

    return sections


def build_sectioned_xyz(
    face: BluemiraFace,
    name: str,
    n_TF: int,
    plot_colour: Tuple[float],
    degree: float = 360,
    enable_sectioning: bool = True,
    material: Optional[SerialisedMaterial] = None,
) -> List[PhysicalComponent]:
    """
    Build the x-y-z components of sectioned component

    Parameters
    ----------
    face:
        xz face to build xyz component
    name:
        PhysicalComponent name
    n_TF:
        number of TF coils
    plot_colour:
        colour tuple for component
    degree:
        angle to sweep through
    enable_sectioning:
        Switch on/off sectioning (#1319 Topology issue)
    material:
        Optional material to apply to physical component

    Returns
    -------
    List of PhysicalComponents

    Notes
    -----
    When `enable_sectioning=False` a list with a single component rotated a maximum
    of 359 degrees will be returned. This is a workaround for two issues
    from the topology naming issue #1319:

        - Some objects fail to be rebuilt when rotated
        - Some objects cant be rotated 360 degrees due to DisjointedFaceError

    """
    sector_degree, n_sectors = get_n_sectors(n_TF, degree)

    if isinstance(face, BluemiraFace):
        face = [face]
    if isinstance(name, str):
        name = [name]
    if isinstance(plot_colour, Tuple):
        plot_colour = [plot_colour]
    if not isinstance(material, list):
        material = [material]

    if not (len(face) == len(name) == len(plot_colour) == len(material)):
        raise ValueError(
            "Lengths of the face, name, plot_colour, and material lists are not equal."
        )

    bodies = []
    for fac, nam, color, mat in zip(face, name, plot_colour, material):
        shape = revolve_shape(
            fac,
            base=(0, 0, 0),
            direction=(0, 0, 1),
            degree=sector_degree if enable_sectioning else min(359, degree),
        )
        body = PhysicalComponent(nam, shape, material=mat)
        apply_component_display_options(body, color=color)
        bodies.append(body)

    # this is currently broken in some situations
    # because of #1319 and related Topological naming issues
    return (
        circular_pattern_component(bodies, n_sectors, degree=sector_degree * n_sectors)
        if enable_sectioning
        else bodies
    )
