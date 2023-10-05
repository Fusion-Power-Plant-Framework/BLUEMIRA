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
from typing import List

import numpy as np

from bluemira.base.components import PhysicalComponent
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import slice_shape


def get_inner_cut_point(breeding_blanket_xz, r_inner_cut):
    """
    Get the inner cut point of the breeding blanket geometry.
    """
    cut_plane = BluemiraPlane.from_3_points(
        [r_inner_cut, 0, 0], [r_inner_cut, 0, 1], [r_inner_cut, 1, 1]
    )
    # Get the first intersection with the vertical inner cut plane
    intersections = slice_shape(breeding_blanket_xz.boundary[0], cut_plane)
    intersections = intersections[intersections[:, -1] > 0.0]  # noqa: PLR2004
    return sorted(intersections, key=lambda x: x[-1])[0]


def make_2d_view_components(
    view: str, azimuthal_angle: float, components: List[PhysicalComponent]
) -> List[PhysicalComponent]:
    """
    Make a 2-D slice of a list of 3-D components

    Parameters
    ----------
    view:
        View to make the components for. From ['xz', 'xy']
    azimuthal_angle:
        Angle at which to cut the x-z plane [degree]. Has no effect for the
        x-y plane
    components:
        List of PhysicalComponents to take slices from

    Returns
    -------
    List of PhysicalComponents in the desired view plane
    """
    azimuthal_angle = np.deg2rad(azimuthal_angle)
    if view == "xz":
        plane = BluemiraPlane.from_3_points(
            [0, 0, 0], [0, 0, 1], [np.cos(azimuthal_angle), np.sin(azimuthal_angle), 0]
        )
    elif view == "xy":
        plane = BluemiraPlane.from_3_points([0, 0, 0], [0, 1, 0], [1, 0, 0])
    else:
        raise ValueError(f"Unrecognised view: {view}, please choose from ['xz', 'xy']")

    view_comps = []
    for comp in components:
        comp_slices = []
        pieces = slice_shape(comp.shape, plane)
        # TODO: slice_shape is unreliable for complex shapes...

        if pieces:
            for i, piece in enumerate(pieces):
                face = BluemiraFace(piece)
                if azimuthal_angle != 0:
                    face.rotate(degree=-np.rad2deg(azimuthal_angle))
                new_comp = PhysicalComponent(
                    f"{comp.name} {i}", face, material=comp.material
                )
                comp_slices.append(new_comp)
            view_comps.append(comp_slices)
    return view_comps
