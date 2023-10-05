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
"""Functions related to EUDEMO PF coils."""

from typing import List

import numpy as np

from bluemira.base.constants import EPS
from bluemira.base.error import BuilderError
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.equilibria.coils import Coil, CoilSet
from bluemira.geometry.constants import VERY_BIG
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import (
    boolean_cut,
    distance_to,
    make_polygon,
    offset_wire,
    split_wire,
)
from bluemira.geometry.wire import BluemiraWire
from bluemira.utilities.positioning import PathInterpolator, PositionMapper


def make_solenoid(
    r_cs: float,
    tk_cs: float,
    z_min: float,
    z_max: float,
    g_cs: float,
    tk_cs_ins: float,
    tk_cs_cas: float,
    n_CS: int,
) -> List[Coil]:
    """
    Make a set of solenoid coils in an EU-DEMO fashion. If n_CS is odd, the central
    module is twice the size of the others. If n_CS is even, all the modules are the
    same size.

    Parameters
    ----------
    r_cs:
        Radius of the solenoid
    tk_cs:
        Half-thickness of the solenoid in the radial direction (including insulation and
        casing)
    z_min:
        Minimum vertical position of the solenoid
    z_max:
        Maximum vertical position of the solenoid
    g_cs:
        Gap between modules
    tk_cs_ins:
        Insulation thickness around modules
    tk_cs_cas:
        Casing thickness around modules
    n_CS:
        Number of modules in the solenoid

    Returns
    -------
    List of solenoid coil(s)
    """

    def make_CS_coil(z_coil, dz_coil, i):
        return Coil(
            r_cs,
            z_coil,
            current=0,
            dx=tk_cs - tk_inscas,
            dz=dz_coil,
            ctype="CS",
            name=f"CS_{i+1}",
        )

    if z_max < z_min:
        z_min, z_max = z_max, z_min
    if np.isclose(z_max, z_min):
        raise BuilderError(f"Cannot make a solenoid with z_min==z_max=={z_min}")

    total_height = z_max - z_min
    tk_inscas = tk_cs_ins + tk_cs_cas
    total_gaps = (n_CS - 1) * g_cs + n_CS * 2 * tk_inscas
    if total_gaps >= total_height:
        raise BuilderError(
            "Cannot make a solenoid where the gaps and insulation + casing are larger"
            " than the height available."
        )

    coils = []
    if n_CS == 1:
        # Single CS module solenoid (no gaps)
        module_height = total_height - 2 * tk_inscas
        coil = make_CS_coil(0.5 * total_height, 0.5 * module_height, 0)
        coils.append(coil)

    elif n_CS % 2 == 0:
        # Equally-spaced CS modules for even numbers of CS coils
        module_height = (total_height - total_gaps) / n_CS
        dz_coil = 0.5 * module_height
        z_iter = z_max
        for i in range(n_CS):
            z_coil = z_iter - tk_inscas - dz_coil
            coil = make_CS_coil(z_coil, dz_coil, i)
            coils.append(coil)
            z_iter = z_coil - dz_coil - tk_inscas - g_cs

    else:
        # Odd numbers of modules -> Make a central module that is twice the size of the
        # others.
        module_height = (total_height - total_gaps) / (n_CS + 1)
        z_iter = z_max
        for i in range(n_CS):
            if i == n_CS // 2:
                # Central module
                dz_coil = module_height
                z_coil = z_iter - tk_inscas - dz_coil

            else:
                # All other modules
                dz_coil = 0.5 * module_height
                z_coil = z_iter - tk_inscas - dz_coil

            coil = make_CS_coil(z_coil, dz_coil, i)
            coils.append(coil)
            z_iter = z_coil - dz_coil - tk_inscas - g_cs

    return coils


def _get_intersections_from_angles(boundary, ref_x, ref_z, angles):
    n_angles = len(angles)
    x_c, z_c = np.zeros(n_angles), np.zeros(n_angles)
    for i, angle in enumerate(angles):
        line = make_polygon(
            [
                [ref_x, ref_x + VERY_BIG * np.cos(angle)],
                [0, 0],
                [ref_z, ref_z + VERY_BIG * np.sin(angle)],
            ]
        )
        _, intersection = distance_to(boundary, line)
        x_c[i], _, z_c[i] = intersection[0][0]
    return x_c, z_c


def make_PF_coil_positions(tf_boundary, n_PF, R_0, kappa, delta):
    """
    Make a set of PF coil positions crudely with respect to the intended plasma
    shape.
    """
    # Project plasma centroid through plasma upper and lower extrema
    angle_upper = np.arctan2(kappa, -delta)
    angle_lower = np.arctan2(-kappa, -delta)
    scale = 1.1

    angles = np.linspace(scale * angle_upper, scale * angle_lower, n_PF)
    return _get_intersections_from_angles(tf_boundary, R_0, 0.0, angles)


def make_coilset(
    tf_boundary: BluemiraWire,
    R_0: float,
    kappa: float,
    delta: float,
    r_cs: float,
    tk_cs: float,
    g_cs: float,
    tk_cs_ins: float,
    tk_cs_cas: float,
    n_CS: int,
    n_PF: int,
    CS_jmax: float,
    CS_bmax: float,
    PF_jmax: float,
    PF_bmax: float,
) -> CoilSet:
    """
    Make an initial EU-DEMO-like coilset.
    """
    bb = tf_boundary.bounding_box
    z_min = bb.z_min
    z_max = bb.z_max
    solenoid = make_solenoid(r_cs, tk_cs, z_min, z_max, g_cs, tk_cs_ins, tk_cs_cas, n_CS)
    for s in solenoid:
        s.fix_size()

    tf_track = offset_wire(tf_boundary, 1)
    x_c, z_c = make_PF_coil_positions(
        tf_track,
        n_PF,
        R_0,
        kappa,
        delta,
    )
    pf_coils = []
    for i, (x, z) in enumerate(zip(x_c, z_c)):
        coil = Coil(
            x,
            z,
            current=0,
            ctype="PF",
            name=f"PF_{i+1}",
            j_max=PF_jmax,
            b_max=PF_bmax,
        )
        pf_coils.append(coil)
    coilset = CoilSet(*pf_coils + solenoid, control_names=True)
    coilset.assign_material("PF", j_max=PF_jmax, b_max=PF_bmax)
    coilset.assign_material("CS", j_max=CS_jmax, b_max=CS_bmax)
    return coilset


def make_reference_coilset(
    tf_track: BluemiraWire,
    lcfs_shape: BluemiraWire,
    r_cs: float,
    tk_cs: float,
    g_cs_mod: float,
    tk_cs_casing: float,
    tk_cs_insulation: float,
    n_CS: int,
    n_PF: int,
) -> CoilSet:
    """
    Make a reference coilset.
    """
    bb = tf_track.bounding_box
    z_min = bb.z_min
    z_max = bb.z_max
    solenoid = make_solenoid(
        r_cs,
        tk_cs,
        z_min,
        z_max,
        g_cs_mod,
        tk_cs_ins=tk_cs_insulation,
        tk_cs_cas=tk_cs_casing,
        n_CS=n_CS,
    )

    lcfs_coords = lcfs_shape.discretize(byedges=True)
    arg_z_max = np.argmax(lcfs_coords.z)
    arg_z_min = np.argmin(lcfs_coords.z)

    r_mid = 0.5 * (np.min(lcfs_coords.x) + np.max(lcfs_coords.x))
    d_delta_u = lcfs_coords.x[arg_z_max] - r_mid
    d_kappa_u = lcfs_coords.z[arg_z_max]
    d_delta_l = lcfs_coords.x[arg_z_min] - r_mid
    d_kappa_l = lcfs_coords.z[arg_z_min]

    angle_upper = np.arctan2(d_kappa_u, d_delta_u)
    angle_lower = np.arctan2(d_kappa_l, d_delta_l)
    angles = np.linspace(angle_upper, angle_lower, n_PF)
    x_c, z_c = _get_intersections_from_angles(tf_track, r_mid, 0.0, angles)

    pf_coils = []
    for i, (x, z) in enumerate(zip(x_c, z_c)):
        coil = Coil(
            x,
            z,
            current=0,
            ctype="PF",
            name=f"PF_{i+1}",
            j_max=100.0e6,
        )
        pf_coils.append(coil)
    return CoilSet(*pf_coils + solenoid, control_names=True)


def make_coil_mapper(
    track: BluemiraWire, exclusion_zones: List[BluemiraFace], coils: List[Coil]
) -> PositionMapper:
    """
    Make a PositionMapper for the given coils.

    Break a track down into individual interpolator segments
    incorporating exclusion zones and mapping tracks to coils.

    Parameters
    ----------
    track:
        Full length interpolator track for PF coils
    exclusion_zones:
        List of exclusion zones
    coils:
        List of coils

    Returns
    -------
    Position mapper for coil position interpolation

    Notes
    -----
    TODO use coilset directly instead of list of coils
    """
    # Break down the track into subsegments
    segments = boolean_cut(track, exclusion_zones) if exclusion_zones else [track]

    # Sort the coils into the segments
    coil_bins = [[] for _ in range(len(segments))]
    for coil in coils:
        distances = [distance_to([coil.x, 0, coil.z], seg)[0] for seg in segments]
        coil_bins[np.argmin(distances)].append(coil)

    # Check if multiple coils are on the same segment and split the segments and make
    # PathInterpolators
    interpolator_dict = {}
    for segment, _bin in zip(segments, coil_bins):
        if len(_bin) < 1:
            bluemira_warn("There is a segment of the track which has no coils on it.")
        elif len(_bin) == 1:
            coil = _bin[0]
            interpolator_dict[coil.name] = PathInterpolator(segment)
        else:
            coils = _bin
            l_values = np.array(
                [segment.parameter_at([c.x, 0, c.z], tolerance=VERY_BIG) for c in coils]
            )
            idx = np.argsort(l_values)
            l_values = l_values[idx]
            split_values = l_values[:-1] + 0.5 * np.diff(l_values)
            split_positions = [segment.value_at(alpha=split) for split in split_values]

            sub_segs = _split_segment(segment, split_positions)

            # Sorted coils
            for coil, sub_seg in zip([coils[i] for i in idx], sub_segs):
                interpolator_dict[coil.name] = PathInterpolator(sub_seg)

    return PositionMapper(interpolator_dict)


def _split_segment(segment, split_positions):
    """
    Split a segment into sub-segments at various split positions
    """
    sub_segs = []
    for split_pos in split_positions:
        split = segment.parameter_at(split_pos, tolerance=10 * EPS)
        sub_seg_1, segment = split_wire(
            segment, segment.value_at(alpha=split), tolerance=10 * EPS
        )
        if sub_seg_1:
            sub_segs.append(sub_seg_1)
        else:
            bluemira_warn("Sub-segment of 0 length!")
    sub_segs.append(segment)
    return sub_segs


def make_pf_coil_path(tf_boundary: BluemiraWire, offset_value: float) -> BluemiraWire:
    """
    Make an open wire along which the PF coils can move.

    Parameters
    ----------
    tf_boundary:
        Outside edge of the TF coil in the x-z plane
    offset_value:
        Offset value from the TF coil edge

    Returns
    -------
    Path along which the PF coil centroids should be positioned
    """
    tf_offset = offset_wire(tf_boundary, offset_value)

    # Find top-left and bottom-left "corners"
    coordinates = tf_offset.discretize(byedges=True, ndiscr=200)
    x_min = np.min(coordinates.x)
    z_min, z_max = 0.0, 0.0
    eps = 0.0
    while np.isclose(z_min, z_max):
        # This is unlikely, but if so, shifting x_min a little ensures the boolean cut
        # can be performed and that an open wire will be returned
        idx_inner = np.nonzero(np.isclose(coordinates.x, x_min))[0]
        z_min = np.min(coordinates.z[idx_inner])
        z_max = np.max(coordinates.z[idx_inner])
        x_min += eps
        eps += 1e-3

    cutter = BluemiraFace(
        make_polygon(
            {"x": [0, x_min, x_min, 0], "z": [z_min, z_min, z_max, z_max]}, closed=True
        )
    )

    result = boolean_cut(tf_offset, cutter)
    if len(result) > 1:
        bluemira_warn(
            "Boolean cut of the TF boundary resulted in more than one wire.. returning"
            " the longest one. Fingers crossed."
        )
        result.sort(key=lambda wire: -wire.length)
    return result[0]
