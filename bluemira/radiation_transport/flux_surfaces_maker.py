# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
A simplified 2-D solver for calculating charged particle heat loads.
"""

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

from bluemira.base.constants import EPS
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.find import find_flux_surface_through_point
from bluemira.equilibria.find_legs import LegFlux, NumNull, SortSplit
from bluemira.equilibria.flux_surfaces import OpenFluxSurface
from bluemira.geometry.coordinates import Coordinates, coords_plane_intersect
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.wire import BluemiraWire
from bluemira.radiation_transport.error import RadiationTransportError

if TYPE_CHECKING:
    from numpy import typing as npt

    from bluemira.equilibria.flux_surfaces import PartialOpenFluxSurfaces

__all__ = ["analyse_first_wall_flux_surfaces"]


def analyse_first_wall_flux_surfaces(
    equilibrium: Equilibrium, first_wall: BluemiraWire, dx_mp: float
):
    """
    A simplified charged particle transport model along open field lines.

    Parameters
    ----------
    equilibrium:
        The equilibrium defining flux surfaces.
    first_wall:
    dx_mp:
        The midplane spatial resolution between flux surfaces [m]
        (default: 0.001).
    """
    o_point = equilibrium.get_OX_points()[0][0]  # 1st o_point
    z = o_point.z
    yz_plane = BluemiraPlane.from_3_points([0, 0, z], [1, 0, z], [1, 1, z])

    first_wall = _process_first_wall(first_wall)

    if equilibrium.is_double_null:
        dx_omp, dx_imp, flux_surfaces, x_sep_omp, x_sep_imp = _analyse_DN(
            first_wall, dx_mp, equilibrium, o_point, yz_plane
        )
    else:
        dx_omp, flux_surfaces, x_sep_omp = _analyse_SN(
            first_wall, dx_mp, equilibrium, o_point, yz_plane
        )
        dx_imp = None
        x_sep_imp = None
    return dx_omp, dx_imp, flux_surfaces, x_sep_omp, x_sep_imp


def _process_first_wall(first_wall: Coordinates) -> Coordinates:
    """
    Force working first wall geometry to be closed and counter-clockwise.
    """
    first_wall = deepcopy(first_wall)

    if not first_wall.check_ccw(axis=[0, 1, 0]):
        bluemira_warn("First wall should be oriented counter-clockwise. Reversing it.")
        first_wall.reverse()

    if not first_wall.closed:
        bluemira_warn("First wall should be a closed geometry. Closing it.")
        first_wall.close()
    return first_wall


def _analyse_SN(first_wall, dx_mp, equilibrium, o_point, yz_plane):
    """
    Calculation for the case of single nulls.
    """
    x_sep_omp, x_out_omp = _get_sep_out_intersection(
        equilibrium, first_wall, yz_plane, outboard=True
    )

    flux_surfaces_ob = _make_flux_surfaces_ibob(
        dx_mp, equilibrium, o_point, yz_plane, x_sep_omp, x_out_omp, outboard=True
    )

    return (
        get_array_x_mp(flux_surfaces_ob[0]) - x_sep_omp,  # Calculate values at OMP
        _clip_flux_surfaces(  # Find flux surface intersections with the first wall
            first_wall, flux_surfaces_ob
        ),
        x_sep_omp,
    )


def _analyse_DN(first_wall, dx_mp, equilibrium: Equilibrium, o_point, yz_plane):
    """
    Calculation for the case of double nulls.
    """
    x_sep_omp, x_out_omp = _get_sep_out_intersection(
        equilibrium, first_wall, yz_plane, outboard=True
    )
    x_sep_imp, x_out_imp = _get_sep_out_intersection(
        equilibrium, first_wall, yz_plane, outboard=False
    )

    flux_surfaces_ob = _make_flux_surfaces_ibob(
        dx_mp, equilibrium, o_point, yz_plane, x_sep_omp, x_out_omp, outboard=True
    )
    flux_surfaces_ib = _make_flux_surfaces_ibob(
        dx_mp, equilibrium, o_point, yz_plane, x_sep_imp, x_out_imp, outboard=False
    )
    return (
        get_array_x_mp(flux_surfaces_ob[0]) - x_sep_omp,  # Calculate values at OMP
        abs(get_array_x_mp(flux_surfaces_ib[0]) - x_sep_imp),  # Calculate values at IMP
        _clip_flux_surfaces(  # Find flux surface intersections with the first wall
            first_wall,
            [*flux_surfaces_ob, *flux_surfaces_ib],
        ),
        x_sep_omp,
        x_sep_imp,
    )


def _clip_flux_surfaces(
    first_wall: Coordinates, flux_surfaces: list[PartialOpenFluxSurfaces]
) -> list[PartialOpenFluxSurfaces]:
    """
    Clip the flux surfaces to a first wall. Catch the cases where no intersections
    are found.

    Returns
    -------
    flux_surfaces:
        A list of flux surface groups. Each group only contains flux surfaces that
        intersect the first_wall.
    """
    for group in flux_surfaces:
        if group:
            for i, flux_surface in enumerate(group):
                flux_surface.clip(first_wall)
                if flux_surface.alpha is None:
                    # No intersection detected between flux surface and first wall
                    # Drop the flux surface from the group
                    group.pop(i)  # noqa: B909
                    # TODO: fix this ^ B909 error
    return flux_surfaces


def get_array_x_mp(flux_surfaces) -> npt.NDArray[float]:
    """
    Get x_mp array of flux surface values.

    Returns
    -------
    :
        np.array of x_mp.

    """
    return np.array([fs.x_start for fs in flux_surfaces])


def get_array_z_mp(flux_surfaces) -> npt.NDArray[float]:
    """
    Get z_mp array of flux surface values.

    Returns
    -------
    :
        np.array of z_mp.
    """
    return np.array([fs.z_start for fs in flux_surfaces])


def get_array_x_fw(flux_surfaces) -> npt.NDArray[float]:
    """
    Get x_fw array of flux surface values.

    Returns
    -------
    :
        np.array of x_fw.
    """
    return np.array([fs.x_end for fs in flux_surfaces])


def get_array_z_fw(flux_surfaces) -> npt.NDArray[float]:
    """
    Get z_fw array of flux surface values.

    Returns
    -------
    :
        np.array of z_fw.
    """
    return np.array([fs.z_end for fs in flux_surfaces])


def get_array_alpha(flux_surfaces) -> npt.NDArray[float]:
    """
    Get alpha angle array of flux surface values.

    Returns
    -------
    :
        np.array of alpha.
    """
    return np.array([fs.alpha for fs in flux_surfaces])


def _get_sep_out_intersection(eq: Equilibrium, first_wall, yz_plane, *, outboard=True):
    """
    Find the middle and maximum inboard/outboard mid-plane psi norm values.

    Raises
    ------
    RadiationTransportError
        Separatrix doesnt cross midplane
    """
    sep = LegFlux(eq)

    if sep.n_null == NumNull.SN:
        sep_intersections = coords_plane_intersect(sep.separatrix, yz_plane)
        sep_arg = np.argmin(np.abs(sep_intersections.T[0] - sep.o_point.x))
        x_sep_mp = sep_intersections.T[0][sep_arg]
    elif sep.sort_split == SortSplit.X:
        sep1_intersections = coords_plane_intersect(sep.separatrix[0], yz_plane)
        sep2_intersections = coords_plane_intersect(sep.separatrix[1], yz_plane)
        sep1_arg = np.argmin(np.abs(sep1_intersections.T[0] - sep.o_point.x))
        sep2_arg = np.argmin(np.abs(sep2_intersections.T[0] - sep.o_point.x))
        x_sep1_mp = sep1_intersections.T[0][sep1_arg]
        x_sep2_mp = sep2_intersections.T[0][sep2_arg]
        x_sep_mp = max(x_sep2_mp, x_sep1_mp) if outboard else min(x_sep2_mp, x_sep1_mp)
    else:
        # separatrix list is sorted by loop length when found,
        # so separatrix[0] will have the intersection
        sep_intersections = coords_plane_intersect(sep.separatrix, yz_plane)
        if isinstance(sep_intersections, Coordinates):
            sep_arg = np.argmin(np.abs(sep_intersections.T[0] - sep.o_point.x))
            x_sep_mp = sep_intersections.T[0][sep_arg]
        else:
            raise RadiationTransportError("Your seperatrix does not cross the midplane.")

    out_intersections = coords_plane_intersect(first_wall, yz_plane)
    x_out_mp = (
        np.max(out_intersections.T[0]) if outboard else np.min(out_intersections.T[0])
    )

    return x_sep_mp, x_out_mp


def _make_flux_surfaces(x, z, equilibrium, o_point, yz_plane):
    """
    Make individual PartialOpenFluxSurfaces through a point.

    Returns
    -------
    :
        The PartialOpenFluxSurfaces that passes through the point.
    """
    coords = find_flux_surface_through_point(
        equilibrium.x, equilibrium.z, equilibrium.psi(), x, z, equilibrium.psi(x, z)
    )
    return OpenFluxSurface(Coordinates({"x": coords[0], "z": coords[1]})).split(
        o_point, plane=yz_plane
    )


def _make_flux_surfaces_ibob(
    dx_mp, equilibrium, o_point, yz_plane, x_sep_mp, x_out_mp, *, outboard: bool
) -> tuple[list]:
    """
    Make the flux surfaces on the inboard or outboard.

    Returns
    -------
    flux_surfaces_lfs:
        inboard flux surfaces
    flux_surfaces_hfs:
        outboard flux surfaces
    """
    sign = 1 if outboard else -1

    flux_surfaces_lfs = []
    flux_surfaces_hfs = []

    for x in np.arange(
        x_sep_mp + (sign * dx_mp), x_out_mp - (sign * EPS), (sign * dx_mp)
    ):
        lfs, hfs = _make_flux_surfaces(x, o_point.z, equilibrium, o_point, yz_plane)
        flux_surfaces_lfs.append(lfs)
        flux_surfaces_hfs.append(hfs)
    return flux_surfaces_lfs, flux_surfaces_hfs
