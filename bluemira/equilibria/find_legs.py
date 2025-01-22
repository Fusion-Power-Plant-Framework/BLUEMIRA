# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Class and Methods for separatrix legs.
"""

import operator
from enum import Enum, auto

import numpy as np
import numpy.typing as npt

from bluemira.base.error import BluemiraError
from bluemira.base.look_and_feel import bluemira_print
from bluemira.equilibria.equilibrium import Equilibrium, Grid
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.find import (
    find_LCFS_separatrix,
    find_flux_surface_through_point,
    find_flux_surfs,
)
from bluemira.equilibria.flux_surfaces import (
    OpenFluxSurface,
    PartialOpenFluxSurface,
    calculate_connection_length_flt,
    calculate_connection_length_fs,
)
from bluemira.geometry.coordinates import Coordinates, get_intersect, join_intersect


class NumNull(Enum):
    """Class for use with LegFlux."""

    DN = auto()
    """Double Null"""
    SN = auto()
    """Single Null"""


class SortSplit(Enum):
    """Class for use with LegFlux."""

    X = auto()
    """Split the flux in x-direction"""
    Z = auto()
    """Split the flux in z-direction"""


class LegFlux:
    """
    Class for those pesky separatrix legs.

    Parameters
    ----------
    eq:
        Input Equilibrium
    psi_n_tol:
        The normalised psi tolerance to use
    delta_start:
        Search range value for finding LCFS. Will search for the transition from a
        "closed" to "open" flux surface for normalised flux values
        between 1 - delta_start and 1 + delta_start.
    rtol:
        Relative tolerance used for finding configuration of
        separatrix split for double null
    """

    def __init__(
        self,
        eq: Equilibrium,
        psi_n_tol: float = 1e-6,
        delta_start: float = 0.01,
        rtol: float = 1e-3,
    ):
        self.eq = eq
        o_points, x_points = eq.get_OX_points()
        lcfs, self.separatrix = find_LCFS_separatrix(
            eq.x,
            eq.z,
            eq.psi(),
            o_points,
            x_points,
            double_null=eq.is_double_null,
            psi_n_tol=psi_n_tol,
            delta_start=delta_start,
        )
        self.o_point = o_points[0]
        self.x_points = x_points[:2]
        self.rtol = rtol
        self.x_range_lcfs = [min(lcfs.x), max(lcfs.x)]
        self.delta = np.max(eq.grid.x) - np.min(eq.grid.x)
        self.delta_offsets = eq.grid.dx
        self.dx_offsets = None
        self.n_null, self.sort_split = self.which_legs()

    def which_legs(self):
        """
        Determine how to find and sort legs.
        For a double null this function:
        - sorts the x-points by lower then upper
        - keeps the separatrix list sorted by longest then shortest

        Returns
        -------
        n_null:
            Weather it is single (SN) or double null (DN) Configuration
        sort_split:
            How the separatrix has been split.
            Z - split into Upper and Lower
            X - split into Inner and Outer

        """
        # --- Double Null ---
        if isinstance(self.separatrix, list):
            # Sort LOWER then UPPER
            self.x_points.sort(key=lambda x_point: x_point.z)
            # Check to determine configuration (separatrix list is sorted by
            # loop length when it is found (longest first), so use [0])
            z0 = self.separatrix[0].z[
                (self.separatrix[0].x > self.x_range_lcfs[0])
                & (self.separatrix[0].x < self.x_range_lcfs[1])
            ]
            z1 = self.separatrix[1].z[
                (self.separatrix[1].x > self.x_range_lcfs[0])
                & (self.separatrix[1].x < self.x_range_lcfs[1])
            ]
            legs_upper_lower = (max(z0) < min(z1)) or (min(z0) > max(z1))
            if legs_upper_lower:
                # Sort LOWER then UPPER when use get_legs
                # self.separatrix remains sorted by loop length
                return NumNull.DN, SortSplit.Z
            # Sort IN then OUT when use get_legs
            # self.separatrix remains sorted by loop length
            return NumNull.DN, SortSplit.X
        # --- Single Null ---
        self.x_points = self.x_points[0]
        return NumNull.SN, SortSplit.X

    def get_leg_offsets(self, leg_dict):
        """
        Expands the leg list if user requires offset flux surfaces.
        """  # noqa: DOC201
        for name in leg_dict:
            leg = leg_dict[name]
            direction = -1 if name.find("inner") != -1 else 1
            if len(leg) > 0 and leg[0] is not None:
                    leg.extend(
                        _extract_offsets(
                            self.eq,
                            leg[0],
                            direction,
                            self.o_point,
                            self.dx_offsets,
                            self.delta_offsets,
                        )
                    )
                    leg_dict[name] = leg
        return leg_dict

    def get_legs(
        self,
        n_layers: int = 1,
        dx_off: float = 0.0,
        delta: float | None = None,
        delta_offsets: float | None = None,
    ):
        """
        Get separatrix legs.

        Parameters
        ----------
        n_layers:
            Number of flux surfaces to extract for each leg
        dx_off:
            Total span in radial space of the flux surfaces to extract
        delta:
            intersection point x value +- delta is used to find starting point
            of leg flux see '_extract_leg'.
        delta_offsets:
            intersection point x value +- delta_offsets is used to find starting point
            of offsets leg flux see '_extract_offsets'.

        Returns
        -------
        leg_dict:
            Dictionary of separatrix legs (lists of coordinates), with keys:
            - lower_inner
            - lower_outer
            - upper_inner
            - upper_outer

        Raises
        ------
        EquilibriaError: if a strange number of legs would be found for an X-point

        Notes
        -----
        Will return two legs in the case of a single null (an upper or lower pair).
        Will return four legs in the case of a double null.

        We can't rely on the X-point being contained within the two legs, due to
        interpolation and local minimum finding tolerances.

        """
        if delta is not None:
            self.delta = delta
        if delta_offsets is not None:
            self.delta_offsets = delta_offsets
        if n_layers > 1:
            self.dx_offsets = np.linspace(0, dx_off, n_layers)[1:]
        "Get the legs of a separatrix."
        # --- Single Null ---
        if self.n_null == NumNull.SN:
            leg_dict = get_single_null_legs(
                self.separatrix,
                self.delta,
                self.o_point,
                self.x_points,
            )
            if self.dx_offsets is not None:
                return self.get_leg_offsets(leg_dict)
            return leg_dict

        # --- Double Null ---
        if self.sort_split == SortSplit.Z:
            leg_dict = get_legs_double_null_zsplit(
                self.separatrix,
                self.delta,
                self.x_points,
                self.o_point,
                self.x_range_lcfs,
            )
            if self.dx_offsets is not None:
                return self.get_leg_offsets(leg_dict)
            return leg_dict

        leg_dict = get_legs_double_null_xsplit(
            self.separatrix,
            self.delta,
            self.x_points,
            self.o_point,
        )
        if self.dx_offsets is not None:
            return self.get_leg_offsets(leg_dict)
        return leg_dict


def get_legs_length_and_angle(
    eq: Equilibrium,
    leg_dict: dict[str, npt.NDArray[np.float64] | None],
    plasma_facing_boundary: Grid | Coordinates | None = None,
):
    """Calculates the length of all the divertor legs in a dictionary.

    Returns
    -------
    :
        leg length dictionary for a given leg
    :
        the angle dictionary for a given leg
    """
    length_dict = {}
    angle_dict = {}
    for name, leg_list in leg_dict.items():
        lengths = []
        angles = []
        for leg in leg_list:
            if leg is None:
                con_length = 0.0
                grazing_ang = np.pi
            else:
                leg_fs = PartialOpenFluxSurface(leg)
                if plasma_facing_boundary is not None:
                    leg_fs.clip(plasma_facing_boundary)
                con_length = OpenFluxSurface(leg_fs.coords).connection_length(eq)
                alpha = leg_fs.alpha
                if alpha is None:
                    grazing_ang = np.pi
                elif alpha <= 0.5 * np.pi:
                    grazing_ang = alpha
                else:
                    grazing_ang = np.pi - alpha
            lengths.append(con_length)
            angles.append(grazing_ang)
        length_dict.update({name: lengths})
        angle_dict.update({name: angles})
    return length_dict, angle_dict


def get_single_null_legs(separatrix, delta, o_point, x_points, imin=None):
    """
    Returns
    -------
    :
        The legs from a single null separatrix as a dictionary.
    """
    sorted_legs = get_leg_list(
        separatrix,
        delta,
        o_p=o_point,
        x_p=x_points,
        imin=imin,
    )
    return add_pair_to_dict(sorted_legs, x_p=x_points, o_p=o_point)


def get_legs_double_null_xsplit(separatrix, delta, x_points, o_point):
    """
    Returns
    -------
    :
        The legs from a double null separatrix, split in x-direction,
        as a dictionary.

    """
    # Separatrix list is sorted by INNER then OUTER
    separatrix.sort(key=lambda separatrix: separatrix.x[0])
    legs = []
    for half_sep in separatrix:
        for x_p in x_points:
            leg = get_leg_list(half_sep, delta, o_p=o_point, x_p=x_p)
            legs.append(leg)
    return {
        "lower_inner": legs[0],
        "upper_inner": legs[1],
        "lower_outer": legs[2],
        "upper_outer": legs[3],
    }


def get_legs_double_null_zsplit(separatrix, delta, x_points, o_point, x_range_lcfs):
    """
    Returns
    -------
    :
        The legs from a double null separatrix, split in z-direction,
        as a dictionary.

    """
    # Separatrix list is sorted by loop length
    # Get indices for sorting LOWER then UPPER
    min_z = [np.min(separatrix[0].z), np.min(separatrix[1].z)]
    l_u_idx = np.argsort(min_z)

    # First, handle the flux surface that contains the plasma
    plasma_encompass_pair = get_leg_list(
        separatrix[0],
        delta,
        o_p=o_point,
        x_p=x_points[0] if l_u_idx[0] == 0 else x_points[1],
    )
    leg_dict = add_pair_to_dict(
        plasma_encompass_pair,
        x_p=x_points[0] if l_u_idx[0] == 0 else x_points[1],
        o_p=o_point,
    )

    # Then we deal with the legs with the same normalised flux
    # as the plasma containing flux surface
    z_range = separatrix[1].z[
        (separatrix[1].x > x_range_lcfs[0]) & (separatrix[1].x < x_range_lcfs[1])
    ]
    i_range = np.arange(len(separatrix[1].z))[
        (separatrix[1].x > x_range_lcfs[0]) & (separatrix[1].x < x_range_lcfs[1])
    ]
    if z_range.any():
        i = np.argmin(np.abs(z_range))
        imin = i_range[i]
        no_plasma_legs = get_leg_list(
            separatrix[1],
            delta,
            o_p=o_point,
            imin=imin,
        )
    else:
        no_plasma_legs = None

    leg_dict.update(
        add_pair_to_dict(
            no_plasma_legs,
            x_p=x_points[1] if l_u_idx[0] == 0 else x_points[0],
            o_p=o_point,
        )
    )
    return leg_dict


def get_leg_list(leg_pair, delta, o_p, x_p=None, imin=None):
    """
    Extracts leg/s from given flux line and return as sorted list.
    Legs are sorted by inner then outer.

    Returns
    -------
    :
        the legs sort by in and out

    Raises
    ------
    BluemiraError
        if x_p and imin are both None

    """
    # Looking at flux surface with the same normalised flux as the
    # plasma containing flux surface (double null only)
    if imin is not None:
        legs = _extract_leg_using_index_value(leg_pair, imin)
    # Looking at flux surface contains the plasma (single or double null)
    elif x_p is not None:
        legs = _extract_leg(leg_pair, x_p.x, x_p.z, delta, o_p.z)
    else:
        raise BluemiraError("Please enter either a value for x_p or imin.")
    if legs is None:
        return None
    if isinstance(legs, Coordinates):
        return [legs]
    # Sort IN then OUT
    legs.sort(key=lambda leg: leg.x[0])
    return legs


def _extract_offsets(eq, ref_leg, direction, o_p, dx_offsets, delta_offsets) -> list:
    """
    Returns
    -------
    :
         Flux surfaces offset from separatix leg
    """
    offset_legs = []
    for dx in dx_offsets:
        x, z = ref_leg.x[0] + direction * dx, ref_leg.z[0]
        xl, zl = find_flux_surface_through_point(
            eq.x,
            eq.z,
            eq.psi(),
            x,
            z,
            eq.psi(x, z),
        )
        offset_legs.append(
            _extract_leg(Coordinates({"x": xl, "z": zl}), x, z, delta_offsets, o_p.z)
        )
    return offset_legs


def add_pair_to_dict(sorted_legs, x_p, o_p):
    """
    Convert a upper or lower pair of sorted legs into a dictionary.
    """  # noqa: DOC201
    location = "lower" if x_p.z < o_p.z else "upper"
    if sorted_legs is None:
        return {
            f"{location}_inner": [None],
            f"{location}_outer": [None],
        }
    if len(sorted_legs) <= 1:
        return {
            f"{location}_inner": [None],
            f"{location}_outer": [None],
        }
    # Legs always sorted IN then OUT
    return {
        f"{location}_inner": [sorted_legs[0]],
        f"{location}_outer": [sorted_legs[1]],
    }


def _extract_leg(
    flux_line: Coordinates,
    x_cut: float,
    z_cut: float,
    delta_x: float,
    o_point_z: float,
):
    """Extract legs from a flux surface using a chosen intersection point.

    Returns
    -------
    :
        the flux legs
    """
    radial_line = Coordinates({
        "x": [x_cut - delta_x, x_cut + delta_x],
        "z": [z_cut, z_cut],
    })
    arg_inters = join_intersect(flux_line, radial_line, get_arg=True)
    arg_inters.sort()
    # Lower null vs upper null
    func = operator.lt if z_cut < o_point_z else operator.gt

    if len(arg_inters) > 2:  # noqa: PLR2004
        EquilibriaError(
            "Unexpected number of intersections with the separatrix around the X-point."
        )

    flux_legs = []
    for arg in arg_inters:
        if func(flux_line.z[arg + 1], flux_line.z[arg]):
            leg = Coordinates(flux_line[:, arg:])
        else:
            leg = Coordinates(flux_line[:, : arg + 1])

        # Make the leg flow away from the plasma core
        if leg.argmin((x_cut, 0, z_cut)) > 3:  # noqa: PLR2004
            leg.reverse()

        flux_legs.append(leg)
    if len(flux_legs) == 1:
        flux_legs = flux_legs[0]
    return flux_legs


def _extract_leg_using_index_value(
    flux_line: Coordinates,
    i_cut: float,
):
    """
    Extract legs from a flux surface using
    an intersection point chosen by index value.

    Returns
    -------
    :
        The flux legs

    """
    leg1, leg2 = Coordinates(flux_line[:, i_cut:]), Coordinates(flux_line[:, :i_cut])
    check_len1, check_len2 = (
        (len(leg1.x) > len(flux_line.x) * 0.1),
        (len(leg2.x) > len(flux_line.x) * 0.1),
    )

    flux_legs = []
    if leg1.x.any() and check_len1:
        flux_legs.append(leg1)
    if leg2.x.any() and check_len2:
        # Make the leg flow away from the plasma core
        leg2.reverse()
        flux_legs.append(leg2)

    if len(flux_legs) == 1:
        flux_legs = flux_legs[0]
    if not flux_legs:
        flux_legs = None
    return flux_legs


class CalcMethod(Enum):
    """
    Class for use with calculate_connection_length function.
    User can choose how the connection length is calculated
    """

    FIELD_LINE_TRACER = auto()
    FLUX_SURFACE_GEOMETRY = auto()


def calculate_connection_length(
    eq: Equilibrium,
    div_target_start_point: Coordinates | None = None,
    first_wall: Coordinates | None = None,
    div_norm_psi: float | None = None,
    forward: bool = True,  # noqa: FBT001, FBT002
    psi_n_tol: float = 1e-6,
    delta_start: float = 0.01,
    rtol: float = 1e-1,
    n_turns_max: int = 50,
    n_points: int = 1000,
    calculation_method: str = "flux_surface_geometry",
):
    """
    Calculate the parallel connection length from a starting point to a flux-intercepting
    surface using either flux surface geometry or a field line tracer.

    User can choose an xz point or a normalised psi value to select a flux surface
    of interest - please NOTE, an input div_norm_psi will override an input
    div_target_start_point if both are entered.

    If no starting point is selected then use the separatrix at the Outboard Midplane.

    Returns
    -------
    :
        The connection length

    Raises
    ------
    BluemiraError
        If an invalid option calculation_method is selected.
        If an invalid div_norm_psi value is entered.
        If no target is provided for FLT calculation_method - this is because the
        flux interception point found is not accurate enough to be used
        on a seperatrix automatically found by bluemira (n.b., the FLT can not
        distingish between open and closed flux).

    """
    calculation_method = CalcMethod[calculation_method.upper()]

    if first_wall is None:
        x1, x2 = eq.grid.x_min, eq.grid.x_max
        z1, z2 = eq.grid.z_min, eq.grid.z_max
        first_wall = Coordinates({"x": [x1, x2, x2, x1, x1], "z": [z1, z1, z2, z2, z1]})

    # Use intersection between plasma facing surface and flux surface
    # with chosen normalised psi. Note: this will override an input
    # div_target_start_point.
    if div_norm_psi is not None:
        if div_norm_psi <= 1:
            raise BluemiraError("div_norm_psi value must be > 1.")
        op, xp = eq.get_OX_points()
        fs_lst = find_flux_surfs(eq.grid.x, eq.grid.z, eq.psi(), div_norm_psi, op, xp)
        coords = [Coordinates({"x": fs_arr.T[0], "z": fs_arr.T[1]}) for fs_arr in fs_lst]
        coords.sort(key=lambda coords: -coords.length)
        xcrss, zcrss = np.array([]), np.array([])
        for c in coords[:2]:
            x_int, z_int = get_intersect(c.xz, first_wall.xz)
            xcrss, zcrss = np.append(xcrss, x_int), np.append(zcrss, z_int)
        # Pick lower inner or outer corner for div crossing points
        # N.B. forward = True = LFS
        mid = np.median(eq.get_LCFS().x)
        xcond = (xcrss >= mid) if forward else (xcrss <= mid)
        xcrss, zcrss = xcrss[xcond & (zcrss <= 0.0)], zcrss[xcond & (zcrss <= 0.0)]
        if isinstance(xcrss, np.ndarray):
            # Sort by x crossing value
            xcrss, zcrss = zip(*sorted(zip(xcrss, zcrss, strict=False)), strict=False)
            # Choose lowest x crossing value if outer (forward=True) and highest if inner
            xcrss, zcrss = (xcrss[0], zcrss[0]) if forward else (xcrss[-1], zcrss[-1])
        div_target_start_point = Coordinates({"x": xcrss, "z": zcrss})

    # Use Separatrix (in BM is first 'open' fs) flux if div target point not chosen
    if div_target_start_point is None:
        if calculation_method == CalcMethod.FIELD_LINE_TRACER:
            raise BluemiraError(
                "Field line tracer method requires input div_target_start_point."
                "Please use flux surface geometry method or input a target location."
            )

        legflux = LegFlux(
            eq=eq,
            psi_n_tol=psi_n_tol,
            delta_start=delta_start,
            rtol=rtol,
        )

        if legflux.n_null == NumNull.DN:
            if legflux.sort_split == SortSplit.X:
                legflux.separatrix.sort(key=lambda leg: -leg.x[0])
            f_s = legflux.separatrix[0]
        else:
            f_s = legflux.separatrix

    else:
        xfs, zfs = find_flux_surface_through_point(
            eq.x,
            eq.z,
            eq.psi(),
            div_target_start_point.x,
            div_target_start_point.z,
            eq.psi(div_target_start_point.x, div_target_start_point.z),
        )
        f_s = Coordinates({"x": xfs, "z": zfs})

    if f_s.closed:
        bluemira_print("Flux surface is closed. No connection length calculated.")
        return 0.0

    # OMP is taken to be start point regardless of input
    z_abs = np.abs(f_s.z)
    z = np.min(z_abs)
    # If flux surface is not passing though midplane then do
    # not calculate connection length.
    midplane = np.isclose(z, 0.0, atol=eq.grid.dz)
    if not midplane:
        return 0.0
    x = np.max(f_s.x[z_abs == np.min(z_abs)])

    if calculation_method == CalcMethod.FIELD_LINE_TRACER:
        return calculate_connection_length_flt(
            eq=eq,
            x=x,
            z=z,
            forward=forward,
            n_points=n_points,
            first_wall=first_wall,
            n_turns_max=n_turns_max,
        )

    if calculation_method == CalcMethod.FLUX_SURFACE_GEOMETRY:
        return calculate_connection_length_fs(
            eq=eq,
            x=x,
            z=z,
            forward=forward,
            first_wall=first_wall,
            f_s=f_s,
        )

    raise BluemiraError("Please select a valid calculation_method option.")
