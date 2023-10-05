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
Coil positioning routines (automatic and adjustable)
"""

import re
from copy import deepcopy
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull

from bluemira.base.constants import EPS
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.equilibria.coils import Coil, CoilSet, get_max_current
from bluemira.equilibria.constants import NB3SN_J_MAX, NBTI_J_MAX
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.plotting import RegionPlotter, XZLPlotter
from bluemira.geometry._private_tools import offset
from bluemira.geometry.constants import VERY_BIG
from bluemira.geometry.coordinates import (
    Coordinates,
    coords_plane_intersect,
    get_area_2d,
    interpolate_midpoints,
    join_intersect,
    vector_lengthnorm,
)
from bluemira.geometry.error import GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.inscribed_rect import inscribed_rect_in_poly
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import boolean_cut, boolean_fuse, make_polygon, offset_wire
from bluemira.utilities import tools


class CoilPositioner:
    """
    Initial coil positioning tools for ab initio equilibrium design

    Parameters
    ----------
    R_0:
        Machine major radius [m]
    A:
        Plasma aspect ratio
    delta:
        Plasma triangularity
    kappa:
        Plasma elongation
    track:
        Track along which PF coils are positioned
    x_cs:
        Central Solenoid radius
    tk_cs:
        Central Solenoid thickness either side
    n_PF:
        Number of PF coils
    n_CS:
        Number of CS modules
    csgap:
        The gap between CS modules [m]
    rtype:
        The type of reactor ['ST', 'Normal']. Used for default coil positioning
    cslayout:
        The layout of the CS modules ['ITER', 'DEMO']
    """

    def __init__(
        self,
        R_0: float,
        A: float,
        delta: float,
        kappa: float,
        track: Coordinates,
        x_cs: float,
        tk_cs: float,
        n_PF: int,
        n_CS: int,
        csgap: float = 0.1,
        rtype: str = "Normal",
        cslayout: str = "DEMO",
    ):
        self.ref = [R_0, 0]
        self.A = A
        self.R_0 = R_0
        self.delta = delta
        self.kappa = kappa
        self.track = track
        self.x_cs = x_cs
        self.tk_cs = tk_cs
        self.n_PF = n_PF
        self.n_CS = n_CS
        self.csgap = csgap
        self.rtype = rtype
        self.cslayout = cslayout

    def equispace_PF(self, track: Coordinates, n_PF: int) -> List[Coil]:
        """
        Equally spaces PF coils around a TF coil boundary track, picking
        some starting positions for the uppermost and lowermost PF coil
        based on plasma shape considerations (mirror about X-points)
        """
        a = np.rad2deg(np.arctan(abs(self.delta) / self.kappa))
        if self.rtype == "Normal":
            angle_upper = 90 + a * 1.6
            angle_lower = -90 - a * 1.6
        elif self.rtype == "ST":
            angle_upper = 90 + a * 1.2
            angle_lower = -90 - a * 1.0

        angle = np.radians(angle_lower)

        line = Coordinates(
            {
                "x": [self.ref[0], self.ref[0] + VERY_BIG * np.cos(angle)],
                "z": [self.ref[1], self.ref[1] + VERY_BIG * np.sin(angle)],
            }
        )

        arg_lower = join_intersect(track, line, get_arg=True)

        angle = np.radians(angle_upper)

        line = Coordinates(
            {
                "x": [self.ref[0], self.ref[0] + VERY_BIG * np.cos(angle)],
                "z": [self.ref[1], self.ref[1] + VERY_BIG * np.sin(angle)],
            }
        )

        arg_upper = join_intersect(track, line, get_arg=True)

        arg_lower = arg_lower[0] if arg_lower else 0

        arg_upper = arg_upper[0] if arg_upper else len(track) - 1

        tf_loop = Coordinates(track[:, arg_lower : arg_upper + 1])
        l_norm = vector_lengthnorm(tf_loop.x, tf_loop.z)
        pos = np.linspace(0, 1, n_PF)
        xint, zint = interp1d(l_norm, tf_loop.x)(pos), interp1d(l_norm, tf_loop.z)(pos)
        return [
            Coil(xint[i], zint[i], ctype="PF", j_max=NBTI_J_MAX) for i in range(n_PF)
        ]

    def equispace_CS(
        self,
        x_cs: float,
        tk_cs: float,
        z_min: float,
        z_max: float,
        n_CS: int,
        j_max: float = NB3SN_J_MAX,
    ) -> List[Coil]:
        """
        Defines a Solenoid object with equally spaced nCS modules
        """
        dz = ((z_max - z_min) - self.csgap * (n_CS - 1)) / n_CS / 2
        v1 = np.arange(0, n_CS)
        v2 = np.arange(1, n_CS * 2, 2)
        zc = z_max - self.csgap * v1 - dz * v2

        return [
            Coil(
                x_cs,
                _zc,
                current=0,
                n_turns=1,
                ctype="CS",
                j_max=j_max,
                dx=tk_cs,
                dz=dz,
            )
            for _zc in zc
        ]

    def demospace_CS(
        self, x_cs: float, tk_cs: float, z_min: float, z_max: float, n_CS: int
    ) -> List[Coil]:
        """
        Defines a Solenoid object with DEMO like layout of nCS modules
        """
        if n_CS <= 2 or n_CS % 2 == 0:
            bluemira_warn(
                "So was kann ich mit einem DEMO-spacing nicht machen. "
                "Stattdessen gib ich dir einen ITER-spacing CS."
            )
            return self.equispace_CS(x_cs, tk_cs, z_min, z_max, n_CS)
        length = ((z_max - z_min) - (n_CS - 1) * self.csgap) / (
            n_CS + 1
        )  # Module length
        a = np.linspace(1, n_CS * 2 - 1, n_CS)
        a[n_CS // 2 :] += 2
        a[n_CS // 2] = n_CS + 1  # Central module
        b = np.linspace(0, n_CS - 1, n_CS)
        z_cs = z_max * np.ones(n_CS)
        z_cs -= a * length / 2 + b * self.csgap
        heights = length / 2 * np.ones(n_CS)
        heights[n_CS // 2] = length  # Central module
        return [
            Coil(
                x_cs,
                z_cs[i],
                dx=tk_cs,
                dz=heights[i],
                ctype="CS",
                j_max=NBTI_J_MAX,
            )
            for i in range(n_CS)
        ]

    def make_coilset(self, d_coil: float = 0.5) -> CoilSet:
        """
        Returns a CoilSet object
        """
        coils = self.equispace_PF(self.track, self.n_PF)
        z_max = max(self.track.z)
        z_min = -z_max
        if self.n_CS != 0:
            if self.cslayout == "ITER":
                coils.append(
                    self.equispace_CS(self.x_cs, self.tk_cs, z_min, z_max, self.n_CS)
                )
            elif self.cslayout == "DEMO":
                coils.extend(
                    self.demospace_CS(self.x_cs, self.tk_cs, z_min, z_max, self.n_CS)
                )
            else:
                raise ValueError(
                    f"Valid options are 'ITER' and 'DEMO', not '{self.cslayout}'"
                )
        cset = CoilSet(*coils)
        cset.discretisation = d_coil
        return cset


class XZLMapper:
    """
    Coil positioning tools for use in optimisation

    Parameters
    ----------
    pftrack:
        Track (x, z) along which PF coils are positioned
    cs_x:
        Radius of the centre of the central solenoid [m]
    cs_zmin:
        Minimum z location of the CS [m]
    cs_zmax:
        Maximum z location of the CS [m]
    cs_gap:
        Gap between modules of the CS [m]
    CS:
        Whether or not to XL map CS
    """

    def __init__(
        self,
        pf_coords: Coordinates,
        cs_x: float = 1.0,
        cs_zmin: float = 1.0,
        cs_zmax: float = 1.0,
        cs_gap: float = 0.1,
        CS: bool = False,
    ):
        while len(pf_coords) < 4:
            pf_coords = Coordinates(np.c_[interpolate_midpoints(*pf_coords.xyz)])

        self.pf_coords = deepcopy(pf_coords)  # Stored as loop too

        ln = vector_lengthnorm(pf_coords.x, pf_coords.z)

        x_ius = InterpolatedUnivariateSpline(ln, pf_coords.x)
        z_ius = InterpolatedUnivariateSpline(ln, pf_coords.z)
        self.pftrack = {
            "x": x_ius,
            "z": z_ius,
            "L": self.pf_coords.length,
            "dx": x_ius.derivative(),
            "dz": z_ius.derivative(),
        }

        self.flag_CS = CS
        if self.flag_CS:
            self.Xcs = cs_x
            self.z_min = cs_zmin
            self.z_max = cs_zmax
            self.gap = cs_gap
            self.make_cstrack()
        else:  # Due diligence
            self.Xcs = None
            self.z_min = None
            self.z_max = None
            self.gap = None
            self.cstrack = None

        self.exclusions = None
        self.excl_zones = []
        self.excl_loops = None
        self.incl_loops = None
        self._coilset = None  # PLotting utility

    def make_cstrack(self):
        """
        Make a normalised straight segment track for the central solenoid.
        """
        z = [self.z_max, self.z_min]
        self.cstrack = {"L": interp1d(z, [0, 1]), "z": interp1d([0, 1], z)}

    @staticmethod
    def PFnorm(l_values, coords, point):
        """
        Optimisation function for the positioning of the coils along the track.
        """
        return (coords["x"](l_values) - point[0]) ** 2 + (
            coords["z"](l_values) - point[1]
        ) ** 2

    def xz_to_L(self, x: float, z: float) -> float:  # noqa: N802
        """
        Translation of (x-z) coordinates to linear normalised coordinates (L) for the PF
        coils.
        """
        return minimize_scalar(
            self.PFnorm, method="bounded", args=(self.pftrack, [x, z]), bounds=[0, 1]
        ).x

    def L_to_xz(
        self, l_values: Union[float, np.ndarray]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:  # noqa: N802
        """
        Translation of linear normalised coordinates (L) to (x-z) coordinates for the PF
        coils.
        """
        return self.pftrack["x"](l_values), self.pftrack["z"](l_values)

    def z_to_L(self, zc_vec):  # noqa: N802
        """
        Convert z values for the CS in L values of the CS track.
        """
        zc_vec = np.sort(zc_vec)[::-1]
        if len(zc_vec) == 1:
            return np.array([0.5])
        z_edge = np.zeros(len(zc_vec))
        z_edge[0] = self.z_max - 2 * abs(self.z_max - zc_vec[0])
        for i in range(1, len(zc_vec) - 1):
            z_edge[i] = zc_vec[i] - (z_edge[i - 1] - zc_vec[i] - self.gap)
        z_edge[len(zc_vec) - 1] = self.z_min
        return self.cstrack["L"](z_edge)

    def L_to_zdz(self, l_values):
        """
        Convert L values for the CS track into z and dz values for the CS.
        """
        l_values = tools.clip(l_values, 0, 1)
        l_values = np.sort(l_values)
        z_edge = self.cstrack["z"](l_values)
        dz, zc = np.zeros(len(l_values)), np.zeros(len(l_values))
        dz[0] = abs(self.z_max - z_edge[0]) / 2
        zc[0] = self.z_max - dz[0]
        for i in range(1, len(l_values)):
            dz[i] = abs(z_edge[i - 1] - z_edge[i] - self.gap) / 2
            zc[i] = z_edge[i - 1] - dz[i] - self.gap
        # dz[-1] = abs(z_edge[-1]-self.Zmin-self.gap)/2
        # zc[-1] = self.Zmin+dz[-1]
        return self.Xcs * np.ones(len(l_values)), zc[::-1], dz[::-1]  # Coil numbering

    def get_Lmap(
        self, coilset: CoilSet, mapping: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # noqa: N802
        """
        Calculates initial L vector and lb and ub constraints on L vector.

        Parameters
        ----------
        coilset:
            The coilset to map
        mapping:
            List of PF coil names on the track

        Returns
        -------
        L:
            The initial position vector for the coilset position optimiser
        lb:
            The lower bounds on the L vector to be respected by the optimiser
        ub:
            The upper bounds on the L vector to be respected by the optimiser
        """
        self._coilset = coilset  # for plotting
        track_coils = len(mapping)
        l_values = np.zeros(track_coils)
        lb = np.zeros(track_coils)
        ub = np.zeros(track_coils)
        pf_coils = [self._coilset[m] for m in mapping]
        for i, coil in enumerate(pf_coils):
            loc = self.xz_to_L(coil.x, coil.z)
            if self.exclusions is not None:
                for ex in self.exclusions:
                    if ex[0] < loc < ex[1]:
                        back = -(
                            loc
                            - ex[0]
                            + 2 * coil._current_radius / self.pf_coords.length
                        )
                        forw = (
                            ex[1]
                            - loc
                            + 2 * coil._current_radius / self.pf_coords.length
                        )
                        if abs(back) >= abs(forw):
                            d_l = forw
                            break
                        else:
                            d_l = back
                            break
                    else:
                        d_l = 0
                l_values[i] = loc + d_l
                lb[i], ub[i] = self._get_bounds(l_values[i])
            else:
                l_values[i] = loc
                lb[i], ub[i] = 0, 1
        lb, ub = self._segment_tracks(lb, ub)
        # The L vector must be adjusted to its new bounds
        l_values = tools.clip(l_values, lb, ub)
        if self.flag_CS:
            n_CS = coilset.n_coils("CS")
            z = coilset.get_coiltype("CS").z
            l_values = np.append(l_values, self.z_to_L(np.sort(z)[::-1]))
            lb = np.append(lb, np.zeros(n_CS))
            ub = np.append(ub, np.ones(n_CS))
        return l_values, lb, ub

    def _get_bounds(self, l_values):
        """
        Generates an initial set of bounds for L based on the exclusion zones
        for the PF coils
        """
        e = [e for b in self.exclusions for e in b]
        lb, ub = 0, 1
        for ex in e:
            if l_values < ex:
                ub = ex
                break
            else:
                lb = ex
        return lb, ub

    @staticmethod
    def _segment_tracks(lb, ub):
        """
        Applies additional (silent) constraints, effectively chopping up a
        sub-track into two, so that two coils don't end up on top of each other
        """
        # beware of np.zeros_like!
        # TODO this feels temperamental
        lb_new, ub_new = np.zeros(len(lb)), np.zeros(len(ub))
        lb, ub = list(lb), list(ub)
        flag = False
        last_n = -1
        for i, (lower, upper) in enumerate(zip(lb, ub)):
            n = lb.count(lower)
            if i == last_n:
                flag = False
            if n == 1:  # No duplicates
                flag = False
                lb_new[i] = lower
                ub_new[i] = upper
            elif n != 1 and flag is False:
                flag = True
                last_n = i + n
                if last_n > len(lb_new):
                    continue
                delta = (upper - lower) / n
                for k, j in enumerate(range(i, i + n)):
                    lb_new[j] = upper - (k + 1) * delta
                    ub_new[j] = upper - k * delta
            else:
                continue
        return lb_new, ub_new

    def _get_unique_zone(self, zones: List[Coordinates]) -> BluemiraFace:
        """
        Makes a single "cutting" shape. This is a cheap way of avoiding a
        complicated merging list, checking for overlaps between zones.

        Parameters
        ----------
        zones:
            The list of exclusion zones

        Returns
        -------
        The boolean union of all the exclusion zones
        """
        self.excl_zones.extend(zones)

        offset_coords = offset(*self.pf_coords.xz, -0.0001)
        joiner = Coordinates({"x": offset_coords[0], "z": offset_coords[1]})
        joiner.close()

        joiner = BluemiraFace(make_polygon(joiner.xyz, closed=True))
        zones = [
            BluemiraFace(make_polygon(zone.xyz, closed=True)) for zone in self.excl_zones
        ]

        return boolean_fuse([joiner, *zones])

    def add_exclusion_zones(self, zones: List[Coordinates]):
        """
        Fügt der PFspulenbahn Aussschlusszonen hinzu

        Parameters
        ----------
        zones:
            List of Coordinates exclusion zones in x, z coordinates
        """
        excl_zone = self._get_unique_zone(zones)

        pf_wire = make_polygon(self.pf_coords.xyz, closed=True)
        incl_wires = boolean_cut(pf_wire, excl_zone)
        incl_loops = [w.discretize(byedges=True, ndiscr=100) for w in incl_wires]

        outer_wire = offset_wire(excl_zone.boundary[0], 100)
        negative = BluemiraFace([outer_wire, excl_zone.boundary[0]])
        excl_wires = boolean_cut(pf_wire, negative)
        excl_loops = [w.discretize(byedges=True, ndiscr=100) for w in excl_wires]
        self.incl_loops = incl_loops
        self.excl_loops = excl_loops

        # Track start and end points
        p0 = self.pf_coords.xz.T[0]
        p1 = self.pf_coords.xz.T[-1]

        # Calculate exclusion sections in parametric space
        exclusions = []
        for _i, excl in enumerate(self.excl_loops):
            # Check if the start point lies in the exclusion
            if np.allclose(p0, excl.xz.T[0]) or np.allclose(p0, excl.xz.T[-1]):
                start = 0
            else:
                start = self.xz_to_L(*excl.xz.T[0])

            # Check if the end point lies in the inclusion
            if np.allclose(p1, excl.xz.T[-1]) or np.allclose(p1, excl.xz.T[0]):
                stop = 1
            else:
                stop = self.xz_to_L(*excl.xz.T[-1])

            exclusions.append(sorted([start, stop]))

        # Sort by order in parametric space
        self.exclusions = sorted(exclusions, key=lambda x: x[0])

    def plot(self, ax=None):
        """
        Plot the XZLMapper.
        """
        return XZLPlotter(self, ax=ax)


class RegionMapper:
    """
    Coil positioning tools for use in optimisation for regions.

    Parameters
    ----------
    pfregions:
        Regions in which each PF coil resides. The Coordinates objects must be 2-D in
        x, z.
    """

    def __init__(self, pfregions: Dict[str, Coordinates]):
        self.pfregions = pfregions

        self.regions = {}
        self.name_str = "R_{}"

        try:
            for pf_name, loop_reg in self.pfregions.items():
                self._region_setup(pf_name, loop_reg)
        except AttributeError:
            raise EquilibriaError("pfregions is not a dictionary")

        self.no_regions = len(self.regions)

        self.l_values = np.zeros((self.no_regions, 2))
        self.l_map = self.l_values.flatten()

        self.max_currents = np.zeros(self.no_regions)

    def _region_setup(self, pf_name, loop_reg):
        if all(loop_reg.y != 0):
            raise EquilibriaError(
                "Coordinates object must be 2D- in x, z for RegionMapper"
            )

        region_name = self._name_converter(pf_name, True)
        self.regions[region_name] = RegionInterpolator(loop_reg)

    def _regionname(self, region):
        if not isinstance(region, str):
            return self.name_str.format(region)
        elif re.match("^R_[0-9]+([.][0-9]+)?$", region):
            return region
        elif re.match("^PF_[0-9]+([.][0-9]+)?$", region):
            return self._name_converter(region, True)
        else:
            raise NameError("RegionName not valid")

    def _name_converter(self, regionname, coil_to_region=False):
        num = int(regionname.split("_")[-1])
        if coil_to_region:
            return self.name_str.format(num)
        else:
            return f"PF_{num}"

    def add_region(self, pfregion: Dict[str, Coordinates]):
        """
        Add an extra region to map.

        Parameters
        ----------
        pfregion:
            A region where a PF coil will reside

        """
        self.pfregions = {**self.pfregions, **pfregion}
        name, region = next(iter(pfregion.items()))
        self.no_regions += 1
        self.l_values = np.zeros((self.no_regions, 2))
        self.max_currents = np.zeros(self.no_regions)
        self._region_setup(name, region)

    def L_to_xz(self, region, l_values):
        """
        Convert L values to x,z values for a given region.
        """
        reg = self.regions[self._regionname(region)]
        # l_values = self.region_coil_overlap(l_values)
        xv, zv = reg.to_xz(l_values)
        return xv, zv

    def xz_to_L(
        self, region, x: Union[float, np.ndarray], z: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert x,z values to L values for a given region.
        """
        reg = self.regions[self._regionname(region)]
        l_0, l_1 = reg.to_L(x, z)
        return l_0, l_1

    def get_Lmap(self, coilset: CoilSet) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get 1D array of mapped position coordinates from coil positions
        in a provided coilset, along with mapped position bounds.

        Parameters
        ----------
        coilset:
            A coilset object to map
        """
        self._coilset = coilset

        for no, region in enumerate(self.regions.keys()):
            try:
                coil = coilset[self._name_converter(region)]
            except KeyError:
                bluemira_warn(f"{self._name_converter(region)} not found in coilset")
                continue

            self.l_values[no] = self.xz_to_L(region, coil.x, coil.z)

        # Force all initial positions to be within region
        self.l_map = tools.clip(self.l_values, 0, 1).flatten()
        return (
            self.l_map,
            np.zeros_like(self.l_map),
            np.ones_like(self.l_map),
        )

    def set_Lmap(self, l_map):
        """
        Sets the mapped positions from a provided 1D array.
        """
        if np.size(l_map) == 2 * self.no_regions:
            self.l_map = l_map
            self.l_values = l_map.reshape(-1, 2)
        else:
            raise EquilibriaError(
                "Provided l_map does not contain exactly one pair of mapped"
                "coordinates for each region in RegionMapper"
            )

    def get_xz_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get arrays containing x and z coordinates for all coils from the position
        map.

        Returns
        -------
        x:
            Array containing radial positions of all coils in mapped regions,
            enumerated by region index in self.regions.
        z:
            Array containing vertical positions of all coils in mapped regions,
            enumerated by region index in self.regions.

        """
        x, z = np.zeros(len(self.l_values)), np.zeros(len(self.l_values))
        for i, region in enumerate(self.regions.keys()):
            x[i], z[i] = self.L_to_xz(region, self.l_values[i])
        return x, z

    def get_size_current_limit(self) -> np.ndarray:
        """
        Get maximum coil current while staying within region boundaries.

        Coils are set up as current per unit area therefore limiting the max current
        limits the area a coil covers.

        Returns
        -------
        Max current for coil location within region
        """
        for no, (name, region) in enumerate(self.regions.items()):
            coil = self._coilset[self._name_converter(name)]
            self.max_currents[no] = get_max_current(
                *inscribed_rect_in_poly(
                    region.loop.x, region.loop.z, coil.x, coil.z, coil.dx / coil.dz
                ),
                coil.j_max,
            )

        return self.max_currents

    def plot(self, ax=None):
        """
        Plot the RegionMapper.
        """
        return RegionPlotter(self, ax=ax)


class RegionInterpolator:
    """
    Sets up a region for a PF coil to move within.

    We are treating the region as a flat surface.

    The normalisation occurs by cutting the shape in two axes and
    normalising over the cut length within the region.

    Currently this is limited to convex polygons (also know as convex hulls).
    Generalisation to all polygons is possible but unimplemented
    and possibly quite slow when converting from normalised to real coordinates.

    When the coil position provided is outside the given region the coil will
    be moved to the closest edge of the region.

    The mapping from outside to the edge of the region is not strictly defined.
    The only certainty is that the coil will be moved into the region.

    Parameters
    ----------
    coords:
        Region to interpolate within
    """

    def __init__(self, coords: Coordinates):
        self.x = coords.x
        self.z = coords.z
        self.coords = coords

        self.check_loop_feasibility(coords)

        self.coords = coords
        self.z_min = min(self.coords.z)
        self.z_max = max(self.coords.z)

    def to_xz(self, l_values: List[float]) -> Tuple[float, float]:
        """
        Convert L values to x,z values for xy_cut.

        Parameters
        ----------
        l_values:
            Coordinates in normalised space

        Returns
        -------
        x:
            Radial coordinates in real space
        z:
            Vertical coordinate in real space

        Raises
        ------
        GeometryError
            When coordinates are not a Convex Hull
        """
        l_0, l_1 = l_values
        z = self.z_min + (self.z_max - self.z_min) * l_1

        plane = BluemiraPlane.from_3_points([0, 0, z], [1, 0, z], [0, 1, z])

        intersect = coords_plane_intersect(self.coords, plane)
        if len(intersect) == 1:
            x = intersect[0][0]
        elif len(intersect) == 2:
            x_min, x_max = sorted([intersect[0][0], intersect[1][0]])
            x = x_min + (x_max - x_min) * l_0
        else:
            raise GeometryError("Region must be a Convex Hull")

        return x, z

    def to_L(self, x: float, z: float) -> Tuple[float, float]:
        """
        Convert x.z values to L values for xy_cut.

        Parameters
        ----------
        x:
            Radial coordinates in real space
        z:
            Vertical coordinate in real space

        Returns
        -------
        l_0:
            Coordinate 1 in normalised space
        l_1:
            Coordinate 2 in normalised space

        Raises
        ------
        GeometryError
            When coordinates are not a Convex Hull
        """
        l_1 = (z - self.z_min) / (self.z_max - self.z_min)
        l_1 = tools.clip(l_1, 0.0, 1.0)

        plane = BluemiraPlane.from_3_points([x, 0, z], [x + 1, 0, z], [x, 1, z])
        intersect = coords_plane_intersect(self.coords, plane)

        return self._intersect_filter(x, l_1, intersect)

    def _intersect_filter(
        self, x: float, l_1: float, intersect: BluemiraPlane
    ) -> Tuple[float, float]:
        """
        Checks where points are based on number of intersections
        with a plane. Should initially be called with a plane involving z.

        No intersection could mean above 1 edge therefore a plane in xy
        is checked before recalling this function.
        If there is one intersection point we are on an edge (either bottom or top),
        if there is two intersection points we are in the region,
        otherwise the region is not a convex hull.

        Parameters
        ----------
        x:
            x coordinate
        l_1:
            Normalised z coordinate
        intersect:
            A plane through xz

        Returns
        -------
        l_0:
            Coordinate 1 in normalised space
        l_1:
            Coordinate 2 in normalised space

        Raises
        ------
        GeometryError
            When coordinates are not a Convex Hull
        """
        if intersect is None:
            plane = BluemiraPlane.from_3_points([x, 0, 0], [x + 1, 0, 0], [x, 1, 0])
            intersect = coords_plane_intersect(self.coords, plane)
            l_0, l_1 = self._intersect_filter(
                x, l_1, [False] if intersect is None else intersect
            )
        elif len(intersect) == 2:
            x_min, x_max = sorted([intersect[0][0], intersect[1][0]])
            l_0 = tools.clip((x - x_min) / (x_max - x_min), 0.0, 1.0)
        elif len(intersect) == 1:
            l_0 = float(l_1 == 1.0)
        else:
            raise GeometryError("Region must be a Convex Hull")
        return l_0, l_1

    @staticmethod
    def check_loop_feasibility(coords: Coordinates):
        """
        Checks the provided region is a ConvexHull.

        This is a current limitation of RegionMapper
        not providing a 'smooth' interpolation surface.

        Parameters
        ----------
        coords:
            Region to check

        Raises
        ------
        GeometryError
            When coordinates are not a Convex Hull

        """
        if not np.allclose(
            ConvexHull(coords.xz.T).volume, get_area_2d(coords.x, coords.z), atol=EPS
        ):
            raise GeometryError("Region must be a Convex Hull")
