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
Flux surface utility classes and calculations
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Iterable, Optional, Tuple, Union

if TYPE_CHECKING:
    from bluemira.equilibria.equilibrium import Equilibrium
    from bluemira.equilibria.find import PsiPoint

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from scipy.integrate import solve_ivp

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.equilibria.constants import PSI_NORM_TOL
from bluemira.equilibria.error import EquilibriaError, FluxSurfaceError
from bluemira.equilibria.find import find_flux_surface_through_point
from bluemira.equilibria.grid import Grid
from bluemira.geometry.coordinates import (
    Coordinates,
    check_linesegment,
    coords_plane_intersect,
    get_angle_between_points,
    get_area_2d,
    get_intersect,
    join_intersect,
)
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import _signed_distance_2D


@nb.jit(nopython=True, cache=True)
def _flux_surface_dl(x, dx, dz, Bp, Bt):
    Bp = 0.5 * (Bp[1:] + Bp[:-1])
    Bt = Bt[:-1] * x[:-1] / (x[:-1] + 0.5 * dx)
    B_ratio = Bt / Bp
    return np.sqrt(1 + B_ratio**2) * np.hypot(dx, dz)


class FluxSurface:
    """
    Flux surface base class.

    Parameters
    ----------
    geometry:
        Flux surface geometry object
    """

    __slots__ = ("coords",)

    def __init__(self, geometry: Coordinates):
        self.coords = geometry

    @property
    def x_start(self) -> float:
        """
        Start radial coordinate of the FluxSurface.
        """
        return self.coords.x[0]

    @property
    def z_start(self) -> float:
        """
        Start vertical coordinate of the FluxSurface.
        """
        return self.coords.z[0]

    @property
    def x_end(self) -> float:
        """
        End radial coordinate of the FluxSurface.
        """
        return self.coords.x[-1]

    @property
    def z_end(self) -> float:
        """
        End vertical coordinate of the FluxSurface.
        """
        return self.coords.z[-1]

    def _dl(self, eq):
        x, z = self.coords.x, self.coords.z
        Bp = eq.Bp(x, z)
        Bt = eq.Bt(x)
        return _flux_surface_dl(x, np.diff(x), np.diff(z), Bp, Bt)

    def connection_length(self, eq: Equilibrium) -> float:
        """
        Calculate the parallel connection length along a field line (i.e. flux surface).

        Parameters
        ----------
        eq:
            Equilibrium from which the FluxSurface was extracted

        Returns
        -------
        Connection length from the start of the flux surface to the end of the flux
        surface
        """
        return np.sum(self._dl(eq))

    def plot(self, ax: Optional[plt.Axes] = None, **kwargs):
        """
        Plot the FluxSurface.
        """
        if ax is None:
            ax = plt.gca()

        kwargs["linewidth"] = kwargs.get("linewidth", 0.05)
        kwargs["color"] = kwargs.get("color", "r")

        self.coords.plot(ax, **kwargs)

    def copy(self):
        """
        Make a deep copy of the FluxSurface.
        """
        return deepcopy(self)


class ClosedFluxSurface(FluxSurface):
    """
    Utility class for closed flux surfaces.
    """

    __slots__ = ("_p1", "_p2", "_p3", "_p4", "_z_centre")

    def __init__(self, geometry: Coordinates):
        if not geometry.closed:
            raise FluxSurfaceError(
                "Cannot make a ClosedFluxSurface from an open geometry."
            )
        super().__init__(geometry)
        i_p1 = np.argmax(self.coords.x)
        i_p2 = np.argmax(self.coords.z)
        i_p3 = np.argmin(self.coords.x)
        i_p4 = np.argmin(self.coords.z)
        self._p1 = (self.coords.x[i_p1], self.coords.z[i_p1])
        self._p2 = (self.coords.x[i_p2], self.coords.z[i_p2])
        self._p3 = (self.coords.x[i_p3], self.coords.z[i_p3])
        self._p4 = (self.coords.x[i_p4], self.coords.z[i_p4])

        # Still debatable what convention to follow...
        self._z_centre = 0.5 * (self.coords.z[i_p1] + self.coords.z[i_p3])

    @property
    @lru_cache(1)
    def major_radius(self) -> float:
        """
        Major radius of the ClosedFluxSurface.
        """
        return np.min(self.coords.x) + self.minor_radius

    @property
    @lru_cache(1)
    def minor_radius(self) -> float:
        """
        Minor radius of the ClosedFluxSurface.
        """
        return 0.5 * (np.max(self.coords.x) - np.min(self.coords.x))

    @property
    @lru_cache(1)
    def aspect_ratio(self) -> float:
        """
        Aspect ratio of the ClosedFluxSurface.
        """
        return self.major_radius / self.minor_radius

    @property
    @lru_cache(1)
    def kappa(self) -> float:
        """
        Average elongation of the ClosedFluxSurface.
        """
        return 0.5 * (np.max(self.coords.z) - np.min(self.coords.z)) / self.minor_radius

    @property
    @lru_cache(1)
    def kappa_upper(self) -> float:
        """
        Upper elongation of the ClosedFluxSurface.
        """
        return (np.max(self.coords.z) - self._z_centre) / self.minor_radius

    @property
    @lru_cache(1)
    def kappa_lower(self) -> float:
        """
        Lower elongation of the ClosedFluxSurface.
        """
        return abs(np.min(self.coords.z) - self._z_centre) / self.minor_radius

    @property
    @lru_cache(1)
    def delta(self) -> float:
        """
        Average triangularity of the ClosedFluxSurface.
        """
        return 0.5 * (self.delta_upper + self.delta_lower)

    @property
    @lru_cache(1)
    def delta_upper(self) -> float:
        """
        Upper triangularity of the ClosedFluxSurface.
        """
        return (self.major_radius - self._p2[0]) / self.minor_radius

    @property
    @lru_cache(1)
    def delta_lower(self) -> float:
        """
        Lower triangularity of the ClosedFluxSurface.
        """
        return (self.major_radius - self._p4[0]) / self.minor_radius

    @property
    @lru_cache(1)
    def zeta(self) -> float:
        """
        Average squareness of the ClosedFluxSurface.
        """
        return 0.5 * (self.zeta_upper + self.zeta_lower)

    @property
    @lru_cache(1)
    def zeta_upper(self) -> float:
        """
        Outer upper squareness of the ClosedFluxSurface.
        """
        z_max = np.max(self.coords.z)
        arg_z_max = np.argmax(self.coords.z)
        x_z_max = self.coords.x[arg_z_max]
        x_max = np.max(self.coords.x)
        arg_x_max = np.argmax(self.coords.x)
        z_x_max = self.coords.z[arg_x_max]

        a = z_max - z_x_max
        b = x_max - x_z_max
        return self._zeta_calc(a, b, x_z_max, z_x_max, x_max, z_max)

    @property
    @lru_cache(1)
    def zeta_lower(self) -> float:
        """
        Outer lower squareness of the ClosedFluxSurface.
        """
        z_min = np.min(self.coords.z)
        arg_z_min = np.argmin(self.coords.z)
        x_z_min = self.coords.x[arg_z_min]
        x_max = np.max(self.coords.x)
        arg_x_max = np.argmax(self.coords.x)
        z_x_max = self.coords.z[arg_x_max]

        a = z_min - z_x_max
        b = x_max - x_z_min

        return self._zeta_calc(a, b, x_z_min, z_x_max, x_max, z_min)

    def _zeta_calc(self, a, b, xa, za, xd, zd):
        """
        Actual squareness calculation

        Notes
        -----
        Squareness defined here w.r.t an ellipse intersection along a projected line
        """
        xc = xa + b * np.sqrt(0.5)
        zc = za + a * np.sqrt(0.5)

        line = Coordinates({"x": [xa, xd], "z": [za, zd]})
        xb, zb = get_intersect(self.coords.xz, line.xz)
        d_ab = np.hypot(xb - xa, zb - za)
        d_ac = np.hypot(xc - xa, zc - za)
        d_cd = np.hypot(xd - xc, zd - zc)
        return float((d_ab - d_ac) / d_cd)

    @property
    @lru_cache(1)
    def area(self) -> float:
        """
        Enclosed area of the ClosedFluxSurface.
        """
        return get_area_2d(*self.coords.xz)

    @property
    @lru_cache(1)
    def volume(self) -> float:
        """
        Volume of the ClosedFluxSurface.
        """
        return 2 * np.pi * self.area * self.coords.center_of_mass[0]

    def shafranov_shift(self, eq: Equilibrium) -> Tuple[float, float]:
        """
        Calculate the Shafranov shift of the ClosedFluxSurface.

        Parameters
        ----------
        eq:
            Equilibrium with which to calculate the safety factor

        Returns
        -------
        dx_shaf:
            Radial Shafranov shift
        dz_shaf:
            Vertical Shafranov shift
        """
        o_point = eq.get_OX_points()[0][0]  # magnetic axis
        return o_point.x - self.major_radius, o_point.z - self._z_centre

    def safety_factor(self, eq: Equilibrium) -> float:
        """
        Calculate the cylindrical safety factor of the ClosedFluxSurface. The ratio of
        toroidal turns to a single full poloidal turn.

        Parameters
        ----------
        eq:
            Equilibrium with which to calculate the safety factor

        Returns
        -------
        Cylindrical safety factor of the closed flux surface
        """
        x, z = self.coords.x, self.coords.z
        dx, dz = np.diff(x), np.diff(z)
        x = x[:-1] + 0.5 * dx  # Segment centre-points
        z = z[:-1] + 0.5 * dz
        dl = np.hypot(dx, dz)  # Poloidal plane dl
        Bp = eq.Bp(x, z)
        Bt = eq.Bt(x)
        return np.sum(dl * Bt / (Bp * x)) / (2 * np.pi)


class OpenFluxSurface(FluxSurface):
    """
    Utility class for handling open flux surface geometries.
    """

    __slots__ = ()

    def __init__(self, coords: Coordinates):
        if coords.closed:
            raise FluxSurfaceError(
                "OpenFluxSurface cannot be made from a closed geometry."
            )
        super().__init__(coords)

    def split(
        self, o_point: PsiPoint, plane: Optional[BluemiraPlane] = None
    ) -> Tuple[PartialOpenFluxSurface, PartialOpenFluxSurface]:
        """
        Split an OpenFluxSurface into two separate PartialOpenFluxSurfaces about a
        horizontal plane.

        Parameters
        ----------
        o_point:
            The magnetic centre of the plasma
        plane:
            The x-y cutting plane. Will default to the O-point x-y plane

        Returns
        -------
        down:
            The downwards open flux surfaces from the splitting point
        up:
            The upwards open flux surfaces from the splitting point
        """

        def reset_direction(coords):
            if coords.argmin([x_mp, 0, z_mp]) != 0:
                coords.reverse()
            return coords

        if plane is None:
            plane = BluemiraPlane.from_3_points(
                [o_point.x, 0, o_point.z],
                [o_point.x + 1, 0, o_point.z],
                [o_point.x, 1, o_point.z],
            )

        ref_coords = deepcopy(self.coords)
        intersections = coords_plane_intersect(ref_coords, plane)
        x_inter = intersections.T[0]

        # Pick the first intersection, travelling from the o_point outwards
        deltas = x_inter - o_point.x
        arg_inter = np.argmax(deltas > 0)
        x_mp = x_inter[arg_inter]
        z_mp = o_point.z

        # Split the flux surface geometry into LFS and HFS geometries

        delta = 1e-1 if o_point.x < x_mp else -1e-1
        radial_line = Coordinates({"x": [o_point.x, x_mp + delta], "z": [z_mp, z_mp]})
        # Add the intersection point to the Coordinates
        arg_inter = join_intersect(ref_coords, radial_line, get_arg=True)[0]

        # Split the flux surface geometry
        coords1 = Coordinates(ref_coords[:, : arg_inter + 1])
        coords2 = Coordinates(ref_coords[:, arg_inter:])

        coords1 = reset_direction(coords1)
        coords2 = reset_direction(coords2)

        # Sort the segments into down / outboard and up / inboard geometries
        if coords1.z[1] > z_mp:
            lfs_coords = coords2
            hfs_coords = coords1
        else:
            lfs_coords = coords1
            hfs_coords = coords2
        return PartialOpenFluxSurface(lfs_coords), PartialOpenFluxSurface(hfs_coords)


class PartialOpenFluxSurface(OpenFluxSurface):
    """
    Utility class for a partial open flux surface, i.e. an open flux surface that has
    been split at the midplane and only has one intersection with the wall.
    """

    __slots__ = ["alpha"]

    def __init__(self, coords: Coordinates):
        super().__init__(coords)

        self.alpha = None

    def clip(self, first_wall: Coordinates):
        """
        Clip the PartialOpenFluxSurface to a first wall.

        Parameters
        ----------
        first_wall:
            The geometry of the first wall to clip the OpenFluxSurface to
        """
        first_wall = deepcopy(first_wall)

        args = join_intersect(self.coords, first_wall, get_arg=True)

        if not args:
            bluemira_warn(
                "No intersection detected between flux surface and first_wall."
            )
            self.alpha = None
            return

        # Because we oriented the Coordinates the "right" way, the first intersection
        # is at the smallest argument
        self.coords = Coordinates(self.coords[:, : min(args) + 1])

        fw_arg = int(first_wall.argmin([self.x_end, 0, self.z_end]))

        if fw_arg + 1 == len(first_wall):
            pass
        elif check_linesegment(
            first_wall.xz.T[fw_arg],
            first_wall.xz.T[fw_arg + 1],
            np.array([self.x_end, self.z_end]),
        ):
            fw_arg = fw_arg + 1

        # Relying on the fact that first wall is ccw, get the intersection angle
        self.alpha = get_angle_between_points(
            self.coords.points[-2], self.coords.points[-1], first_wall.points[fw_arg]
        )

    def flux_expansion(self, eq: Equilibrium) -> float:
        """
        Flux expansion of the PartialOpenFluxSurface.

        Parameters
        ----------
        eq:
            Equilibrium with which to calculate the flux expansion

        Returns
        -------
        Target flux expansion
        """
        return (
            self.x_start
            * eq.Bp(self.x_start, self.z_start)
            / (self.x_end * eq.Bp(self.x_end, self.z_end))
        )


@dataclass
class CoreResults:
    """
    Dataclass for core results.
    """

    psi_n: Iterable
    R_0: Iterable
    a: Iterable
    A: Iterable
    area: Iterable
    V: Iterable
    kappa: Iterable
    delta: Iterable
    zeta: Iterable
    kappa_upper: Iterable
    delta_upper: Iterable
    zeta_upper: Iterable
    kappa_lower: Iterable
    delta_lower: Iterable
    zeta_lower: Iterable
    q: Iterable
    Delta_shaf: Iterable


def analyse_plasma_core(eq: Equilibrium, n_points: int = 50) -> CoreResults:
    """
    Analyse plasma core parameters across the normalised 1-D flux coordinate.

    Returns
    -------
    Results dataclass
    """
    psi_n = np.linspace(PSI_NORM_TOL, 1 - PSI_NORM_TOL, n_points, endpoint=False)
    coords = [eq.get_flux_surface(pn) for pn in psi_n]
    coords.append(eq.get_LCFS())
    psi_n = np.append(psi_n, 1.0)
    flux_surfaces = [ClosedFluxSurface(coord) for coord in coords]
    _vars = ["major_radius", "minor_radius", "aspect_ratio", "area", "volume"]
    _vars += [
        f"{v}{end}"
        for end in ["", "_upper", "_lower"]
        for v in ["kappa", "delta", "zeta"]
    ]
    return CoreResults(
        psi_n,
        *[[getattr(fs, var) for fs in flux_surfaces] for var in _vars],
        [fs.safety_factor(eq) for fs in flux_surfaces],
        [fs.shafranov_shift(eq)[0] for fs in flux_surfaces],
    )


class FieldLine:
    """
    Field line object.

    Parameters
    ----------
    coords:
        Geometry of the FieldLine
    connection_length:
        Connection length of the FieldLine
    """

    def __init__(self, coords: Coordinates, connection_length: float):
        self.coords = coords
        self.connection_length = connection_length

    def plot(self, ax: Optional[plt.Axes] = None, **kwargs):
        """
        Plot the FieldLine.

        Parameters
        ----------
        ax:
            Matplotlib axes onto which to plot
        """
        self.coords.plot(ax=ax, **kwargs)

    def pointcare_plot(self, ax: Optional[plt.Axes] = None):
        """
        Pointcaré plot of the field line intersections with the half-xz-plane.

        Parameters
        ----------
        ax:
            Matplotlib axes onto which to plot
        """
        if ax is None:
            ax = plt.gca()

        xz_plane = BluemiraPlane.from_3_points([0, 0, 0], [1, 0, 0], [0, 0, 1])
        xi, _, zi = coords_plane_intersect(self.coords, xz_plane).T
        idx = np.nonzero(xi >= 0)
        xi = xi[idx]
        zi = zi[idx]
        ax.plot(xi, zi, linestyle="", marker="o", color="r", ms=5)
        ax.set_aspect("equal")


class FieldLineTracer:
    """
    Field line tracing tool.

    Parameters
    ----------
    eq:
        Equilibrium in which to trace a field line
    first_wall:
        Boundary at which to stop tracing the field line

    Notes
    -----
    Totally pinched some maths from Ben Dudson's FreeGS here... Perhaps one day I can
    return the favours.

    I needed it to compare the analytical connection length calculation with something,
    so I nicked this but changed the way the equation is solved.

    Note that this will properly trace field lines through Coils as it doesn't rely on
    the psi map (which is inaccurate inside Coils).
    """

    class CollisionTerminator:
        """
        Private Event handling object for solve_ivp

        Parameters
        ----------
        boundary: Union[Grid, Coordinates]
            Boundary at which to stop tracing the field line.
        """

        def __init__(self, boundary: Union[Grid, Coordinates]):
            self.boundary = boundary
            self.terminal = True

        def __call__(self, _phi, xz, *_args):
            """
            Function handle for the CollisionTerminator
            """
            if isinstance(self.boundary, Grid):
                return self._call_grid(xz)
            return self._call_coordinates(xz)

        def _call_grid(self, xz):
            """
            Function handle for the CollisionTerminator in the case of a Grid.
            (slight speed improvement)
            """
            if self.boundary.point_inside(xz[:2]):
                return np.min(self.boundary.distance_to(xz[:2]))
            return -np.min(self.boundary.distance_to(xz[:2]))

        def _call_coordinates(self, xz):
            """
            Function handle for the CollisionTerminator in the case of Coordinates.
            """
            return _signed_distance_2D(xz[:2], self.boundary.xz.T)

    def __init__(
        self, eq: Equilibrium, first_wall: Optional[Union[Grid, Coordinates]] = None
    ):
        self.eq = eq
        if first_wall is None:
            first_wall = self.eq.grid
        elif isinstance(first_wall, Coordinates) and not first_wall.is_planar:
            raise EquilibriaError(
                "When tracing a field line, the coordinates object of the boundary must"
                " be planar."
            )
        self.first_wall = first_wall

    def trace_field_line(
        self,
        x: float,
        z: float,
        n_points: int = 200,
        forward: bool = True,
        n_turns_max: int = 20,
    ) -> FieldLine:
        """
        Trace a single field line starting at a point.

        Parameters
        ----------
        x:
            Radial coordinate of the starting point
        z:
            Vertical coordinate of the starting point
        n_points:
            Number of points along the field line
        forward:
            Whether or not to step forward or backward (+B or -B)
        n_turns_max: Union[int, float]
            Maximum number of toroidal turns to trace the field line

        Returns
        -------
        Resulting field line
        """
        phi = np.linspace(0, 2 * np.pi * n_turns_max, n_points)

        result = solve_ivp(
            self._dxzl_dphi,
            y0=np.array([x, z, 0]),
            t_span=(0, 2 * np.pi * n_turns_max),
            t_eval=phi,
            events=self.CollisionTerminator(self.first_wall),
            method="LSODA",
            args=(forward,),
        )
        r, z, phi, connection_length = self._process_result(result)

        x = r * np.cos(phi)
        y = r * np.sin(phi)
        coords = Coordinates({"x": x, "y": y, "z": z})
        return FieldLine(coords, connection_length)

    def _dxzl_dphi(self, _phi, xz, forward):
        """
        Credit: Dr. B. Dudson, FreeGS.
        """
        f = 1.0 if forward is True else -1.0
        Bx = self.eq.Bx(*xz[:2])
        Bz = self.eq.Bz(*xz[:2])
        Bt = self.eq.Bt(xz[0])
        B = np.sqrt(Bx**2 + Bz**2 + Bt**2)
        dx, dz, dl = xz[0] / Bt * np.array([f * Bx, f * Bz, B])
        return np.array([dx, dz, dl])

    @staticmethod
    def _process_result(result):
        if len(result["y_events"][0]) != 0:
            # Field line tracing was terminated by a collision
            end = len(result["y"][0])
            r, z = result["y"][0][:end], result["y"][1][:end]
            phi = result["t"][:end]
            termination = result["y_events"][0].flatten()
            r = np.append(r, termination[0])
            z = np.append(z, termination[1])
            connection_length = termination[2]
            phi = np.append(phi, result["t_events"][0][0])

        else:
            # Field line tracing was not terminated by a collision
            r, z, length = result["y"][0], result["y"][1], result["y"][2]
            phi = result["t"]
            connection_length = length[-1]
        return r, z, phi, connection_length


def calculate_connection_length_flt(
    eq: Equilibrium,
    x: float,
    z: float,
    forward: bool = True,
    first_wall=Optional[Union[Coordinates, Grid]],
    n_turns_max: int = 50,
) -> float:
    """
    Calculate the parallel connection length from a starting point to a flux-intercepting
    surface using a field line tracer.

    Parameters
    ----------
    eq:
        Equilibrium in which to calculate the connection length
    x:
        Radial coordinate of the starting point
    z:
        Vertical coordinate of the starting point
    forward:
        Whether or not to follow the field line forwards or backwards (+B or -B)
    first_wall:
        Flux-intercepting surface. Defaults to the grid of the equilibrium
    n_turns_max:
        Maximum number of toroidal turns to trace the field line

    Returns
    -------
    Parallel connection length along the field line from the starting point to the
    intersection point [m]

    Notes
    -----
    More mathematically accurate, but needs additional configuration. Will not likely
    return a very accurate flux interception point. Also works for closed flux surfaces,
    but can't tell the difference. Not sensitive to equilibrium grid discretisation.
    Will work correctly for flux surfaces passing through Coils, but really they should
    be intercepted beforehand!
    """
    flt = FieldLineTracer(eq, first_wall)
    field_line = flt.trace_field_line(
        x, z, forward=forward, n_points=2, n_turns_max=n_turns_max
    )
    return field_line.connection_length


def calculate_connection_length_fs(
    eq: Equilibrium,
    x: float,
    z: float,
    forward: bool = True,
    first_wall=Optional[Union[Coordinates, Grid]],
) -> float:
    """
    Calculate the parallel connection length from a starting point to a flux-intercepting
    surface using flux surface geometry.

    Parameters
    ----------
    eq:
        Equilibrium in which to calculate the connection length
    x:
        Radial coordinate of the starting point
    z:
        Vertical coordinate of the starting point
    forward:
        Whether or not to follow the field line forwards or backwards
    first_wall:
        Flux-intercepting surface. Defaults to the grid of the equilibrium

    Returns
    -------
    Parallel connection length along the field line from the starting point to the
    intersection point [m]

    Raises
    ------
    FluxSurfaceError
        If the flux surface at the point in the equilibrium is not an open flux surface

    Notes
    -----
    Less mathematically accurate. Will return exact intersection point. Sensitive to
    equilibrium grid discretisation. Presently does not correctly work for flux surfaces
    passing through Coils, but really they should be intercepted beforehand!
    """
    if first_wall is None:
        x1, x2 = eq.grid.x_min, eq.grid.x_max
        z1, z2 = eq.grid.z_min, eq.grid.z_max
        first_wall = Coordinates({"x": [x1, x2, x2, x1, x1], "z": [z1, z1, z2, z2, z1]})

    xfs, zfs = find_flux_surface_through_point(eq.x, eq.z, eq.psi(), x, z, eq.psi(x, z))
    f_s = OpenFluxSurface(Coordinates({"x": xfs, "z": zfs}))

    class Point:
        def __init__(self, x, z):
            self.x = x
            self.z = z

    lfs, hfs = f_s.split(Point(x=x, z=z))
    fs = lfs if forward else hfs

    fs.clip(first_wall)
    return fs.connection_length(eq)


def poloidal_angle(Bp_strike: float, Bt_strike: float, gamma: float) -> float:
    """
    From glancing angle (gamma) to poloidal angle.

    Parameters
    ----------
    Bp_strike:
        Poloidal magnetic field value at the desired point
    Bt_strike:
        Toroidal magnetic field value at the desired point
    gamma:
        Glancing angle at the strike point [deg]

    Returns
    -------
    Poloidal angle at the strike point [deg]
    """
    # From deg to rad
    gamma_rad = np.radians(gamma)

    # Total magnetic field
    Btot = np.sqrt(Bp_strike**2 + Bt_strike**2)

    # Numerator of next operation
    num = Btot * np.sin(gamma_rad)
    # Poloidal projection of the glancing angle
    sin_theta = num / Bp_strike

    return np.rad2deg(np.arcsin(sin_theta))
