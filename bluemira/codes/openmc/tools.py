# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
The OpenMC Environment, and tools to interact with it by creating surfaces and defining
regions inside of this environment.
"""


from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import openmc
import openmc.region

from bluemira.geometry.error import GeometryError
from bluemira.geometry.tools import (
    make_circle_arc_3P,
)
from bluemira.radiation_transport.neutronics.constants import (
    DTOL_CM,
    to_cm,
    to_m,
)
from bluemira.radiation_transport.neutronics.wires import CircleInfo

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy.typing as npt

    from bluemira.radiation_transport.neutronics.wires import (
        StraightLineInfo,
        WireInfoList,
    )

SHRINK_DISTANCE = 0.0005  # [m] = 0.05cm = 0.5 mm # Found to work by trial and error.


def torus_from_3points(
    point1: npt.NDArray[np.float64],
    point2: npt.NDArray[np.float64],
    point3: npt.NDArray[np.float64],
    surface_id: int | None = None,
    name: str = "",
) -> openmc.ZTorus:
    """
    Make a circular torus centered on the z-axis using 3 points.
    All 3 points should lie on the RZ plane AND the surface of the torus simultaneously.

    Parameters
    ----------
    point1, point2, point3:
        RZ coordinates of the 3 points on the surface of the torus.
    surface_id, name:
        See openmc.Surface

    Returns
    -------
    :
        An axisymmetric torus whose surface runs through those three points.
    """
    point1 = point1[0], 0, point1[-1]
    point2 = point2[0], 0, point2[-1]
    point3 = point3[0], 0, point3[-1]
    circle = make_circle_arc_3P(point1, point2, point3)
    cad_circle = circle.shape.OrderedEdges[0].Curve
    center, radius = cad_circle.Center, cad_circle.Radius
    return torus_from_circle(
        [center[0], center[-1]], radius, surface_id=surface_id, name=name
    )


def torus_from_circle(
    center: Sequence[float],
    minor_radius: float,
    surface_id: int | None = None,
    name: str = "",
) -> openmc.ZTorus:
    """
    Make a circular torus centered on the z-axis.
    The circle would lie on the RZ plane AND the surface of the torus simultaneously.

    Parameters
    ----------
    minor_radius:
        Radius of the cross-section circle, which forms the minor radius of the torus.
    center:
        Center of the cross-section circle, which forms the center of the torus.
    surface_id, name:
        See openmc.Surface

    Returns
    -------
    :
        A torus whose surface intersects with this circle at the RZ plane.
    """
    return z_torus(
        [center[0], center[-1]], minor_radius, surface_id=surface_id, name=name
    )


def z_torus(
    center: npt.ArrayLike,
    minor_radius: float,
    surface_id: int | None = None,
    name: str = "",
) -> openmc.ZTorus:
    """
    A circular torus centered on the z-axis.
    The center refers to the major radius and it's z coordinate.

    Parameters
    ----------
    center:
        The center of the torus' RZ plane cross-section
    minor_radius:
        minor radius of the torus

    Returns
    -------
    :
        A torus of the given parameters.
    """
    major_radius, height, minor_radius = to_cm([center[0], center[-1], minor_radius])
    return openmc.ZTorus(
        z0=height,
        a=major_radius,
        b=minor_radius,
        c=minor_radius,
        surface_id=surface_id,
        name=name,
    )


def choose_halfspace(
    surface: openmc.Surface, choice_points: npt.NDArray
) -> openmc.Halfspace:
    """
    Simply take the centroid point of all of the choice_points, and choose the
    corresponding half space

    Parameters
    ----------
    surface:
        an openmc surface

    Returns
    -------
    :
        The half space containing the choice_points.

    Raises
    ------
    GeometryError
        Point is directly on surface
    """
    pt = np.mean(to_cm(choice_points), axis=0)
    value = surface.evaluate(pt)
    if value > 0:
        return +surface
    if value < 0:
        return -surface
    raise GeometryError("Centroid point is directly on the surface")


def choose_plane_cylinders(
    surface: openmc.ZPlane | openmc.ZCylinder, choice_points: npt.NDArray
) -> openmc.Halfspace:
    """
    Choose a side of the Halfspace in the region of ZPlane and ZCylinder.

    Parameters
    ----------
    surface
        :class:`openmc.surface.Surface` of a openmc.ZPlane or openmc.ZCylinder
    choice_points: np.ndarray of shape (N, 3)
        a list of points representing the vertices of a convex polygon in RZ plane

    Returns
    -------
    :
        The half space containing the choice_points.

    Raises
    ------
    GeometryError
        Points on both sides of surface
    """
    x, y, z = np.array(to_cm(choice_points)).T
    values = surface.evaluate([x, y, z])
    threshold = DTOL_CM
    if isinstance(surface, openmc.ZCylinder):
        threshold = 2 * DTOL_CM * surface.r + DTOL_CM**2

    if all(values >= -threshold):
        return +surface
    if all(values <= threshold):
        return -surface

    raise GeometryError(f"There are points on both sides of this {type(surface)}!")


# simplfying openmc.Intersection by associativity
def flat_intersection(region_list: Iterable[openmc.Region]) -> openmc.Intersection:
    """
    Get the flat intersection of an entire list of regions.
    e.g. (a (b c)) becomes (a b c)

    Parameters
    ----------
    region_list:
        The list of regions (or intersection of regions) that we want to find the
        intersection of.

    Returns
    -------
    :
        A flattened representation of the intersection of regions.
    """
    return openmc.Intersection(
        intersection_dictionary(openmc.Intersection(region_list)).values()
    )


def intersection_dictionary(region: openmc.Region) -> dict[str, openmc.Region]:
    """Get a dictionary of all of the elements that shall be intersected together,
    applying the rule of associativity

    Parameters
    ----------
    region:
        Region

    Returns
    -------
    :
        A dictionary of the regions that needs to be intersectioned together; each key
        is the str representation of that region (e.g. '-3', or '9|-11').
    """
    if isinstance(region, openmc.Halfspace):  # termination condition
        return {region.side + str(region.surface.id): region}
    if isinstance(region, openmc.Intersection):
        final_intersection = {}
        for _r in region:
            final_intersection.update(intersection_dictionary(_r))
        return final_intersection
    return {str(region): region}


# simplfying openmc.Union by associativity
def flat_union(region_list: Iterable[openmc.Region]) -> openmc.Union:
    """
    Get the flat union of an entire list of regions.
    e.g. (a | (b | c)) becomes (a | b | c)

    Parameters
    ----------
    region_list:
        A list of regions to be unioned together.

    Returns
    -------
    :
        A union of all of the regions listed.
    """
    return openmc.Union(union_dictionary(openmc.Union(region_list)).values())


# TODO @OceanNuclear: issue 3530: check if simplifying the boolean expressions can yield
# speedup or not, and if so, we should attempt to implement a simplification algorithm.
# https://github.com/Fusion-Power-Plant-Framework/bluemira/issues/3530
# E.g. the expression (-1 ((-1107 -1) | -1108)) can be simplified to (-1107 | -1108) -1;
# which is (hopefully) faster to evaluate.


def union_dictionary(region: openmc.Region) -> dict[str, openmc.Region]:
    """Get a dictionary of all of the elements that shall be unioned together,
    applying the rule of associativity

    Parameters
    ----------
    region:
        Multiple openmc.Halfspaces unioned together.

    Returns
    -------
    :
        A dictionary representation of all of the regions that needs to be unioned
        together.
    """
    if isinstance(region, openmc.Halfspace):  # termination condition
        return {region.side + str(region.surface.id): region}
    if isinstance(region, openmc.Union):
        final_intersection = {}
        for _r in region:
            final_intersection.update(union_dictionary(_r))
        return final_intersection
    return {str(region): region}


def round_up_next_openmc_ids(surface_step_size: int = 1000, cell_step_size: int = 100):
    """
    Make openmc's surfaces' and cells' next IDs to be incremented to the next
    pre-determined interval.

    Parameters
    ----------
    surface_step_size:
        How much should the next surface id be. e.g. if =1000, and current ID = 2200,
        then it will round up to 3000 (next nearest 1000).
    cell_step_size:
        How much should the next cell id be. e.g. if =100, and current ID = 150, then
        it will round up to 200 (next nearest 100).
    """
    openmc.Surface.next_id = (
        int(max(openmc.Surface.used_ids) / surface_step_size + 1) * surface_step_size + 1
    )
    openmc.Cell.next_id = (
        int(max(openmc.Cell.used_ids) / cell_step_size + 1) * cell_step_size + 1
    )


class OpenMCEnvironment:
    """Container for CSG planes to enable reuse of planes, very eco friendly"""

    def __init__(self):
        # it's called a hangar because it's where the planes are parked ;)
        self.hangar = {}

    def surface_from_2points(
        self,
        point1: npt.NDArray[np.float64],
        point2: npt.NDArray[np.float64],
        surface_id: int | None = None,
        name: str = "",
    ) -> openmc.Surface | None:
        """
        Create either a cylinder, a cone, or a surface from 2 points using only the
        rz coordinates of any two points on it.

        Parameters
        ----------
        point1, point2:
            any two non-trivial (i.e. cannot be the same) points on the rz cross-section
            of the surface, each containing the r and z coordinates
            Units: [m]
        surface_id, name:
            see openmc.Surface

        Returns
        -------
        surface:
            if the two points provided are redundant: don't return anything, as this is a
            single point pretending to be a surface. This will come in handy for handling
            the creation of BlanketCells made with 3 surfaces rather than 4.
        """
        point1, point2 = to_cm((point1, point2))
        dr, dz = point2 - point1
        if np.isclose(dr, 0, rtol=0, atol=DTOL_CM):
            if np.isclose(dz, 0, rtol=0, atol=DTOL_CM):
                return None
            return openmc.ZCylinder(r=point1[0], surface_id=surface_id, name=name)
        if np.isclose(dz, 0, rtol=0, atol=DTOL_CM):
            z = point1[-1]
            self.hangar[z] = openmc.ZPlane(z0=z, surface_id=surface_id, name=name)
            return self.hangar[z]
        slope = dz / dr
        z_intercept = -slope * point1[0] + point1[-1]
        return openmc.ZCone(
            z0=z_intercept, r2=slope**-2, surface_id=surface_id, name=name
        )

    def surface_from_straight_line(
        self,
        straight_line_info: StraightLineInfo,
        surface_id: int | None = None,
        name: str = "",
    ) -> openmc.Surface:
        """Create a surface to match the straight line info provided.

        Parameters
        ----------
        straight_line_info:
            The info of the straight line that we're supposed to replicate using a
            surface.
        surface_id, name:
            see openmc.Surface

        Returns
        -------
        :
            the straight line re-produced as a openmc.Surface.
        """
        start_end = np.array(straight_line_info[:2])[:, ::2]
        return self.surface_from_2points(*start_end, surface_id=surface_id, name=name)

    def surfaces_from_info_list(
        self, wire_info_list: WireInfoList, name: str = ""
    ) -> tuple[tuple[openmc.Surface]]:
        """
        Create a list of surfaces using a list of wire infos.

        Parameters
        ----------
        wire_info_list
            List of wires
        name
            Name of the surface that is created from this series of WireInfo.
            This name will be *reused* across all of the surfaces created in this list.

        Returns
        -------
        :
            A collection of 1-tuple/2-tuple of surfaces, each corresponding to a
            WireInfo.

            For every circular arc used, we'll return a pair of shapes (a plane/cone/
            cylinder paired with a torus), otherwise for all straight-lines, we'll just
            return a plane/cone/cylinder, wrapped in a tuple.
        """
        surface_list = []
        for wire in wire_info_list:
            info = wire.key_points
            plane_cone_cylinder = self.surface_from_straight_line(info, name=name)
            if isinstance(info, CircleInfo):
                torus = torus_from_circle(info.center, info.radius, name=name)
                # will need the openmc.Union of these two objects later.
                surface_list.append((plane_cone_cylinder, torus))
            else:
                surface_list.append((plane_cone_cylinder,))
        return tuple(surface_list)

    def find_suitable_z_plane(
        self,
        z0: float,
        z_range: Iterable[float] | None = None,
        surface_id: int | None = None,
        name: str = "",
        **kwargs,
    ):
        """
        Find a suitable z from the hangar, or create a new one if no matches are found.

        Parameters
        ----------
        z0:
            The height of the plane, if we need to create it. Unit: [m]
        z_range:
            If we a suitable z-plane already exists, then we only accept it if it lies
            within this range of z. Unit: [m]
        surface_id, name:
            See openmc.Surface

        Returns
        -------
        :
            A z-plane, either retrieved from the self.hangar or created from scratch and
            saved into self.hangar.
        """
        if z_range is not None:
            z_min, z_max = min(z_range), max(z_range)
            for key in self.hangar:
                if z_min <= key <= z_max:
                    return self.hangar[key]  # return the first match
        self.hangar[z0] = openmc.ZPlane(
            z0=to_cm(z0), surface_id=surface_id, name=name, **kwargs
        )
        return self.hangar[z0]

    def choose_region_cone(
        self,
        surface: openmc.ZCone,
        choice_points: npt.NDArray,
        *,
        control_id: bool = False,
    ) -> openmc.Region:
        """
        Choose the region for a ZCone.
        When reading this function's code, bear in mind that a Z cone can be separated
        into 3 parts:

            A. the upper cone (evaluates to negative),
            B. outside of the cone (evaluates to positive),
            C. the lower cone (evaluates to negative).

        We have to account for the following cases:

        +------------+---------+------------+
        | upper cone | outside | lower cone |
        +============+=========+============+
        |      Y     |    N    |      N     |
        +------------+---------+------------+
        |      Y     |    Y    |      N     |
        +------------+---------+------------+
        |      N     |    Y    |      N     |
        +------------+---------+------------+
        |      N     |    Y    |      Y     |
        +------------+---------+------------+
        |      N     |    N    |      Y     |
        +------------+---------+------------+

        All other cases should raise an error.
        The tricky part to handle is the floating point precision problem.
        It's possible that th every point used to create the cone does not lie on the
        cone/ lies on the wrong side of the cone.
        Hence the first step is to shrink the choice_points by SHRINK_DISTANCE
        towards the centroid.

        Parameters
        ----------
        surface:
            where all points are expected to be excluded from at least one of its two
            one-sided cones.
        choice_points:
            An array of points that, after choosing the appropriate region,
            should all lie in the chosen region.
        control_id:
            When an ambiguity plane is needed, we ned to create a surface. if
            control_id = True, then this would force the surface to be created with
            id = 1000 + the id of the cone. This is typically only used so that
            we have full control of (and easily understandable records of)
            every surfaces' ID.
            Thus elsewhere in the code, most other classes/methods turns control_id
            on when cell_ids are also provided (proving intention on controlling IDs
            of OpenMC objects).

        Returns
        -------
        region
            openmc.Region, specifically (openmc.Halfspace) or
            (openmc.Union of 2 openmc.Halfspaces)

        Raises
        ------
        GeometryError
            cone construction invalid
        """
        # shrink to avoid floating point number comparison imprecision issues.
        # Especially important when the choice point sits exactly on the surface.
        centroid = np.mean(choice_points, axis=0)
        step_dir = centroid - choice_points
        unit_step_dir = (step_dir.T / np.linalg.norm(step_dir, axis=1)).T
        shrunken_points = choice_points + SHRINK_DISTANCE * unit_step_dir

        # # Alternative
        # shrunken_points = (choice_points + 0.01 * centroid) / 1.01
        # take one step towards the centroid = 0.1 cm

        x, y, z = np.array(to_cm(shrunken_points)).T
        values = surface.evaluate([x, y, z])
        middle = values > 0
        if all(middle):  # exist outside of cone
            return +surface

        z_dist = z - surface.z0
        upper_cone = np.logical_and(~middle, z_dist > 0)
        lower_cone = np.logical_and(~middle, z_dist < 0)

        # all of the following cases requires a plane to be made.
        if control_id:
            surface_id = 1000 + surface.id
        else:
            surface_id = max(openmc.Surface.used_ids)
            surface_id += 1000 + surface_id % 1000

        if all(upper_cone):
            # everything in the upper cone.
            # the highest we can cut is at the lowest z.
            plane = self.find_suitable_z_plane(
                to_m(surface.z0),
                to_m([surface.z0 - DTOL_CM, min(z) - DTOL_CM]),
                surface_id=surface_id,
                name=f"Ambiguity plane for cone {surface.id}",
            )
            return -surface & +plane
        if all(lower_cone):
            # everything in the lower cone
            # the lowest we can cut is at the highest z.
            plane = self.find_suitable_z_plane(
                to_m(surface.z0),
                to_m([max(z) + DTOL_CM, surface.z0 + DTOL_CM]),
                surface_id=surface_id,
                name=f"Ambiguity plane for cone {surface.id}",
            )
            return -surface & -plane
        if all(np.logical_or(upper_cone, lower_cone)):
            raise GeometryError(
                "Both cones have vertices lying inside! Cannot compute a contiguous "
                "region that works for both. Check if polygon is convex?"
            )
        # In this rare case, make its own plane.
        plane = self.find_suitable_z_plane(
            to_m(surface.z0),
            to_m([surface.z0 + DTOL_CM, surface.z0 - DTOL_CM]),
            surface_id=surface_id,
            name=f"Ambiguity plane for cone {surface.id}",
        )
        if all(np.logical_or(upper_cone, middle)):
            return +surface | +plane
        if all(np.logical_or(lower_cone, middle)):
            return +surface | -plane
        raise GeometryError("Can't have points in upper cone, lower cone AND outside!")

    def choose_region(
        self,
        surface: openmc.Surface
        | tuple[openmc.Surface]
        | tuple[openmc.Surface | openmc.ZTorus],
        vertices_array: npt.NDArray,
        *,
        control_id: bool = False,
    ) -> openmc.Region:
        """
        Pick the correct region of the surface that includes all of the points in
        vertices_array.

        Parameters
        ----------
        surface
            Either a :class:`openmc.Surface`, or a 1-tuple or 2-tuple of
            :class:`openmc.Surface`. If it is a tuple, the first element is always a
            :class:`openmc.ZPlane`/:class:`openmc.ZCone`/:class:`openmc.ZCylinder`;
            the second element (if present) is always :class:`openmc.ZTorus`.
        vertices_array
            array of shape (?, 3), that the final region should include.
        control_id
            Passed as argument onto
            :meth:`~bluemira.radiation_transport.neutronics.csg_env.OpenMCEnvironment.choose_region_cone`

        Returns
        -------
            An openmc.Region built from surface provided and includes all of these
        """
        if isinstance(surface, openmc.ZPlane | openmc.ZCylinder):
            return choose_plane_cylinders(surface, vertices_array)

        if isinstance(surface, openmc.ZCone):
            return self.choose_region_cone(
                surface, vertices_array, control_id=control_id
            )

        if isinstance(surface, tuple):
            chosen_first_region = self.choose_region(
                surface[0], vertices_array, control_id=control_id
            )
            # cone, or cylinder, or plane, or (cone | ambiguity_surface)
            if len(surface) == 1:
                return chosen_first_region
            # -surface[1] = inside of torus
            return flat_union((chosen_first_region, -surface[1]))
        raise NotImplementedError(
            f"Surface {type(surface)} is not ready for use in the axis-symmetric case!"
        )

    def region_from_surface_series(
        self,
        series_of_surfaces: Sequence[
            openmc.Surface | tuple[openmc.Surface, openmc.ZTorus | None] | None
        ],
        vertices_array: npt.NDArray,
        *,
        control_id: bool = False,
    ) -> openmc.Intersection:
        """
        Switch between choose_region() and choose_region_from_tuple_of_surfaces()
        depending on the type of each element in the series_of_surfaces.

        Parameters
        ----------
        series_of_surfaces
            Each of them can be a None, a 1-tuple of surface, a 2-tuple of surfaces, or a
            surface. For the last 3 options, see
            :func:`~bluemira.radiation_transport.neutronics.csg_env.OpenMCEnvironment.choose_region`
            for more.

        vertices_array
            array of shape (?, 3), where every single point should be included by, or at
            least on the edge of the returned Region.
        control_id
            Passed as argument onto
            :meth:`~bluemira.radiation_transport.neutronics.csg_env.OpenMCEnvironment.choose_region_cone`

        Returns
        -------
        intersection_region:
            openmc.Intersection of a list of
            [(openmc.Halfspace) or (openmc.Union of openmc.Halfspace)]
        """
        return flat_intersection([
            self.choose_region(surface, vertices_array, control_id=control_id)
            for surface in series_of_surfaces
            if surface is not None
        ])
