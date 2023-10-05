#  bluemira is an integrated inter-disciplinary design tool for future fusion
#  reactors. It incorporates several modules, some of which rely on other
#  codes, to carry out a range of typical conceptual fusion reactor design
#  activities.
#  #
#  Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                     J. Morris, D. Short
#  #
#  bluemira is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#  #
#  bluemira is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#  Lesser General Public License for more details.
#  #
#  You should have received a copy of the GNU Lesser General Public
#  License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
Useful functions for bluemira geometries.
"""

import datetime
import functools
import inspect
import json
from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
from warnings import warn

import numba as nb
import numpy as np
from scipy.spatial import ConvexHull

from bluemira.base.constants import EPS
from bluemira.base.file import force_file_extension, get_bluemira_path
from bluemira.base.look_and_feel import bluemira_debug, bluemira_warn
from bluemira.codes import _freecadapi as cadapi
from bluemira.geometry.base import BluemiraGeo, GeoMeshable
from bluemira.geometry.compound import BluemiraCompound
from bluemira.geometry.constants import D_TOLERANCE
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.error import GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.shell import BluemiraShell
from bluemira.geometry.solid import BluemiraSolid
from bluemira.geometry.wire import BluemiraWire
from bluemira.mesh import meshing


@cadapi.catch_caderr(GeometryError)
def convert(apiobj: cadapi.apiShape, label: str = "") -> BluemiraGeo:
    """Convert a FreeCAD shape into the corresponding BluemiraGeo object."""
    if isinstance(apiobj, cadapi.apiWire):
        output = BluemiraWire(apiobj, label)
    elif isinstance(apiobj, cadapi.apiFace):
        output = BluemiraFace._create(apiobj, label)
    elif isinstance(apiobj, cadapi.apiShell):
        output = BluemiraShell._create(apiobj, label)
    elif isinstance(apiobj, cadapi.apiSolid):
        output = BluemiraSolid._create(apiobj, label)
    elif isinstance(apiobj, cadapi.apiCompound):
        output = BluemiraCompound._create(apiobj, label)
    else:
        raise TypeError(f"Cannot convert {type(apiobj)} object into a BluemiraGeo.")
    return output


class BluemiraGeoEncoder(json.JSONEncoder):
    """
    JSON Encoder for BluemiraGeo.
    """

    def default(self, obj: Union[BluemiraGeo, np.ndarray, Any]):  # noqa: PLR6301
        """
        Override the JSONEncoder default object handling behaviour for BluemiraGeo.
        """
        if isinstance(obj, BluemiraGeo):
            return serialize_shape(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _reconstruct_function_call(signature, *args, **kwargs) -> dict:
    """
    Reconstruct the call of a function with inputs arguments and defaults.
    """
    data = {}

    # Inspect the function call and reconstruct defaults
    for i, key in enumerate(signature.parameters.keys()):
        if i < len(args):
            data[key] = args[i]
        elif key not in kwargs:
            value = signature.parameters[key].default
            if value != inspect._empty:
                data[key] = value
        else:
            data[key] = kwargs[key]

    # Catch any kwargs not in signature
    for k, v in kwargs.items():
        if k not in data:
            data[k] = v
    return data


def _make_debug_file(name: str) -> Path:
    """
    Make a new file in the geometry debugging folder.
    """
    path = get_bluemira_path("generated_data/naughty_geometry", subfolder="")
    now = datetime.datetime.now()
    timestamp = now.strftime("%m-%d-%Y-%H-%M")
    fmt_string = "{}-{}{}.json"
    ts_name = fmt_string.format(name, timestamp, "")
    filename = Path(path, ts_name)

    i = 0
    while filename.exists():
        i += 1
        increment = f"_{i}"
        ts_name = fmt_string.format(name, timestamp, increment)
        filename = Path(path, ts_name)
    return filename


def log_geometry_on_failure(func):
    """
    Decorator for debugging of failed geometry operations.
    """
    signature = inspect.signature(func)
    func_name = func.__name__

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except cadapi.FreeCADError:
            data = _reconstruct_function_call(signature, *args, **kwargs)
            filename = _make_debug_file(func_name)

            # Dump the data in the file
            try:
                with open(filename, "w") as file:
                    json.dump(data, file, indent=4, cls=BluemiraGeoEncoder)

                bluemira_debug(
                    f"Function call {func_name} failed. Debugging information was saved"
                    f" to: {filename}"
                )
            except Exception:  # noqa: BLE001
                bluemira_warn(
                    f"Failed to save the failed geometry operation {func_name} to JSON."
                )

            raise

    return wrapper


def fallback_to(fallback_func, exception):
    """
    Decorator for a fallback to an alternative geometry operation.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception:
                bluemira_warn(
                    f"{func.__name__} failed, falling back to {fallback_func.__name__}."
                )
                return fallback_func(*args, **kwargs)

        return wrapper

    return decorator


# # =============================================================================
# # Geometry creation
# # =============================================================================
def _make_vertex(point: Iterable[float]) -> cadapi.apiVertex:
    """
    Make a vertex.

    Parameters
    ----------
    point:
        Coordinates of the point

    Returns
    -------
    Vertex at the point
    """
    if len(point) != 3:  # noqa: PLR2004
        raise GeometryError("Points must be of dimension 3.")

    return cadapi.apiVertex(*point)


def closed_wire_wrapper(drop_closure_point: bool):
    """
    Decorator for checking / enforcing closures on wire creation functions.
    """

    def decorator(func: Callable):
        def wrapper(
            points: Union[list, np.ndarray, Dict], label: str = "", closed: bool = False
        ):
            points = Coordinates(points)
            if points.closed:
                if closed is False:
                    bluemira_warn(
                        f"{func.__name__}: input points are closed but closed=False,"
                        " defaulting to closed=True."
                    )
                closed = True
                if drop_closure_point:
                    points = Coordinates(points.xyz[:, :-1])
            wire = func(points, label=label, closed=closed)
            if closed:
                wire = cadapi.close_wire(wire)
            return BluemiraWire(wire, label=label)

        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


@closed_wire_wrapper(drop_closure_point=True)
def make_polygon(
    points: Union[list, np.ndarray],
    label: str = "",  # noqa: ARG001
    closed: bool = False,  # noqa: ARG001
) -> BluemiraWire:
    """
    Make a polygon from a set of points.

    Parameters
    ----------
    points:
        list of points. It can be given as a list of 3D tuples, a 3D numpy array,
        or similar.
    label:
        Object's label
    closed:
        if True, the first and last points will be connected in order to form a
        closed polygon. Defaults to False.

    Returns
    -------
    BluemiraWire of the polygon

    Notes
    -----
    If the input points are closed, but closed is False, the returned BluemiraWire will
    be closed.
    """
    return cadapi.make_polygon(points.T)


@closed_wire_wrapper(drop_closure_point=False)
def make_bezier(
    points: Union[list, np.ndarray],
    label: str = "",  # noqa: ARG001
    closed: bool = False,  # noqa: ARG001
) -> BluemiraWire:
    """Make a bspline from a set of points.

    Parameters
    ----------
    points:
        list of points. It can be given as a list of 3D tuples, a 3D numpy array,
        or similar.
    label:
        Object's label
    closed:
        if True, the first and last points will be connected in order to form a
        closed bspline. Defaults to False.

    Returns
    -------
    BluemiraWire that contains the bspline

    Notes
    -----
    If the input points are closed, but closed is False, the returned BluemiraWire will
    be closed.
    """
    return cadapi.make_bezier(points.T)


def make_bspline(
    poles: Union[list, np.ndarray],
    mults: Union[list, np.ndarray],
    knots: Union[list, np.ndarray],
    periodic: bool,
    degree: int,
    weights: Union[list, np.ndarray],
    check_rational: bool,
    label: str = "",
) -> BluemiraWire:
    """
    Builds a B-Spline by a lists of Poles, Mults, Knots

    Parameters
    ----------
    poles:
        list of poles.
    mults:
        list of integers for the multiplicity
    knots:
        list of knots
    periodic:
        Whether or not the spline is periodic (same curvature at start and end points)
    degree: int
        bspline degree
    weights:
        sequence of float
    check_rational:
        Whether or not to check if the BSpline is rational

    Returns
    -------
    BluemiraWire of the spline
    """
    return BluemiraWire(
        cadapi.make_bspline(
            poles, mults, knots, periodic, degree, weights, check_rational
        ),
        label=label,
    )


def _make_polygon_fallback(
    points, label="", closed=False, **kwargs  # noqa: ARG001
) -> BluemiraWire:
    """
    Overloaded function signature for fallback option from interpolate_bspline
    """
    return make_polygon(points, label, closed)


@fallback_to(_make_polygon_fallback, cadapi.FreeCADError)
@log_geometry_on_failure
def interpolate_bspline(
    points: Union[list, np.ndarray],
    label: str = "",
    closed: bool = False,
    start_tangent: Optional[Iterable] = None,
    end_tangent: Optional[Iterable] = None,
) -> BluemiraWire:
    """
    Make a bspline from a set of points.

    Parameters
    ----------
    points:
        list of points. It can be given as a list of 3D tuples, a 3D numpy array,
        or similar.
    label:
        Object's label
    closed:
        if True, the first and last points will be connected in order to form a
        closed bspline. Defaults to False.
    start_tangent:
        Tangency of the BSpline at the first pole. Must be specified with end_tangent
    end_tangent:
        Tangency of the BSpline at the last pole. Must be specified with start_tangent

    Returns
    -------
    Bluemira wire that contains the bspline
    """
    points = Coordinates(points)
    return BluemiraWire(
        cadapi.interpolate_bspline(points.T, closed, start_tangent, end_tangent),
        label=label,
    )


def force_wire_to_spline(
    wire: BluemiraWire,
    n_edges_max: int = 200,
    l2_tolerance: float = 5e-3,
) -> BluemiraWire:
    """
    Force a wire to be a spline wire.

    Parameters
    ----------
    wire:
        The BluemiraWire to be turned into a splined wire
    n_edges_max:
        The maximum number of edges in the wire, below which this operation
        does nothing
    l2_tolerance:
        The L2-norm difference w.r.t. the original wire, above which this
        operation will warn that the desired tolerance was not achieved.

    Returns
    -------
    A new spline version of the wire

    Notes
    -----
    This is intended for use with wires that consist of large polygons, often resulting
    from operations that failed with primitives and fallback methods making use of
    of polygons. This can be relatively stubborn to transform back to splines.
    """
    original_n_edges = len(wire.edges)
    if original_n_edges < n_edges_max:
        bluemira_debug(
            f"Wire already has {original_n_edges} < {n_edges_max=}. No point forcing to"
            " a spline."
        )
        return wire

    original_points = wire.discretize(ndiscr=2 * original_n_edges, byedges=False)

    for n_discr in np.array(original_n_edges * np.linspace(0.8, 0.1, 8), dtype=int):
        points = wire.discretize(ndiscr=int(n_discr), byedges=False)
        try:
            wire = BluemiraWire(
                cadapi.interpolate_bspline(points.T, closed=wire.is_closed()),
                label=wire.label,
            )
            break
        except cadapi.FreeCADError:
            continue

    new_points = wire.discretize(ndiscr=2 * original_n_edges, byedges=False)

    delta = np.linalg.norm(original_points.xyz - new_points.xyz, ord=2)
    if delta > l2_tolerance:
        bluemira_warn(
            f"Forcing wire to spline with {n_discr} interpolation points did not achieve"
            f" the desired tolerance: {delta} > {l2_tolerance}"
        )

    return wire


def make_circle(
    radius: float = 1.0,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    start_angle: float = 0.0,
    end_angle: float = 360.0,
    axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    label: str = "",
) -> BluemiraWire:
    """
    Create a circle or arc of circle object with given parameters.

    Parameters
    ----------
    radius:
        Radius of the circle
    center:
        Center of the circle
    start_angle:
        Start angle of the arc [degrees]
    end_angle:
        End angle of the arc [degrees]. If start_angle == end_angle, a circle is created,
        otherwise a circle arc is created
    axis:
        Normal vector to the circle plane. It defines the clockwise/anticlockwise
        circle orientation according to the right hand rule.
    label:
        object's label

    Returns
    -------
    Bluemira wire that contains the arc or circle
    """
    output = cadapi.make_circle(radius, center, start_angle, end_angle, axis)
    return BluemiraWire(output, label=label)


def make_circle_arc_3P(  # noqa: N802
    p1: Iterable[float], p2: Iterable[float], p3: Iterable[float], label: str = ""
):
    """
    Create an arc of circle object given three points.

    Parameters
    ----------
    p1:
        Starting point of the circle arc
    p2:
        Middle point of the circle arc
    p3:
        End point of the circle arc

    Returns
    -------
    Bluemira wire that contains the arc or circle
    """
    # TODO: check what happens when the 3 points are in a line
    output = cadapi.make_circle_arc_3P(p1, p2, p3)
    return BluemiraWire(output, label=label)


def make_ellipse(
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    major_radius: float = 2.0,
    minor_radius: float = 1.0,
    major_axis: Tuple[float, float, float] = (1, 0, 0),
    minor_axis: Tuple[float, float, float] = (0, 1, 0),
    start_angle: float = 0.0,
    end_angle: float = 360.0,
    label: str = "",
) -> BluemiraWire:
    """
    Create an ellipse or arc of ellipse object with given parameters.

    Parameters
    ----------
    center:
        Center of the ellipse
    major_radius:
        Major radius of the ellipse
    minor_radius:
        Minor radius of the ellipse (float). Default to 2.
    major_axis:
        Major axis direction
    minor_axis:
        Minor axis direction
    start_angle:
        Start angle of the arc [degrees]
    end_angle:
        End angle of the arc [degrees].  if start_angle == end_angle, an ellipse is
        created, otherwise a ellipse arc is created
    label:
        Object's label

    Returns
    -------
    Bluemira wire that contains the arc or ellipse
    """
    output = cadapi.make_ellipse(
        center,
        major_radius,
        minor_radius,
        major_axis,
        minor_axis,
        start_angle,
        end_angle,
    )
    return BluemiraWire(output, label=label)


def wire_closure(bmwire: BluemiraWire, label="closure") -> BluemiraWire:
    """
    Close this wire with a line segment

    Parameters
    ----------
    bmwire:
        supporting wire for the closure
    label:
        Object's label

    Returns
    -------
    Closure wire
    """
    wire = bmwire.shape
    return BluemiraWire(cadapi.wire_closure(wire), label=label)


def _offset_wire_discretised(
    wire,
    thickness,
    /,
    join: str = "intersect",  # noqa: ARG001
    open_wire: bool = True,
    label="",
    *,
    fallback_method="square",
    byedges=True,
    ndiscr=200,
    **fallback_kwargs,
) -> BluemiraWire:
    """
    Fallback function for discretised offsetting

    Raises
    ------
    GeometryError
        If the wire is not closed. This function cannot handle the offet of an open
        wire.
    """
    from bluemira.geometry._pyclipper_offset import offset_clipper

    if not wire.is_closed() and not open_wire:
        wire = wire.deepcopy()
        wire.close()

    if not wire.is_closed() and open_wire:
        raise GeometryError(
            "Fallback function _offset_wire_discretised cannot handle open wires."
        )

    coordinates = wire.discretize(byedges=byedges, ndiscr=ndiscr)

    result = offset_clipper(
        coordinates, thickness, method=fallback_method, **fallback_kwargs
    )
    return make_polygon(result, label=label, closed=True)


@fallback_to(_offset_wire_discretised, cadapi.FreeCADError)
@log_geometry_on_failure
def offset_wire(
    wire: BluemiraWire,
    thickness: float,
    /,
    join: str = "intersect",
    open_wire: bool = True,
    label: str = "",
    *,
    fallback_method="square",  # noqa: ARG001
    byedges=True,  # noqa: ARG001
    ndiscr=400,  # noqa: ARG001
    **fallback_kwargs,  # noqa: ARG001
) -> BluemiraWire:
    """
    Make a planar offset from a planar wire.

    Parameters
    ----------
    wire:
        Wire to offset from
    thickness:
        Offset distance. Positive values outwards, negative values inwards
    join:
        Offset method. "arc" gives rounded corners, and "intersect" gives sharp corners
    open_wire:
        For open wires (counter-clockwise default) whether or not to make an open offset
        wire, or a closed offset wire that encompasses the original wire. This is
        disabled for closed wires.

    Other Parameters
    ----------------
    byedges:
        Whether or not to discretise the wire by edges
    ndiscr:
        Number of points to discretise the wire to
    fallback_method:
        Method to use in discretised offsetting, will default to `square` as `round`
        is know to be very slow

    Notes
    -----
    If primitive offsetting failed, will fall back to a discretised offset
    implementation, where the fallback kwargs are used. Discretised offsetting is
    only supported for closed wires.

    Returns
    -------
    Offset wire
    """
    return BluemiraWire(
        cadapi.offset_wire(wire.shape, thickness, join, open_wire), label=label
    )


def convex_hull_wires_2d(
    wires: Sequence[BluemiraWire], ndiscr: int, plane: str = "xz"
) -> BluemiraWire:
    """
    Perform a convex hull around the given wires and return the hull
    as a new wire.

    The operation performs discretisations on the input wires.

    Parameters
    ----------
    wires:
        The wires to draw a hull around.
    ndiscr:
        The number of points to discretise each wire into.
    plane:
        The plane to perform the hull in. One of: 'xz', 'xy', 'yz'.
        Default is 'xz'.

    Returns
    -------
    A wire forming a convex hull around the input wires in the given
    plane.
    """
    if not wires:
        raise ValueError("Must have at least one wire to draw a hull around.")
    if plane == "xz":
        plane_idxs = (0, 2)
    elif plane == "xy":
        plane_idxs = (0, 1)
    elif plane == "yz":
        plane_idxs = (1, 2)
    else:
        raise ValueError(f"Invalid plane: '{plane}'. Must be one of 'xz', 'xy', 'yz'.")

    shape_discretizations = []
    for wire in wires:
        discretized_points = wire.discretize(byedges=True, ndiscr=ndiscr)
        shape_discretizations.append(getattr(discretized_points, plane))
    coords = np.hstack(shape_discretizations)

    hull = ConvexHull(coords.T)
    hull_coords = np.zeros((3, len(hull.vertices)))
    hull_coords[plane_idxs, :] = coords[:, hull.vertices]
    return make_polygon(hull_coords, closed=True)


# # =============================================================================
# # Shape operation
# # =============================================================================
def revolve_shape(
    shape: BluemiraGeo,
    base: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    degree: float = 180,
    label: str = "",
) -> BluemiraGeo:
    """
    Apply the revolve (base, dir, degree) to this shape

    Parameters
    ----------
    shape:
        The shape to be revolved
    base:
        Origin location of the revolution
    direction:
        The direction vector
    degree:
        revolution angle

    Returns
    -------
    The revolved shape.
    """
    if degree > 360:  # noqa: PLR2004
        bluemira_warn("Cannot revolve a shape by more than 360 degrees.")
        degree = 360

    if degree == 360:  # noqa: PLR2004
        # We split into two separate revolutions of 180 degree and fuse them
        if isinstance(shape, BluemiraWire):
            shape = BluemiraFace(shape).shape
            flag_shell = True
        else:
            shape = shape.shape
            flag_shell = False

        shape_1 = cadapi.revolve_shape(shape, base, direction, degree=180)
        shape_2 = deepcopy(shape_1)
        shape_2 = cadapi.rotate_shape(shape_2, base, direction, degree=-180)
        result = cadapi.boolean_fuse([shape_1, shape_2], remove_splitter=False)

        if flag_shell:
            result = result.Shells[0]

        return convert(result, label)

    return convert(cadapi.revolve_shape(shape.shape, base, direction, degree), label)


def extrude_shape(
    shape: BluemiraGeo, vec: Tuple[float, float, float], label=""
) -> BluemiraSolid:
    """
    Apply the extrusion along vec to this shape

    Parameters
    ----------
    shape:
        The shape to be extruded
    vec:
        The vector along which to extrude
    label:
        label of the output shape

    Returns
    -------
    The extruded shape.
    """
    if not label:
        label = shape.label

    return convert(cadapi.extrude_shape(shape.shape, vec), label)


def sweep_shape(
    profiles: Union[BluemiraWire, Iterable[BluemiraWire]],
    path: BluemiraWire,
    solid: bool = True,
    frenet: bool = True,
    label: str = "",
) -> Union[BluemiraSolid, BluemiraShell]:
    """
    Sweep a profile along a path.

    Parameters
    ----------
    profiles:
        Profile(s) to sweep
    path:
        Path along which to sweep the profiles
    solid:
        Whether or not to create a Solid
    frenet:
        If true, the orientation of the profile(s) is calculated based on local curvature
        and tangency. For planar paths, should not make a difference.

    Returns
    -------
    Swept geometry object
    """
    if not isinstance(profiles, Iterable):
        profiles = [profiles]

    profile_shapes = [p.shape for p in profiles]

    result = cadapi.sweep_shape(profile_shapes, path.shape, solid, frenet)

    return convert(result, label=label)


def fillet_chamfer_decorator(chamfer: bool):
    """
    Decorator for fillet and chamfer operations, checking for validity of wire
    and radius.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(wire, radius):
            edges = wire.shape.OrderedEdges
            func_name = "chamfer" if chamfer else "fillet"
            if len(edges) < 2:  # noqa: PLR2004
                raise GeometryError(f"Cannot {func_name} a wire with less than 2 edges!")
            if not cadapi._wire_is_planar(wire.shape):
                raise GeometryError(f"Cannot {func_name} a non-planar wire!")
            if radius == 0:
                return wire.deepcopy()
            if radius < 0:
                raise GeometryError(
                    f"Cannot {func_name} a wire with a negative {radius=}"
                )
            return func(wire, radius)

        return wrapper

    return decorator


@fillet_chamfer_decorator(False)
def fillet_wire_2D(wire: BluemiraWire, radius: float) -> BluemiraWire:
    """
    Fillet all edges of a wire

    Parameters
    ----------
    wire:
        Wire to fillet
    radius:
        Radius of the fillet operation

    Returns
    -------
    The filleted wire
    """
    return BluemiraWire(cadapi.fillet_wire_2D(wire.shape, radius))


@fillet_chamfer_decorator(True)
def chamfer_wire_2D(wire: BluemiraWire, radius: float):
    """
    Chamfer all edges of a wire

    Parameters
    ----------
    wire:
        Wire to chamfer
    radius:
        Radius of the chamfer operation

    Returns
    -------
    The chamfered wire
    """
    return BluemiraWire(cadapi.fillet_wire_2D(wire.shape, radius, chamfer=True))


def distance_to(
    geo1: Union[Iterable[float], BluemiraGeo], geo2: Union[Iterable[float], BluemiraGeo]
) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    Calculate the distance between two BluemiraGeos.

    Parameters
    ----------
    geo1:
        Reference shape. If an iterable of length 3, converted to a point.
    geo2:
        Target shape. If an iterable of length 3, converted to a point.

    Returns
    -------
    dist:
        Minimum distance
    vectors:
        List of tuples corresponding to the nearest points between geo1 and geo2. The
        distance between those points is the minimum distance given by dist.
    """
    # Check geometry for vertices
    shape1 = _make_vertex(geo1) if isinstance(geo1, Iterable) else geo1.shape
    shape2 = _make_vertex(geo2) if isinstance(geo2, Iterable) else geo2.shape
    return cadapi.dist_to_shape(shape1, shape2)


def split_wire(
    wire: BluemiraWire, vertex: Iterable[float], tolerance: float = EPS * 10
) -> Tuple[Union[None, BluemiraWire], Union[None, BluemiraWire]]:
    """
    Split a wire at a given vertex.

    Parameters
    ----------
    wire:
        Wire to be split
    vertex:
        Vertex at which to split the wire
    tolerance:
        Tolerance within which to find the closest vertex on the wire

    Returns
    -------
    wire_1:
        First half of the wire. Will be None if the vertex is the start point of the wire
    wire_2:
        Last half of the wire. Will be None if the vertex is the start point of the wire

    Raises
    ------
    GeometryError:
        If the vertex is further away to the wire than the specified tolerance
    """
    wire_1, wire_2 = cadapi.split_wire(wire.shape, vertex, tolerance=tolerance)
    if wire_1:
        wire_1 = BluemiraWire(wire_1)
    if wire_2:
        wire_2 = BluemiraWire(wire_2)
    return wire_1, wire_2


def slice_shape(
    shape: BluemiraGeo, plane: BluemiraPlane
) -> Union[List[np.ndarray], None, List[BluemiraWire]]:
    """
    Calculate the plane intersection points with an object

    Parameters
    ----------
    shape:
        Shape to intersect with a plane
    plane:
        Plane to intersect with

    Returns
    -------
    Wire: Union[List[np.ndarray], None]
        returns array of intersection points
    Face, Solid, Shell: Union[List[BluemiraWire], None]
        list of intersections lines

    Notes
    -----
    Degenerate cases such as tangets to solid or faces do not return intersections
    if the shape and plane are acting at the Plane base.
    Further investigation needed.

    """
    _slice = cadapi.slice_shape(shape.shape, plane.base, plane.axis)

    if isinstance(_slice, np.ndarray) and _slice.size > 0:
        return _slice

    _slice = [convert(obj) for obj in _slice]

    if len(_slice) > 0:
        return _slice
    return None


def circular_pattern(
    shape: BluemiraGeo,
    origin: Tuple[float, float, float] = (0, 0, 0),
    direction: Tuple[float, float, float] = (0, 0, 1),
    degree: float = 360,
    n_shapes: int = 10,
) -> List[BluemiraGeo]:
    """
    Make a equally spaced circular pattern of shapes.

    Parameters
    ----------
    shape:
        Shape to pattern
    origin:
        Origin vector of the circular pattern
    direction:
        Direction vector of the circular pattern
    degree:
        Angle range of the patterning
    n_shapes:
        Number of shapes to pattern

    Returns
    -------
    List of patterned shapes, the first element is the original shape
    """
    angle = degree / n_shapes

    shapes = [shape]
    for i in range(1, n_shapes):
        new_shape = shape.deepcopy()
        new_shape.rotate(origin, direction, i * angle)
        shapes.append(new_shape)
    return shapes


def mirror_shape(
    shape: BluemiraGeo, base: tuple, direction: tuple, label=""
) -> BluemiraGeo:
    """
    Get a mirrored copy of a shape about a plane.

    Parameters
    ----------
    shape:
        Shape to mirror
    base:
        Mirror plane base
    direction:
        Mirror plane normal direction

    Returns
    -------
    The mirrored shape

    Raises
    ------
    GeometryError: if the norm of the direction tuple is <= 3*EPS
    """
    if np.linalg.norm(direction) <= 3 * EPS:
        raise GeometryError("Direction vector cannot have a zero norm.")
    return convert(cadapi.mirror_shape(shape.shape, base, direction), label=label)


# # =============================================================================
# # Save functions
# # =============================================================================
def save_as_STP(
    shapes: Union[BluemiraGeo, Iterable[BluemiraGeo]],
    filename: str,
    unit_scale: str = "metre",
    **kwargs,
):
    """
    Saves a series of Shape objects as a STEP assembly

    Parameters
    ----------
    shapes:
        List of shape objects to be saved
    filename:
        Full path filename of the STP assembly
    unit_scale:
        The scale in which to save the Shape objects
    """
    filename = force_file_extension(filename, [".stp", ".step"])

    if not isinstance(shapes, list):
        shapes = [shapes]

    cadapi.save_as_STP([s.shape for s in shapes], filename, unit_scale, **kwargs)


def save_cad(
    shapes: Union[BluemiraGeo, List[BluemiraGeo]],
    filename: str,
    cad_format: Union[str, cadapi.CADFileType] = "stp",
    names: Optional[Union[str, List[str]]] = None,
    **kwargs,
):
    """
    Save the CAD of a component (eg a reactor) or a list of components

    Parameters
    ----------
    shapes:
        shapes to save
    filename:
        Full path filename of the STP assembly
    cad_format:
        file format to save as
    names:
        Names of shapes to save
    kwargs:
        arguments passed to cadapi save function
    """
    if kw_formatt := kwargs.pop("formatt", None):
        warn(
            "Using kwarg 'formatt' is no longer supported. Use cad_format instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        cad_format = kw_formatt

    if not isinstance(shapes, list):
        shapes = [shapes]
    if names is not None and not isinstance(names, list):
        names = [names]

    cadapi.save_cad(
        [s.shape for s in shapes],
        filename,
        cad_format=cad_format,
        labels=names,
        **kwargs,
    )


# ======================================================================================
# Signed distance functions
# ======================================================================================


@nb.jit(nopython=True, cache=True)
def _nb_dot_2D(v_1, v_2):
    """
    Numba 2-D dot product
    """
    return v_1[0] * v_2[0] + v_1[1] * v_2[1]


@nb.jit(nopython=True, cache=True)
def _nb_clip(val, a_min, a_max):
    """
    Numba 1-D clip
    """
    return a_min if val < a_min else a_max if val > a_max else val


@nb.jit(nopython=True, cache=True)
def _signed_distance_2D(point: np.ndarray, polygon: np.ndarray) -> float:
    """
    2-D function for the signed distance from a point to a polygon. The return value is
    negative if the point is outside the polygon, and positive if the point is inside the
    polygon.

    Parameters
    ----------
    point:
        2-D point
    polygon:
        2-D set of points (closed)

    Returns
    -------
    Signed distance value of the point to the polygon

    Notes
    -----
    Credit: Inigo Quilez (https://www.iquilezles.org/)
    """
    sign = -1.0
    point = np.asfarray(point)
    polygon = np.asfarray(polygon)
    n = len(polygon)

    d = _nb_dot_2D(point - polygon[0], point - polygon[0])

    for i in range(n - 1):
        j = i + 1
        e = polygon[j] - polygon[i]
        w = point - polygon[i]
        b = w - e * _nb_clip(_nb_dot_2D(w, e) / _nb_dot_2D(e, e), 0.0, 1.0)
        d_new = _nb_dot_2D(b, b)
        if d_new < d:
            d = d_new

        cond = np.array(
            [
                point[1] >= polygon[i][1],
                point[1] < polygon[j][1],
                e[0] * w[1] > e[1] * w[0],
            ]
        )
        if np.all(cond) or np.all(~cond):
            sign = -sign

    return sign * np.sqrt(d)


@nb.jit(nopython=True, cache=True)
def signed_distance_2D_polygon(
    subject_poly: np.ndarray, target_poly: np.ndarray
) -> np.ndarray:
    """
    2-D vector-valued signed distance function from a subject polygon to a target
    polygon. The return values are negative for points outside the target polygon, and
    positive for points inside the target polygon.

    Parameters
    ----------
    subject_poly:
        Subject 2-D polygon
    target_poly:
        Target 2-D polygon (closed polygons only)

    Returns
    -------
    Signed distances from the vertices of the subject polygon to the target polygon

    Notes
    -----
    This can used as a keep-out-zone constraint, in which the target polygon would
    be the keep-out-zone, and the subject polygon would be the shape which must
    be outsize of the keep-out-zone.

    The target polygon must be closed.
    """
    m = len(subject_poly)
    d = np.zeros(m)

    for i in range(m):
        d[i] = _signed_distance_2D(subject_poly[i], target_poly)

    return d


def signed_distance(wire_1: BluemiraWire, wire_2: BluemiraWire) -> float:
    """
    Single-valued signed "distance" function between two wires. Will return negative
    values if wire_1 does not touch or intersect wire_2, 0 if there is one intersection,
    and a positive estimate of the intersection length if there are overlaps.

    Parameters
    ----------
    wire_1:
        Subject wire
    wire_2:
        Target wire

    Returns
    -------
    Signed distance from wire_1 to wire_2

    Notes
    -----
    This is not a pure implementation of a distance function, as for overlapping wires a
    metric of the quantity of overlap is returned (a positive value). This nevertheless
    enables the use of such a function as a constraint in gradient-based optimisers.
    """
    d, vectors = distance_to(wire_1, wire_2)
    # Intersections are exactly 0.0
    if d == 0.0:  # noqa: PLR2004
        if len(vectors) <= 1:
            # There is only one intersection: the wires are touching but not overlapping
            return 0.0
        # There are multiple intersections: the wires are overlapping
        # For now, without boolean operations, get an estimate of the intersection
        # length
        length = 0
        for i in range(1, len(vectors)):
            p1 = vectors[i - 1][0]
            p2 = vectors[i][0]

            length += np.sqrt(
                (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2
            )

        # TODO: Use a boolean difference operation to get the lengths of the
        # overlapping wire segment(s)
        return length
    # There are no intersections, return minimum distance
    return -d


# ======================================================================================
# Boolean operations
# ======================================================================================
def boolean_fuse(shapes: Iterable[BluemiraGeo], label: str = "") -> BluemiraGeo:
    """
    Fuse two or more shapes together. Internal splitter are removed.

    Parameters
    ----------
    shapes:
        List of shape objects to be saved
    label:
        Label for the resulting shape

    Returns
    -------
    Result of the boolean operation.

    Raises
    ------
    error: GeometryError
        In case the boolean operation fails.
    """
    if not isinstance(shapes, list):
        raise TypeError(f"{shapes} is not a list.")

    if len(shapes) < 2:  # noqa: PLR2004
        raise ValueError("At least 2 shapes must be given")

    # check that all the shapes are of the same time
    _type = type(shapes[0])
    if not all(isinstance(s, _type) for s in shapes):
        raise ValueError(f"All instances in {shapes} must be of the same type.")

    api_shapes = [s.shape for s in shapes]
    try:
        merged_shape = cadapi.boolean_fuse(api_shapes)
        return convert(merged_shape, label)

    except Exception as e:  # noqa: BLE001
        raise GeometryError(f"Boolean fuse operation failed: {e}") from None


def boolean_cut(
    shape: BluemiraGeo, tools: Iterable[BluemiraGeo]
) -> Iterable[BluemiraGeo]:
    """
    Difference of shape and a given (list of) topo shape cut(tools)

    Parameters
    ----------
    shape:
        the reference object
    tools:
        List of BluemiraGeo shape objects to be used as tools.

    Returns
    -------
    Result of the boolean cut operation.

    Raises
    ------
    error: GeometryError
        In case the boolean operation fails.
    """
    apishape = shape.shape
    if not isinstance(tools, list):
        tools = [tools]
    apitools = [t.shape for t in tools]
    cut_shape = cadapi.boolean_cut(apishape, apitools)

    if isinstance(cut_shape, Iterable):
        return [convert(obj, shape.label) for obj in cut_shape]

    return convert(cut_shape, shape.label)


def boolean_fragments(
    shapes: List[BluemiraSolid], tolerance: float = 0.0
) -> Tuple[BluemiraCompound, List[List[BluemiraSolid]]]:
    """
    Split a list of shapes into their Boolean fragments.

    Parameters
    ----------
    shapes:
        List of BluemiraSolids to be split into Boolean fragments
    tolerance:
        Tolerance with which to perform the operation

    Returns
    -------
    compound:
        A compound of the unique fragments
    fragment_map:
        An ordered list of groups of solid Boolean fragments (ordered in terms of
        input ordering)

    Notes
    -----
    Labelling will be lost.
    This function is only tested on solids.
    """
    compound, fragment_map = cadapi.boolean_fragments(
        [s.shape for s in shapes], tolerance
    )
    converted = [[convert(s) for s in group] for group in fragment_map]

    return convert(compound), converted


def point_inside_shape(point: Iterable[float], shape: BluemiraGeo) -> bool:
    """
    Check whether or not a point is inside a shape.

    Parameters
    ----------
    point:
        Coordinates of the point
    shape:
        Geometry to check with

    Returns
    -------
    Whether or not the point is inside the shape
    """
    return cadapi.point_inside_shape(point, shape.shape)


def point_on_plane(
    point: Iterable[float], plane: BluemiraPlane, tolerance: float = D_TOLERANCE
) -> bool:
    """
    Check whether or not a point is on a plane.

    Parameters
    ----------
    point:
        Coordinates of the point
    plane:
        Plane to check
    tolerance:
        Tolerance with which to check

    Returns
    -------
    Whether or not the point is on the plane
    """
    return (
        abs(
            cadapi.apiVector(point).distanceToPlane(
                plane.shape.Position, plane.shape.Axis
            )
        )
        < tolerance
    )


# # =============================================================================
# # Serialize and Deserialize
# # =============================================================================
def serialize_shape(shape: BluemiraGeo):
    """
    Serialize a BluemiraGeo object.
    """
    type_ = type(shape)

    output = []
    if isinstance(shape, BluemiraGeo):
        sdict = {"label": shape.label, "boundary": output}
        for obj in shape.boundary:
            output.append(serialize_shape(obj))
            if isinstance(shape, GeoMeshable) and shape.mesh_options is not None:
                if shape.mesh_options.lcar is not None:
                    sdict["lcar"] = shape.mesh_options.lcar
                if shape.mesh_options.physical_group is not None:
                    sdict["physical_group"] = shape.mesh_options.physical_group
        return {str(type(shape).__name__): sdict}
    if isinstance(shape, cadapi.apiWire):
        return cadapi.serialize_shape(shape)
    raise NotImplementedError(f"Serialization non implemented for {type_}")


def deserialize_shape(buffer: dict):
    """
    Deserialize a BluemiraGeo object obtained from serialize_shape.

    Parameters
    ----------
    buffer:
        Object serialization as stored by serialize_shape

    Returns
    -------
    The deserialized BluemiraGeo object.
    """
    supported_types = [BluemiraWire, BluemiraFace, BluemiraShell]

    def _extract_mesh_options(shape_dict: dict):
        mesh_options = None
        if "lcar" in shape_dict:
            mesh_options = meshing.MeshOptions()
            mesh_options.lcar = shape_dict["lcar"]
        if "physical_group" in shape_dict:
            mesh_options = mesh_options or meshing.MeshOptions()
            mesh_options.physical_group = shape_dict["physical_group"]
        return mesh_options

    def _extract_shape(shape_dict: dict, shape_type: Type[BluemiraGeo]):
        label = shape_dict["label"]
        boundary = shape_dict["boundary"]

        temp_list = []
        for item in boundary:
            if issubclass(shape_type, BluemiraWire):
                for k in item:
                    if k == shape_type.__name__:
                        shape = deserialize_shape(item)
                    else:
                        shape = cadapi.deserialize_shape(item)
                    temp_list.append(shape)
            else:
                temp_list.append(deserialize_shape(item))

        mesh_options = _extract_mesh_options(shape_dict)

        shape = shape_type(label=label, boundary=temp_list)
        if mesh_options is not None:
            shape.mesh_options = mesh_options
        return shape

    for type_, v in buffer.items():
        for supported_type in supported_types:
            if type_ == supported_type.__name__:
                return _extract_shape(v, BluemiraWire)

        raise NotImplementedError(f"Deserialization non implemented for {type_}")
    return None


# # =============================================================================
# # shape utils
# # =============================================================================
def get_shape_by_name(shape: BluemiraGeo, name: str) -> List[BluemiraGeo]:
    """
    Search through the boundary of the shape and get any shapes with a label
    corresponding to the provided name. Includes the shape itself if the name matches
    its label.

    Parameters
    ----------
    shape:
        The shape to search for the provided name.
    name:
        The name to search for.

    Returns
    -------
    The shapes known to the provided shape that correspond to the provided name.
    """
    shapes = []
    if hasattr(shape, "label") and shape.label == name:
        shapes.append(shape)
    if hasattr(shape, "boundary"):
        for o in shape.boundary:
            shapes += get_shape_by_name(o, name)
    return shapes


# ======================================================================================
# Find operations
# ======================================================================================
def find_clockwise_angle_2d(base: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Find the clockwise angle between the 2D vectors ``base`` and
    ``vector`` in the range [0°, 360°).

    Parameters
    ----------
    base:
        The vector to start the angle from.
    vector:
        The vector to end the angle at.

    Returns
    -------
    The clockwise angle between the two vectors in degrees.
    """
    if not isinstance(base, np.ndarray) or not isinstance(vector, np.ndarray):
        raise TypeError(
            f"Input vectors must have type np.ndarray, found '{type(base)}' and "
            f"'{type(vector)}'."
        )
    if base.shape[0] != 2 or vector.shape[0] != 2:  # noqa: PLR2004
        raise ValueError(
            f"Input vectors' axis 0 length must be 2, found shapes '{base.shape}' and "
            f"'{vector.shape}'."
        )
    det = base[1] * vector[0] - base[0] * vector[1]
    dot = np.dot(base, vector)
    # Convert to array here in case arctan2 returns a scalar
    angle = np.array(np.arctan2(det, dot))
    angle[angle < 0] += 2 * np.pi
    return np.degrees(angle)
