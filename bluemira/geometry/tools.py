# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Useful functions for bluemira geometries.
"""

from __future__ import annotations

import datetime
import enum
import functools
import inspect
import json
from collections.abc import Callable, Iterable, Sequence
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numba as nb
import numpy as np
from numpy import typing as npt
from scipy.spatial import ConvexHull

from bluemira.base.constants import EPS
from bluemira.base.file import force_file_extension, try_get_bluemira_path
from bluemira.base.logs import LogLevel, get_log_level
from bluemira.base.look_and_feel import bluemira_debug, bluemira_warn
from bluemira.codes import _freecadapi as cadapi
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.compound import BluemiraCompound
from bluemira.geometry.constants import D_TOLERANCE
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.error import GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.shell import BluemiraShell
from bluemira.geometry.solid import BluemiraSolid
from bluemira.geometry.wire import BluemiraWire
from bluemira.mesh import meshing
from bluemira.utilities.tools import iterable_to_list

if TYPE_CHECKING:
    from bluemira.geometry.base import BluemiraGeoT
    from bluemira.geometry.plane import BluemiraPlane


@cadapi.catch_caderr(GeometryError)
def convert(apiobj: cadapi.apiShape, label: str = "") -> BluemiraGeoT:
    """Convert a FreeCAD shape into the corresponding BluemiraGeo object.

    Raises
    ------
    TypeError
        Cannot convert to BluemiraGeo

    Returns
    -------
    :
        BluemiraGeo object of inputted FreeCAD shape.
    """
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


class HullPlane(Enum):
    """
    Enumeration of planes to perform a hull operation in.
    """

    XZ = auto()
    XY = auto()
    YZ = auto()

    @classmethod
    def _missing_(cls, value: str):
        if not isinstance(value, str):
            raise TypeError(f"Invalid hull plane: {value}. Expected str.")
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid hull plane: {value}. Choose from: {(*cls._member_names_,)}"
            ) from None


class BluemiraGeoEncoder(json.JSONEncoder):
    """
    JSON Encoder for BluemiraGeo.
    """

    def default(self, obj: BluemiraGeoT | np.ndarray | Any):
        """
        Override the JSONEncoder default object handling behaviour for BluemiraGeo.

        Returns
        -------
        :
            List from bluemira shape or numpy array.
        """
        if isinstance(obj, BluemiraGeo):
            return serialise_shape(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _reconstruct_function_call(signature, *args, **kwargs) -> dict:
    """
    Reconstruct the call of a function with inputs arguments and defaults.

    Returns
    -------
    :
        Dictionary of data from a function with inputs arguments and defaults.
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

    Returns
    -------
    :
        Filename for new file in the geometry debugging folder.
    """
    path = try_get_bluemira_path("generated_data/naughty_geometry", subfolder="")

    if path is None:
        path = Path.cwd() / "naughty_geometry"
        Path.mkdir(path, exist_ok=True)

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

    Returns
    -------
    :
        Decorator for debugging of failed geometry operations.
    """
    signature = inspect.signature(func)
    func_name = func.__name__

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except cadapi.FreeCADError:
            # Dump the data in the file
            if LogLevel(int(get_log_level(as_str=False) * 10)) == LogLevel.DEBUG:
                data = _reconstruct_function_call(signature, *args, **kwargs)
                filename = _make_debug_file(func_name)
                try:
                    with open(filename, "w") as file:
                        json.dump(data, file, indent=4, cls=BluemiraGeoEncoder)

                    bluemira_debug(
                        f"Function call {func_name} failed."
                        f" Debugging information was saved to: {filename}"
                    )
                except Exception:  # noqa: BLE001
                    bluemira_warn(
                        "Failed to save the failed geometry operation"
                        f" {func_name} to JSON."
                    )

            else:
                bluemira_warn(
                    "Set logging level to debug to save the geometry"
                    "and other function arguments to json"
                )

            raise

    return wrapper


def fallback_to(fallback_func, exception):
    """
    Decorator for a fallback to an alternative geometry operation.

    Returns
    -------
    :
        Decorator for fallback function.
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
def _make_vertex(point: npt.ArrayLike | Coordinates) -> cadapi.apiVertex:
    """
    Make a vertex.

    Parameters
    ----------
    point:
        Coordinates of the point

    Returns
    -------
    Vertex at the point

    Raises
    ------
    GeometryError
        Shape must a single point
    """
    if isinstance(point, Coordinates):
        if np.shape(point) != (3, 1):
            raise GeometryError(
                "Can only cast the 3D coordinates of a single pointinto a cadapi vertex!"
            )
        point = point.points[0]
    if len(point) != 3:  # noqa: PLR2004
        raise GeometryError("Points must be of dimension 3.")

    return cadapi.apiVertex(*point)


class GeometryCreationIn(Protocol):
    """Typing for closed_wire_wrapper input"""

    def __call__(
        self,
        points: Coordinates,
        label: str = "",
        *,
        closed: bool = False,
    ) -> BluemiraWire:
        """Typing for coordinates wrapping"""
        ...


class GeometryCreationOut(Protocol):
    """Typing for closed_wire_wrapper output"""

    def __call__(
        self,
        points: Coordinates | npt.ArrayLike | dict[str, npt.ArrayLike],
        label: str = "",
        *,
        closed: bool = False,
    ) -> BluemiraWire:
        """Typing for coordinates wrapping"""
        ...


def closed_wire_wrapper(
    *, drop_closure_point: bool
) -> Callable[[GeometryCreationIn], GeometryCreationOut]:
    """
    Decorator for checking / enforcing closures on wire creation functions.

    Returns
    -------
    :
        Decorator on wire creation functions.
    """

    def decorator(func: GeometryCreationIn) -> GeometryCreationOut:
        @functools.wraps(func)
        def wrapper(
            points: npt.ArrayLike | dict[str, npt.ArrayLike] | Coordinates,
            label: str = "",
            *,
            closed: bool = False,
        ) -> BluemiraWire:
            if not isinstance(points, Coordinates):
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
            wire: cadapi.apiWire = func(points, label=label, closed=closed)
            if closed:
                wire = cadapi.close_wire(wire)
            return BluemiraWire(wire, label=label)

        return wrapper

    return decorator


@closed_wire_wrapper(drop_closure_point=True)
def make_polygon(
    points: Coordinates,
    label: str = "",  # noqa: ARG001
    *,
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
    points: Coordinates,
    label: str = "",  # noqa: ARG001
    *,
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

    Bezier curve of degree n  by the control points `P_i` is:

    .. math::
        {\\textbf{C}}(t) = \\sum_{i=0}^{n} B_{i,n}(t) \\cdot {\\textbf{P}}_i

    where :math:`B_{i,n}(t)` are the Bernstein polynomials:

    .. math::
        B_{i,n}(t) = \\binom{n}{i} (1 - t)^{n-i} t^i

    t is a parameter ranging from 0 to 1.
    """
    return cadapi.make_bezier(points.T)


def make_bspline(
    poles: npt.ArrayLike,
    mults: npt.ArrayLike,
    knots: npt.ArrayLike,
    *,
    periodic: bool,
    degree: int,
    weights: npt.ArrayLike,
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

    Notes
    -----
    The B-spline curve is defined by the control points `P_i`:

    .. math::
        \\textbf{C}(t) = \\sum_{i=0}^{n} B_{i,p}(t) {\\textbf{P}}_i,

    where the basis functions :math:`B_{i,p}(t)` are computed as:

    .. math::
        B_{i,0}(t) =
        \\begin{cases}
        1 & \\text{if } t_i \\leq t < t_{i+1}, \\\\
        0 & \\text{otherwise}.
        \\end{cases}

    .. math::
        B_{i,p}(t) = \\frac{t - t_i}{t_{i+p} - t_i} B_{i,p-1}(t) +
        \\frac{t_{i+p+1} - t}{t_{i+p+1} - t_{i+1}} B_{i+1,p-1}(t).

    Here, t is a parameter ranging from 0 to 1, representing the position
    along the curve.
    """
    return BluemiraWire(
        cadapi.make_bspline(
            poles,
            mults,
            knots,
            periodic=periodic,
            degree=degree,
            weights=weights,
            check_rational=check_rational,
        ),
        label=label,
    )


def make_bsplinesurface(
    poles: npt.ArrayLike,
    mults_u: npt.ArrayLike,
    mults_v: npt.ArrayLike,
    knot_vector_u: npt.ArrayLike,
    knot_vector_v: npt.ArrayLike,
    degree_u: int,
    degree_v: int,
    weights: npt.ArrayLike,
    *,
    periodic: bool,
    check_rational: bool,
    label: str = "",
) -> BluemiraWire:
    """
    Builds a B-SplineSurface by a lists of Poles, Mults, Knots

    Parameters
    ----------
    poles:
        Array of poles (control points).
    mults_u:
        list of integers for the u-multiplicity
    mults_v:
        list of integers for the u-multiplicity
    knot_vector_u:
        list of u-knots
    knot_vector_v:
        list of v-knots
    degree_u:
        degree of NURBS in u-direction
    degree_v:
        degree of NURBS in v-direction
    weights:
        point weights.
    periodic:
        Whether or not the spline is periodic (same curvature at start and end points)
    check_rational:
        Whether or not to check if the BSpline is rational (not sure)

    Returns
    -------
    A FreeCAD object that contours the bsplinesurface

    Notes
    -----
    This function wraps the FreeCAD function of bsplinesurface buildFromPolesMultsKnots.

    The B-spline surface is defined as:

    .. math::
        {\\textbf{S}}(u, v) = \\sum_{i=0}^{n} \\sum_{j=0}^{m} N_{i,p}(u) \\\\
         M_{j,q}(v) {\\textbf{P}}_{i,j}

    where :math:`N_{i,p}(u)` and :math:`M_{j,q}(v)` are the B-spline basis functions
    in the u and v dimensions, respectively.
    """
    return convert(
        cadapi.make_bsplinesurface(
            poles,
            mults_v,
            mults_u,
            knot_vector_u,
            knot_vector_v,
            degree_u=degree_u,
            degree_v=degree_v,
            weights=weights,
            periodic=periodic,
            check_rational=check_rational,
        ),
        label=label,
    )


def _make_polygon_fallback(
    points,
    label="",
    *,
    closed=False,
    **kwargs,  # noqa: ARG001
) -> BluemiraWire:
    """
    Overloaded function signature for fallback option from interpolate_bspline

    Returns
    -------
    :
        Closed BluemiraWire from points.
    """
    return make_polygon(points, label, closed=closed)


@fallback_to(_make_polygon_fallback, cadapi.FreeCADError)
@log_geometry_on_failure
def interpolate_bspline(
    points: npt.ArrayLike,
    label: str = "",
    *,
    closed: bool = False,
    start_tangent: Iterable | None = None,
    end_tangent: Iterable | None = None,
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
    return BluemiraWire(
        cadapi.interpolate_bspline(
            Coordinates(points).T,
            closed=closed,
            start_tangent=start_tangent,
            end_tangent=end_tangent,
        ),
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
            f"Wire already has {original_n_edges=} < {n_edges_max=}. No point forcing to"
            " a spline."
        )
        return wire

    original_points = wire.discretise(ndiscr=2 * original_n_edges, byedges=False)

    for n_discr in np.array(original_n_edges * np.linspace(0.8, 0.1, 8), dtype=int):
        points = wire.discretise(ndiscr=int(n_discr), byedges=False)
        try:
            wire = BluemiraWire(
                cadapi.interpolate_bspline(points.T, closed=wire.is_closed()),
                label=wire.label,
            )
            break
        except cadapi.FreeCADError:
            continue

    new_points = wire.discretise(ndiscr=2 * original_n_edges, byedges=False)

    delta = np.linalg.norm(original_points.xyz - new_points.xyz, ord=2)
    if delta > l2_tolerance:
        bluemira_warn(
            f"Forcing wire to spline with {n_discr} interpolation points did not achieve"
            f" the desired tolerance: {delta} > {l2_tolerance}"
        )

    return wire


def make_circle(
    radius: float = 1.0,
    center: npt.ArrayLike = (0.0, 0.0, 0.0),
    start_angle: float = 0.0,
    end_angle: float = 360.0,
    axis: npt.ArrayLike = (0.0, 0.0, 1.0),
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
    output = cadapi.make_circle(
        radius, tuple(center), start_angle, end_angle, tuple(axis)
    )
    return BluemiraWire(output, label=label)


def make_circle_arc_3P(  # noqa: N802
    p1: Iterable[float],
    p2: Iterable[float],
    p3: Iterable[float],
    axis: tuple[float, float, float] | None = None,
    label: str = "",
) -> BluemiraWire:
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

    Raises
    ------
    GeometryError
        Raised if the three points are collinear.
    """
    try:
        output = cadapi.make_circle_arc_3P(p1, p2, p3, axis)
    except cadapi.FreeCADError as e:
        raise GeometryError(
            f"Failed to create BluemiraWire circle/arc from 3 points: {e}"
        ) from None
    return BluemiraWire(output, label=label)


def make_ellipse(
    center: npt.ArrayLike = (0.0, 0.0, 0.0),
    major_radius: float = 2.0,
    minor_radius: float = 1.0,
    major_axis: npt.ArrayLike = (1, 0, 0),
    minor_axis: npt.ArrayLike = (0, 1, 0),
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
        tuple(center),
        major_radius,
        minor_radius,
        tuple(major_axis),
        tuple(minor_axis),
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
    wire: BluemiraWire,
    thickness: float,
    /,
    join: str = "intersect",  # noqa: ARG001
    *,
    open_wire: bool = True,
    label: str = "",
    fallback_method: str = "square",
    fallback_force_spline: bool = False,
    byedges: bool = True,
    ndiscr: int = 200,
    **fallback_kwargs,
) -> BluemiraWire:
    """
    Fallback function for discretised offsetting

    Raises
    ------
    GeometryError
        If the wire is not closed. This function cannot handle the offet of an open
        wire.

    Returns
    -------
    :
        The offset wire.
    """
    from bluemira.geometry._pyclipper_offset import offset_clipper  # noqa: PLC0415

    if not wire.is_closed() and not open_wire:
        wire = wire.deepcopy()
        wire.close()

    if not wire.is_closed() and open_wire:
        raise GeometryError(
            "Fallback function _offset_wire_discretised cannot handle open wires."
        )

    coordinates = wire.discretise(byedges=byedges, ndiscr=ndiscr)

    wire = make_polygon(
        offset_clipper(
            coordinates, thickness, method=fallback_method, **fallback_kwargs
        ),
        label=label,
        closed=True,
    )
    if fallback_force_spline:
        return force_wire_to_spline(wire, n_edges_max=ndiscr)
    return wire


@fallback_to(_offset_wire_discretised, cadapi.FreeCADError)
@log_geometry_on_failure
def offset_wire(
    wire: BluemiraWire,
    thickness: float,
    /,
    join: str = "intersect",
    *,
    open_wire: bool = True,
    label: str = "",
    fallback_method: str = "square",  # noqa: ARG001
    byedges: bool = True,  # noqa: ARG001
    ndiscr: int = 400,  # noqa: ARG001
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
        cadapi.offset_wire(wire.shape, thickness, join, open_wire=open_wire),
        label=label,
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

    Raises
    ------
    ValueError
        not enough wires

    Returns
    -------
    A wire forming a convex hull around the input wires in the given
    plane.
    """
    if not wires:
        raise ValueError("Must have at least one wire to draw a hull around.")

    hull_plane = HullPlane(plane)
    if hull_plane is HullPlane.XZ:
        plane_idxs = (0, 2)
    elif hull_plane is HullPlane.XY:
        plane_idxs = (0, 1)
    elif hull_plane is HullPlane.YZ:
        plane_idxs = (1, 2)
    else:
        raise NotImplementedError

    shape_discretisations = []
    for wire in wires:
        discretised_points = wire.discretise(byedges=True, ndiscr=ndiscr)
        shape_discretisations.append(getattr(discretised_points, plane))
    coords = np.hstack(shape_discretisations)

    hull = ConvexHull(coords.T)
    hull_coords = np.zeros((3, len(hull.vertices)))
    hull_coords[plane_idxs, :] = coords[:, hull.vertices]
    return make_polygon(hull_coords, closed=True)


# # =============================================================================
# # Volume function
# # =============================================================================
def polygon_revolve_signed_volume(polygon: npt.ArrayLike) -> float:
    """
    Revolve a polygon along the z axis, and return the volume.

    A polygon placed in the RHS of the z-axis in the xz plane would have positive volume
    if it runs clockwise, and negative volume if it runs counter-clockwise.

    Similarly a polygon placed on the LHS of the z-axis in the xz plane would have
    negative volume if it runs clockwise, positive volume if it runs counter-clockwise.

    Parameters
    ----------
    polygon:
        Stores the x-z coordinate pairs of the four coordinates.

    Raises
    ------
    ValueError
        shape must be (N, 2)

    Returns
    -------
    :
        Volume of revolved polygon

    Notes
    -----
    Consider one edge of the polygon, which has two vertices, $p$ and $c$.
    TODO: insert graphics

    When revolved around the z-axis, this trapezium forms a the frustum of a cone.
    The expression for the volume of this frustrum needs to be modified to avoid
    ZeroDivisionError, thus it is recast into the following (also the simplest) form:
    :math:`V = \\frac{\\pi}{3} (p_z - c_z) (p_x^2 + p_x c_x + c_x^2)`.

    Adding together the signed volume of all edges, the excess negative volume from one
    side would cancel out the excess positive volume from the other, such that
    abs(signed volume)= the volume of the polygon after being revolved around the z-axis.
    """
    polygon = np.asarray(polygon)
    if np.ndim(polygon) != 2 or np.shape(polygon)[1] != 2:  # noqa: PLR2004
        raise ValueError("This function takes in an np.ndarray of shape (N, 2).")
    previous_points, current_points = polygon, np.roll(polygon, -1, axis=0)
    px, pz = previous_points[:, 0], previous_points[:, -1]
    cx, cz = current_points[:, 0], current_points[:, -1]
    volume_3_over_pi = (pz - cz) * (px**2 + px * cx + cx**2)
    return np.pi / 3 * sum(volume_3_over_pi)


def partial_diff_of_volume(
    three_vertices: Sequence[Sequence[float]],
    normalised_direction_vector: Iterable[float],
) -> float:
    """
    Gives the relationship between how the the solid volume varies with the position of
    one of its verticies. More precisely, it gives gives the the partial derivative of
    the volume of the solid revolved out of a polygon when one vertex of that polygon
    is moved in the direction specified by normalised_direction_vector.

    Parameters
    ----------
    three_vertices:
        Contain (x, z) coordinates of the polygon. It extracts only the vertex being
        moved, and the two vertices around it. three_vertices[0] and three_vertices[2]
        are anchor vertices that cannot be adjusted. shape (3, 2)
    normalised_direction_vector:
        Direction that the point is allowed to move in. shape = (2,)

    Returns
    -------
    :
        Partial differential of volume

    Notes
    -----
    Let there be 3 points, :math:`q`, :math:`r`, and :math:`s`, forming two edges of a
    polygon. When r is moved, the polygon's revolved solid volume changes.
    After a hefty amount of derivation, everything cancels out to give the expression
    .. math::

        \\frac{dV}{d r_z} = q_z q_x - r_z q_x + 2 q_z r_x - 2 s_z r_x + r_z s_x - s_z s_x
        \\frac{dV}{d r_x} = (q_x + r_x + s_x) (s_x - q_x)


    The dot product between the direction of motion and the vector :math:`\\frac{dV}{dr}`
    gives the required scalar derivative showing "how much does the volume change when
    r is moved in a certain direction by one unit length".
    """
    (qx, qz), (rx, rz), (sx, sz) = three_vertices
    x_component = qz * qx - rz * qx + 2 * qz * rx - 2 * sz * rx + rz * sx - sz * sx
    z_component = (qx + rx + sx) * (sx - qx)
    xz_derivatives = np.array([x_component, z_component]).T
    return np.pi / 3 * np.dot(normalised_direction_vector, xz_derivatives)


# # =============================================================================
# # Shape operation
# # =============================================================================
def revolve_shape(
    shape: BluemiraGeoT,
    base: npt.ArrayLike = (0.0, 0.0, 0.0),
    direction: npt.ArrayLike = (0.0, 0.0, 1.0),
    degree: float = 180,
    label: str = "",
) -> BluemiraGeoT:
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
    :
        The revolved shape.

    Raises
    ------
    FreeCADError
        Cannot revolve shape
    """
    base = tuple(base)
    direction = tuple(direction)

    if degree > 360:  # noqa: PLR2004
        bluemira_warn("Cannot revolve a shape by more than 360 degrees.")
        degree = 360

    try:
        return convert(cadapi.revolve_shape(shape.shape, base, direction, degree), label)
    except cadapi.FreeCADError:
        if degree == 360:  # noqa: PLR2004
            # We split into two separate revolutions of 180 degree and fuse them
            bluemira_warn(
                "FreeCAD failed to rotate by 360"
                " falling back to 2 fused 180 degree rotations"
            )

            if isinstance(shape, BluemiraWire):
                if shape.is_closed():
                    shape = BluemiraFace(shape)
                flag_shell = True
            else:
                flag_shell = False

            shape_1 = cadapi.revolve_shape(shape.shape, base, direction, degree=180)
            shape_2 = shape_1.copy()
            shape_2 = cadapi.rotate_shape(shape_2, base, direction, degree=-180)
            result = cadapi.boolean_fuse([shape_1, shape_2], remove_splitter=False)

            if flag_shell:
                result = result.Shells[0]

            return convert(result, label)
        raise


def extrude_shape(
    shape: BluemiraGeo, vec: npt.ArrayLike, label: str = ""
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

    return convert(cadapi.extrude_shape(shape.shape, tuple(vec)), label)


class SweepShapeTransition(enum.IntEnum):
    """Sweep shape corner transition options"""

    DEFAULT = 0
    RIGHT_CORNER = 1
    ROUND_CORNER = 2


def sweep_shape(
    profiles: BluemiraWire | Iterable[BluemiraWire],
    path: BluemiraWire,
    *,
    solid: bool = True,
    frenet: bool = True,
    transition: SweepShapeTransition | int = SweepShapeTransition.DEFAULT,
    label: str = "",
) -> BluemiraSolid | BluemiraShell:
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
    transition:
        transition type between sweep sections

    Returns
    -------
    Swept geometry object
    """
    if not isinstance(profiles, Iterable):
        profiles = [profiles]

    profile_shapes = [p.shape for p in profiles]

    result = cadapi.sweep_shape(
        profile_shapes,
        path.shape,
        solid=solid,
        frenet=frenet,
        transition=SweepShapeTransition(transition),
    )

    return convert(result, label=label)


def loft_shape(
    profiles: Iterable[BluemiraWire],
    *,
    solid: bool = True,
    ruled: bool = False,
    closed: bool = False,
    label: str = "",
) -> BluemiraSolid | BluemiraShell:
    """
    Loft between a set of profiles.

    Parameters
    ----------
    profiles:
        Profile(s) to loft between
    solid:
        Whether or not to create a Solid
    ruled:
        Create a ruled shape, see https://en.wikipedia.org/wiki/Ruled_surface
        for explanation.

    Returns
    -------
    Lofted geometry object
    """
    return convert(
        cadapi.loft(
            [p.shape for p in profiles],
            solid=solid,
            ruled=ruled,
            closed=closed,
        ),
        label=label,
    )


def fillet_chamfer_decorator(*, chamfer: bool):
    """
    Decorator for fillet and chamfer operations, checking for validity of wire
    and radius.

    Raises
    ------
    GeometryError
        Number of edges >= 2, radius >= 0 and planar

    Returns
    -------
    :
        Decorator for fillet and chamfer operations.
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


@fillet_chamfer_decorator(chamfer=False)
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


@fillet_chamfer_decorator(chamfer=True)
def chamfer_wire_2D(wire: BluemiraWire, radius: float) -> BluemiraWire:
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
    geo1: npt.ArrayLike | BluemiraGeo, geo2: npt.ArrayLike | BluemiraGeo
) -> tuple[float, list[tuple[float, float, float]]]:
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
    wire: BluemiraWire, vertex: npt.ArrayLike, tolerance: float = EPS * 10
) -> tuple[BluemiraWire | None, BluemiraWire | None]:
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
    wire_1, wire_2 = cadapi.split_wire(wire.shape, tuple(vertex), tolerance=tolerance)
    if wire_1:
        wire_1 = BluemiraWire(wire_1)
    if wire_2:
        wire_2 = BluemiraWire(wire_2)
    return wire_1, wire_2


def slice_shape(
    shape: BluemiraGeo, plane: BluemiraPlane
) -> list[np.ndarray] | list[BluemiraWire] | None:
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


def get_wire_plane_intersect(
    convex_bm_wire: BluemiraWire,
    plane: BluemiraPlane,
    cut_direction: npt.NDArray[float],
) -> npt.NDArray[float]:
    """
    Cut a wire using a plane.

    Parameters
    ----------
    convex_bm_wire:
        The wire that we're interested in cutting.
    plane:
        Plane that is cutting the wire.
    cut_direction:
        np.ndarray with shape==(3,)

    Returns
    -------
    intersection point:
        np.ndarray with shape==(3,)
    """
    intersection_points = slice_shape(convex_bm_wire, plane)
    if len(intersection_points) > 1:
        if len(intersection_points) > 2:  # noqa: PLR2004
            bluemira_warn(
                "convex_bm_wire expected to be a convex hull, but isn't.\n"
                "Proceeding by choosing the final intersection point..."
            )
        final_intersection = np.argmax(np.dot(intersection_points, cut_direction))
        return intersection_points[final_intersection]
    return intersection_points[0]


def circular_pattern(
    shape: BluemiraGeo,
    origin: npt.ArrayLike = (0, 0, 0),
    direction: npt.ArrayLike = (0, 0, 1),
    degree: float = 360,
    n_shapes: int = 10,
) -> list[BluemiraGeo]:
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
        new_shape.rotate(tuple(origin), tuple(direction), i * angle)
        shapes.append(new_shape)
    return shapes


def mirror_shape(
    shape: BluemiraGeo, base: npt.ArrayLike, direction: npt.ArrayLike, label=""
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
    GeometryError
        if the norm of the direction tuple is <= 3*EPS
    """
    if np.linalg.norm(direction) <= 3 * EPS:
        raise GeometryError("Direction vector cannot have a zero norm.")
    return convert(
        cadapi.mirror_shape(shape.shape, tuple(base), tuple(direction)), label=label
    )


def is_convex(points: npt.NDArray):
    """
    Check that the the list of xz points are strictly convex, i.e.
    Not even collinear points are allowed.

    However, repeated points are allowed, as the repeated point would be ignored;
    and points are allowed to be entered in 3D (xyz), but the y component would be
    ignored as well.

    Parameters
    ----------
    points
        A list of points that we want to check the convexity for. Shape = (n, 2/3)

    Returns
    -------
    boolean
    """
    if np.shape(points)[1] == 3:  # noqa: PLR2004
        points = np.array(points)[:, ::2]  # squash 3D points down to 2D
    return len(np.unique(points, axis=0)) == ConvexHull(points).nsimplex


# # =============================================================================
# # Save functions
# # =============================================================================
def save_as_STP(
    shapes: BluemiraGeoT | Iterable[BluemiraGeoT],
    filename: str,
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
    """
    filename = force_file_extension(filename, [".stp", ".step"])

    if not isinstance(shapes, list):
        shapes = list(shapes) if isinstance(shapes, Iterable) else [shapes]

    cadapi.save_as_STP([s.shape for s in shapes], filename, **kwargs)


def save_cad(
    shapes: BluemiraGeoT | Iterable[BluemiraGeoT],
    filename: str | Path,
    cad_format: str | cadapi.CADFileType = "stp",
    names: str | list[str] | None = None,
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
    if not isinstance(shapes, list):
        shapes = list(shapes) if isinstance(shapes, Iterable) else [shapes]
    if names is not None and not isinstance(names, list):
        names = [names]

    cadapi.save_cad(
        [s.shape for s in shapes],
        Path(filename).as_posix(),
        cad_format=cad_format,
        labels=names,
        **kwargs,
    )


def import_cad(
    filename: str | Path,
    cad_format: str | cadapi.CADFileType | None = None,
    unit_scale: str = "m",
    **kwargs,
) -> BluemiraGeo | list[BluemiraGeo]:
    """Import CAD from file

    Returns
    -------
    :
        The imported objects
    """
    objs = [
        convert(s[0], label=s[1])
        for s in cadapi.import_cad(filename, cad_format, unit_scale, **kwargs)
    ]

    if len(objs) == 1:
        return objs[0]
    return objs


# ======================================================================================
# Signed distance functions
# ======================================================================================


@nb.jit(nopython=True, cache=True)
def _nb_dot_2D(v_1, v_2):
    """
    Returns
    -------
    :
        Numba 2-D dot product
    """
    return v_1[0] * v_2[0] + v_1[1] * v_2[1]


@nb.jit(nopython=True, cache=True)
def _nb_clip(val, a_min, a_max):
    """
    Returns
    -------
    :
        Numba 1-D clip.
    """
    return a_min if val < a_min else min(val, a_max)


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
    point = np.asarray(point, dtype=np.float64)
    polygon = np.asarray(polygon, dtype=np.float64)
    n = len(polygon)

    d = _nb_dot_2D(point - polygon[0], point - polygon[0])

    for i in range(n - 1):
        j = i + 1
        e = polygon[j] - polygon[i]
        w = point - polygon[i]
        b = w - e * _nb_clip(_nb_dot_2D(w, e) / _nb_dot_2D(e, e), 0.0, 1.0)
        d = min(_nb_dot_2D(b, b), d)

        cond = np.array([
            point[1] >= polygon[i][1],
            point[1] < polygon[j][1],
            e[0] * w[1] > e[1] * w[0],
        ])
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


def signed_distance(
    origin: BluemiraWire | Coordinates, target: BluemiraWire | Coordinates
) -> float:
    """
    Single-valued signed "distance" function between two wires. Will return negative
    values if origin does not touch or intersect target, 0 if there is one intersection,
    and a positive estimate of the intersection length if there are overlaps.

    Parameters
    ----------
    origin:
        a 0D/1D set of points
    target:
        a 0D/1D set of points

    Returns
    -------
    Closest distance between origin and target

    Notes
    -----
    This is not a pure implementation of a distance function, as for overlapping wires a
    metric of the quantity of overlap is returned (a positive value). This nevertheless
    enables the use of such a function as a constraint in gradient-based optimisers.

    This function has been extended to allow the target wire to be a point
        (:class:`~bluemira.geometry.coordinates.Coordinates`) as well

    The signed "distance" function can be expressed as:

    .. math::

        d (\\textbf{p}_1, \\textbf{p}_2) =
        \\begin{cases}
            - \\| \\textbf{p}_1 - \\textbf{p}_2 \\| & \\text{if no intersection} \\\\
            0 & \\text{if one intersection} \\\\
            \\text{positive estimate of overlap length} & \\text{if overlap exists}
        \\end{cases}
    """
    d, vectors = distance_to(origin, target)
    # Intersections are exactly 0.0
    if d == 0.0:
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

        # TODO @je-cook: Use a boolean difference operation to get the lengths of the
        # overlapping wire segment(s)
        # 3590
        return length
    # There are no intersections, return minimum distance
    return -d


def raise_error_if_overlap(
    origin: BluemiraWire | Coordinates,
    target: BluemiraWire | Coordinates,
    origin_name: str = "",
    target_name: str = "",
):
    """
    Raises
    ------
    GeometryError
        if two wires/points intersects overlaps.
    """
    check_overlaps = signed_distance(origin, target)
    if check_overlaps < -D_TOLERANCE:
        return
    if not origin_name:
        origin_name = "origin " + origin.__class__.__name__
    if not target_name:
        target_name = "target " + target.__class__.__name__
    if -D_TOLERANCE <= check_overlaps <= 0:
        # Sometimes intersecting lines can still appears to separate (negative),
        # but only by just a little. So a small negative number is included in the check.
        raise GeometryError(f"{origin_name} likely intersects {target_name} !")
    if check_overlaps > 0:
        raise GeometryError(f"{origin_name} and {target_name} partially/fully overlaps!")


def connect_shapes(
    shapes: Iterable[BluemiraGeo], tolerance: float = 0.0, label: str = ""
) -> BluemiraGeo:
    """
    Connect (imprint) a list of shapes together.

    This is similar to a boolean fuse operation, but connects the interior walls
    of solid or shells (removeing the excess).

    For wires it will fuse them, keeping the biggest joined piece.

    Parameters
    ----------
    shapes:
        List of shape objects to be saved
    label:
        Label for the resulting shape

    Returns
    -------
    Result of the connect operation.

    Raises
    ------
    GeometryError
        In case the boolean operation fails.
    ValueError
        All shapes (2 or more) must be the same type
    """
    shapes = iterable_to_list(shapes)

    if len(shapes) < 2:  # noqa: PLR2004
        raise ValueError("At least 2 shapes must be given")

    # check that all the shapes are of the same time
    _type = type(shapes[0])
    if not all(isinstance(s, _type) for s in shapes):
        raise ValueError(f"All instances in {shapes} must be of the same type.")

    api_shapes = [s.shape for s in shapes]
    try:
        connected_shape = cadapi.join_connect(api_shapes, dist_tolerance=tolerance)
        return convert(connected_shape, label)

    except Exception as e:  # noqa: BLE001
        raise GeometryError(f"Connect operation failed: {e}") from None


# ======================================================================================
# Boolean operations
# ======================================================================================


def make_compound(
    shapes: Iterable[BluemiraGeo],
    label: str = "",
) -> BluemiraCompound:
    """
    Make a compound of the given shapes.

    The shapes must all be the same topologic type (all solids, shells etc.). No mixing.

    Parameters
    ----------
    shapes:
        List of shape objects to be saved
    label:
        Label for the resulting shape
    set_constituents:
        If True, shapes will be set at as the compound's consitituents.
        It won't be deriived from the cadapi shape.

    Returns
    -------
    Result of the connect operation.

    Raises
    ------
    GeometryError
        In case the boolean operation fails.
    ValueError
        All shapes (2 or more) must be the same type
    """
    shapes = iterable_to_list(shapes)

    if len(shapes) < 2:  # noqa: PLR2004
        raise ValueError("At least 2 shapes must be given")

    # check that all the shapes are of the same time
    _type = type(shapes[0])
    if not all(isinstance(s, _type) for s in shapes):
        raise ValueError(f"All instances in {shapes} must be of the same type.")

    api_shapes = [s.shape for s in shapes]
    try:
        compound_shape = cadapi.make_compound(api_shapes)
        return BluemiraCompound._create(compound_shape, label)
    except Exception as e:  # noqa: BLE001
        raise GeometryError(f"make_compound operation failed: {e}") from None


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
    GeometryError
        In case the boolean operation fails.
    ValueError
        All shapes (2 or more) must be the same type
    """
    shapes = iterable_to_list(shapes)

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
    shape: BluemiraGeoT, tools: BluemiraGeoT | Iterable[BluemiraGeoT]
) -> BluemiraGeoT | list[BluemiraGeoT]:
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
    if not isinstance(tools, Iterable):
        tools = [tools]
    apitools = [t.shape for t in tools]
    cut_shape = cadapi.boolean_cut(apishape, apitools)

    if isinstance(cut_shape, Iterable):
        return [convert(obj, shape.label) for obj in cut_shape]

    return convert(cut_shape, shape.label)


def boolean_fragments(
    shapes: Iterable[BluemiraSolid], tolerance: float = 0.0
) -> tuple[BluemiraCompound, list[list[BluemiraSolid]]]:
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


def point_inside_shape(point: Iterable[float], shape: BluemiraGeoT) -> bool:
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
# # Serialise and Deserialise
# # =============================================================================
def serialise_shape(shape: BluemiraGeoT):
    """
    Serialise a BluemiraGeo object.

    Returns
    -------
    :
        A serialised BluemiraGeo object.
    """
    type_ = type(shape)

    output = []
    if isinstance(shape, BluemiraGeo):
        sdict = {"label": shape.label, "boundary": output}
        for obj in shape.boundary:
            output.append(serialise_shape(obj))
            if shape.mesh_options is not None:
                if shape.mesh_options.lcar is not None:
                    sdict["lcar"] = shape.mesh_options.lcar
                if shape.mesh_options.physical_group is not None:
                    sdict["physical_group"] = shape.mesh_options.physical_group
        return {str(type(shape).__name__): sdict}
    if isinstance(shape, cadapi.apiWire):
        return cadapi.serialise_shape(shape)
    raise NotImplementedError(f"Serialisation non implemented for {type_}")


def deserialise_shape(buffer: dict) -> BluemiraGeoT | None:
    """
    Deserialise a BluemiraGeo object obtained from serialise_shape.

    Parameters
    ----------
    buffer:
        Object serialisation as stored by serialise_shape

    Returns
    -------
    The deserialised BluemiraGeo object.
    """
    supported_types = [BluemiraWire, BluemiraFace, BluemiraShell]

    def _extract_mesh_options(shape_dict: dict) -> meshing.MeshOptions | None:
        mesh_options = None
        if "lcar" in shape_dict:
            mesh_options = meshing.MeshOptions()
            mesh_options.lcar = shape_dict["lcar"]
        if "physical_group" in shape_dict:
            mesh_options = mesh_options or meshing.MeshOptions()
            mesh_options.physical_group = shape_dict["physical_group"]
        return mesh_options

    def _extract_shape(shape_dict: dict, shape_type: type[BluemiraGeoT]) -> BluemiraGeoT:
        label = shape_dict["label"]
        boundary = shape_dict["boundary"]

        temp_list = []
        for item in boundary:
            if issubclass(shape_type, BluemiraWire):
                for k in item:
                    if k == shape_type.__name__:
                        shape = deserialise_shape(item)
                    else:
                        shape = cadapi.deserialise_shape(item)
                    temp_list.append(shape)
            else:
                temp_list.append(deserialise_shape(item))

        mesh_options = _extract_mesh_options(shape_dict)

        shape = shape_type(label=label, boundary=temp_list)
        if mesh_options is not None:
            shape.mesh_options = mesh_options
        return shape

    for type_, v in buffer.items():
        for supported_type in supported_types:
            if type_ == supported_type.__name__:
                return _extract_shape(v, BluemiraWire)

        raise NotImplementedError(f"Deserialisation non implemented for {type_}")
    return None


# # =============================================================================
# # shape utils
# # =============================================================================
def get_shape_by_name(shape: BluemiraGeoT, name: str) -> list[BluemiraGeoT]:
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

    Raises
    ------
    TypeError
        both base and vector must be arrays
    ValueError
        Array shapes must be (2, N)
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
