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
Supporting functions for the bluemira geometry module.
"""

from __future__ import annotations

import enum
import math
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from types import DynamicClassAttribute
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
)
from warnings import warn

import FreeCAD
import BOPTools
import BOPTools.GeneralFuseResult
import BOPTools.JoinAPI
import BOPTools.JoinFeatures
import BOPTools.ShapeMerge
import BOPTools.SplitAPI
import BOPTools.SplitFeatures
import BOPTools.Utils
import DraftGeomUtils
import FreeCADGui
import Part
import numpy as np
from FreeCAD import Base
from PySide2.QtWidgets import QApplication
from matplotlib import colors
from pivy import coin, quarter

from bluemira.base.constants import EPS, raw_uc
from bluemira.base.file import force_file_extension
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.codes._freecadconfig import _freecad_save_config
from bluemira.codes.error import FreeCADError, InvalidCADInputsError
from bluemira.geometry.constants import MINIMUM_LENGTH
from bluemira.utilities.tools import ColourDescriptor

if TYPE_CHECKING:
    from bluemira.display.palettes import ColorPalette

apiVertex = Part.Vertex  # noqa: N816
apiVector = Base.Vector  # noqa: N816
apiEdge = Part.Edge  # noqa: N816
apiWire = Part.Wire  # noqa: N816
apiFace = Part.Face  # noqa: N816
apiShell = Part.Shell  # noqa: N816
apiSolid = Part.Solid  # noqa: N816
apiShape = Part.Shape  # noqa: N816
apiPlacement = Base.Placement  # noqa:  N816
apiPlane = Part.Plane  # noqa: N816
apiCompound = Part.Compound  # noqa: N816

WORKING_PRECISION = 1e-5
MIN_PRECISION = 1e-5
MAX_PRECISION = 1e-5

# ======================================================================================
# Error catching
# ======================================================================================


def catch_caderr(new_error_type):
    """
    Catch CAD errors with given error
    """

    def argswrap(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FreeCADError as fe:
                raise new_error_type(fe.args[0]) from fe

        return wrapper

    return argswrap


# ======================================================================================
# Array, List, Vector, Point manipulation
# ======================================================================================


def arrange_edges(old_wire: apiWire, new_wire: apiWire) -> apiWire:
    """
    A helper to try and fix some topological naming issues.
    Tries to arrange edges as they were in the old wire

    Parameters
    ----------
    old_wire:
        old wire to emulate edges from
    new_wire:
        new wire to change edge arrangement

    Returns
    -------
    Wire with arranged edges
    """
    old_edges = Part.sortEdges(old_wire.Edges)[0]
    new_edges = Part.sortEdges(new_wire.Edges)[0]
    for old_edge in old_edges:
        for new_edge in new_edges:
            if old_edge.Orientation != new_edge.Orientation:
                apiEdge.reverse(new_edge)
    return apiWire(new_edges)


def check_data_type(data_type):
    """
    Decorator to check the data type of the first parameter input (args[0]) of a
    function.

    Raises
    ------
    TypeError: If args[0] objects are not instances of data_type
    """

    def _apply_to_list(func):
        def wrapper(*args, **kwargs):
            output = []
            objs = args[0]
            is_list = isinstance(objs, list)
            if not is_list:
                objs = [objs]
                args = [objs, args[1:]] if len(args) > 1 else [objs]
            if all(isinstance(o, data_type) for o in objs):
                output = func(*args, **kwargs)
                if not is_list:
                    output = output[0]
            else:
                raise TypeError(
                    f"Only {data_type} instances can be converted to {type(output)}"
                )
            return output

        return wrapper

    return _apply_to_list


@check_data_type(Base.Vector)
def vector_to_list(vectors: List[apiVector]) -> List[List[float]]:
    """Converts a FreeCAD Base.Vector or list(Base.Vector) into a list"""
    return [list(v) for v in vectors]


@check_data_type(Part.Point)
def point_to_list(points: List[Part.Point]) -> List[List[float]]:
    """Converts a FreeCAD Part.Point or list(Part.Point) into a list"""
    return [[p.X, p.Y, p.Z] for p in points]


@check_data_type(Part.Vertex)
def vertex_to_list(vertexes: List[apiVertex]) -> List[List[float]]:
    """Converts a FreeCAD Part.Vertex or list(Part.Vertex) into a list"""
    return [[v.X, v.Y, v.Z] for v in vertexes]


@check_data_type(Base.Vector)
def vector_to_numpy(vectors: List[apiVector]) -> np.ndarray:
    """Converts a FreeCAD Base.Vector or list(Base.Vector) into a numpy array"""
    return np.array([np.array(v) for v in vectors])


@check_data_type(Part.Point)
def point_to_numpy(points: List[Part.Point]) -> np.ndarray:
    """Converts a FreeCAD Part.Point or list(Part.Point) into a numpy array"""
    return np.array([np.array([p.X, p.Y, p.Z]) for p in points])


@check_data_type(Part.Vertex)
def vertex_to_numpy(vertexes: List[apiVertex]) -> np.ndarray:
    """Converts a FreeCAD Part.Vertex or list(Part.Vertex) into a numpy array"""
    return np.array([np.array([v.X, v.Y, v.Z]) for v in vertexes])


# ======================================================================================
# Geometry creation
# ======================================================================================


def make_solid(shell: apiShell) -> apiSolid:
    """Make a solid from a shell."""
    return Part.makeSolid(shell)


def make_shell(faces: List[apiFace]) -> apiShell:
    """Make a shell from faces."""
    return Part.makeShell(faces)


def make_compound(shapes: List[apiShape]) -> apiCompound:
    """
    Make an FreeCAD compound object out of many shapes

    Parameters
    ----------
    shapes:
        A set of objects to be compounded

    Returns
    -------
    A compounded set of shapes
    """
    return Part.makeCompound(shapes)


def make_polygon(points: Union[list, np.ndarray]) -> apiWire:
    """
    Make a polygon from a set of points.

    Parameters
    ----------
    points:
        list of points. It can be given as a list of 3D tuples, a 3D numpy array,
        or similar.

    Returns
    -------
    A FreeCAD wire that contains the polygon
    """
    # Points must be converted into FreeCAD Vectors
    pntslist = [Base.Vector(x) for x in points]
    return Part.makePolygon(pntslist)


def make_bezier(points: Union[list, np.ndarray]) -> apiWire:
    """
    Make a bezier curve from a set of points.

    Parameters
    ----------
    points:
        list of points. It can be given as a list of 3D tuples, a 3D numpy array,
        or similar.

    Returns
    -------
    A FreeCAD wire that contains the bezier curve
    """
    # Points must be converted into FreeCAD Vectors
    pntslist = [Base.Vector(x) for x in points]
    bc = Part.BezierCurve()
    bc.setPoles(pntslist)
    return Part.Wire(bc.toShape())


def make_bspline(
    poles: np.ndarray,
    mults: np.ndarray,
    knots: np.ndarray,
    periodic: bool,
    degree: int,
    weights: np.ndarray,
    check_rational: bool,
) -> apiWire:
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
        Whether or not to check if the BSpline is rational (not sure)

    Returns
    -------
    A FreeCAD wire that contains the bspline curve

    Notes
    -----
    This function wraps the FreeCAD function of bsplines buildFromPolesMultsKnots
    """
    poles = [Base.Vector(p) for p in poles]
    bspline = Part.BSplineCurve()
    bspline.buildFromPolesMultsKnots(
        poles, mults, knots, periodic, degree, weights, check_rational
    )
    return apiWire(bspline.toShape())


def interpolate_bspline(
    points: Union[list, np.ndarray],
    closed: bool = False,
    start_tangent: Optional[Iterable] = None,
    end_tangent: Optional[Iterable] = None,
) -> apiWire:
    """
    Make a B-Spline curve by interpolating a set of points.

    Parameters
    ----------
    points:
        list of points. It can be given as a list of 3D tuples, a 3D numpy array,
        or similar.
    closed:
        if True, the first and last points will be connected in order to form a
        closed shape.
    start_tangent:
        Tangency of the BSpline at the first pole. Must be specified with end_tangent
    end_tangent:
        Tangency of the BSpline at the last pole. Must be specified with start_tangent

    Returns
    -------
    A FreeCAD wire that contains the bspline curve
    """
    # In this case, it is not really necessary to convert points in FreeCAD vector. Just
    # left for consistency with other methods.
    pntslist = [Base.Vector(x) for x in points]

    # Recreate checks that are made in freecad/src/MOD/Draft/draftmake/make_bspline.py
    # function make_bspline, line 75

    if len(pntslist) < 2:  # noqa: PLR2004
        _err = "interpolate_bspline: not enough points"
        raise InvalidCADInputsError(_err + "\n")
    if np.allclose(pntslist[0], pntslist[-1], rtol=0, atol=EPS):
        if len(pntslist) > 2:  # noqa: PLR2004
            if not closed:
                bluemira_warn("interpolate_bspline: equal endpoints forced Closed")
            closed = True
            pntslist.pop()
        else:
            # len == 2 and first == last
            _err = "interpolate_bspline: Invalid pointslist (len == 2 and first == last)"
            raise InvalidCADInputsError(_err)

    kwargs = {}
    if start_tangent and end_tangent:
        kwargs["InitialTangent"] = Base.Vector(start_tangent)
        kwargs["FinalTangent"] = Base.Vector(end_tangent)

    if start_tangent and not end_tangent or end_tangent and not start_tangent:
        bluemira_warn(
            "You must set both start and end tangencies or neither when creating a "
            "bspline. Start and end tangencies ignored."
        )

    try:
        bsc = Part.BSplineCurve()
        bsc.interpolate(pntslist, PeriodicFlag=closed, **kwargs)
        wire = apiWire(bsc.toShape())
    except Part.OCCError as error:
        msg = "\n".join(
            [
                "FreeCAD was unable to make a spline:",
                f"{error.args[0]}",
            ]
        )
        raise FreeCADError(msg) from error
    return wire


def make_circle(
    radius: float = 1.0,
    center: Iterable[float] = [0.0, 0.0, 0.0],
    start_angle: float = 0.0,
    end_angle: float = 360.0,
    axis: Iterable[float] = [0.0, 0.0, 1.0],
) -> apiWire:
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
        circle orientation according to the right hand rule. Default [0., 0., 1.].

    Returns
    -------
    FreeCAD wire that contains the arc or circle
    """
    # TODO: check the creation of the arc when start_angle < end_angle
    output = Part.Circle()
    output.Radius = radius
    output.Center = Base.Vector(center)
    output.Axis = Base.Vector(axis)
    if start_angle != end_angle:
        output = Part.ArcOfCircle(
            output, math.radians(start_angle), math.radians(end_angle)
        )
    return Part.Wire(Part.Edge(output))


def make_circle_arc_3P(  # noqa: N802
    p1: Iterable[float], p2: Iterable[float], p3: Iterable[float]
) -> apiWire:
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
    FreeCAD wire that contains the arc of circle
    """
    # TODO: check what happens when the 3 points are in a line
    arc = Part.ArcOfCircle(Base.Vector(p1), Base.Vector(p2), Base.Vector(p3))

    # next steps are made to create an arc of circle that is consistent with that
    # created by 'make_circle'
    output = Part.Circle()
    output.Radius = arc.Radius
    output.Center = arc.Center
    output.Axis = arc.Axis
    arc = Part.ArcOfCircle(
        output, output.parameter(arc.StartPoint), output.parameter(arc.EndPoint)
    )

    return Part.Wire(Part.Edge(arc))


def make_ellipse(
    center: Iterable[float] = [0.0, 0.0, 0.0],
    major_radius: float = 2.0,
    minor_radius: float = 1.0,
    major_axis: Iterable[float] = [1, 0, 0],
    minor_axis: Iterable[float] = [0, 1, 0],
    start_angle: float = 0.0,
    end_angle: float = 360.0,
) -> apiWire:
    """
    Creates an ellipse or arc of ellipse object with given parameters.

    Parameters
    ----------
    center:
        Center of the ellipse
    major_radius:
        the major radius of the ellipse
    minor_radius:
        the minor radius of the ellipse
    major_axis:
        major axis direction
    minor_axis:
        minor axis direction
    start_angle:
        Start angle of the arc [degrees]
    end_angle:
        End angle of the arc [degrees]. If start_angle == end_angle, an ellipse is
        created, otherwise an arc of ellipse is created

    Returns
    -------
    FreeCAD wire that contains the ellipse or arc of ellipse
    """
    # TODO: check the creation of the arc when start_angle < end_angle
    s1 = Base.Vector(major_axis).normalize().multiply(major_radius) + Base.Vector(center)
    s2 = Base.Vector(minor_axis).normalize().multiply(minor_radius) + Base.Vector(center)
    center = Base.Vector(center)
    output = Part.Ellipse(s1, s2, center)

    start_angle = start_angle % 360.0
    end_angle = end_angle % 360.0

    if start_angle != end_angle:
        output = Part.ArcOfEllipse(
            output, math.radians(start_angle), math.radians(end_angle)
        )

    return Part.Wire(Part.Edge(output))


class JoinType(enum.IntEnum):
    """See Part/PartEnums.py, its not importable"""

    Arc = 0
    Tangent = 1
    Intersect = 2


def offset_wire(
    wire: apiWire, thickness: float, join: str = "intersect", open_wire: bool = True
) -> apiWire:
    """
    Make an offset from a wire.

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

    Returns
    -------
    Offset wire
    """
    if thickness == 0.0:  # noqa: PLR2004
        return deepcopy(wire)

    if _wire_is_straight(wire):
        raise InvalidCADInputsError("Cannot offset a straight line.")

    if not _wire_is_planar(wire):
        raise InvalidCADInputsError("Cannot offset a non-planar wire.")

    f_join = JoinType[join.lower().capitalize()]
    if f_join is JoinType.Tangent:
        # NOTE: The "tangent": 1 option misbehaves in FreeCAD
        bluemira_warn(
            f"Join type: {join} is unstable."
            " Please consider using from ['arc', 'intersect']."
        )

    if wire.isClosed() and open_wire:
        open_wire = False

    shape = apiShape(wire)
    try:
        wire = arrange_edges(
            wire, shape.makeOffset2D(thickness, f_join.value, False, open_wire)
        )
    except Base.FreeCADError as error:
        msg = "\n".join(
            [
                "FreeCAD was unable to make an offset of wire:",
                f"{error.args[0]['sErrMsg']}",
            ]
        )
        raise FreeCADError(msg) from None

    fix_wire(wire)
    return wire


def make_face(wire: apiWire) -> apiFace:
    """
    Make a face given a wire boundary.

    Parameters
    ----------
    wire:
        Wire boundary from which to make a face

    Returns
    -------
    Face created from the wire boundary

    Raises
    ------
    FreeCADError
        If the created face is invalid
    """
    face = apiFace(wire)
    if face.isValid():
        return face
    face.fix(WORKING_PRECISION, MIN_PRECISION, MAX_PRECISION)
    if face.isValid():
        return face
    raise FreeCADError("An invalid face has been generated")


# ======================================================================================
# Object properties
# ======================================================================================
def _get_api_attr(obj: apiShape, prop: str):
    try:
        return getattr(obj, prop)
    except AttributeError:
        raise FreeCADError(
            f"FreeCAD object {obj} does not have an attribute: {prop}"
        ) from None


def length(obj: apiShape) -> float:
    """Object's length"""
    return _get_api_attr(obj, "Length")


def area(obj: apiShape) -> float:
    """Object's Area"""
    return _get_api_attr(obj, "Area")


def volume(obj: apiShape) -> float:
    """Object's volume"""
    return _get_api_attr(obj, "Volume")


def center_of_mass(obj: apiShape) -> np.ndarray:
    """Object's center of mass"""
    return vector_to_numpy(_get_api_attr(obj, "CenterOfMass"))


def is_null(obj: apiShape) -> bool:
    """True if obj is null"""
    return _get_api_attr(obj, "isNull")()


def is_closed(obj: apiShape) -> bool:
    """True if obj is closed"""
    return _get_api_attr(obj, "isClosed")()


def is_valid(obj) -> bool:
    """True if obj is valid"""
    return _get_api_attr(obj, "isValid")()


def is_same(obj1: apiShape, obj2: apiShape) -> bool:
    """True if obj1 and obj2 have the same shape."""
    return obj1.isSame(obj2)


def bounding_box(obj: apiShape) -> Tuple[float, float, float, float, float, float]:
    """Object's bounding box"""
    box = _get_api_attr(obj, "BoundBox")
    return box.XMin, box.YMin, box.ZMin, box.XMax, box.YMax, box.ZMax


def tessellate(obj: apiShape, tolerance: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tessellate a geometry object.

    Parameters
    ----------
    obj:
        Shape to tessellate
    tolerance:
        Tolerance with which to perform the operation

    Raises
    ------
    ValueError:
        If the tolerance is <= 0.0

    Returns
    -------
    vertices:
        Array of the vertices (N, 3, dtype=float) from the tesselation operation
    indices:
        Array of the indices (M, 3, dtype=int) from the tesselation operation

    Notes
    -----
    Once tesselated an object's properties may change. Tesselation cannot be reverted
    to a previous lower value, but can be increased (irreversibly).
    """
    if tolerance <= 0.0:  # noqa: PLR2004
        raise ValueError("Cannot have a tolerance that is less than or equal to 0.0")

    vectors, indices = obj.tessellate(tolerance)
    return vector_to_numpy(vectors), np.array(indices)


def start_point(obj: apiShape) -> np.ndarray:
    """The start point of the object"""
    point = obj.Edges[0].firstVertex().Point
    return vector_to_numpy(point)


def end_point(obj: apiShape) -> np.ndarray:
    """The end point of the object"""
    point = obj.Edges[-1].lastVertex().Point
    return vector_to_numpy(point)


def ordered_vertexes(obj: apiShape) -> np.ndarray:
    """Ordered vertexes of the object"""
    vertexes = _get_api_attr(obj, "OrderedVertexes")
    return vertex_to_numpy(vertexes)


def vertexes(obj: apiShape) -> np.ndarray:
    """Wires of the object"""
    vertexes = _get_api_attr(obj, "Vertexes")
    return vertex_to_numpy(vertexes)


def orientation(obj: apiShape) -> bool:
    """True if obj is valid"""
    return _get_api_attr(obj, "Orientation")


def edges(obj: apiShape) -> list[apiWire]:
    """Edges of the object"""
    return _get_api_attr(obj, "Edges")


def ordered_edges(obj: apiShape) -> np.ndarray:
    """Ordered edges of the object"""
    return _get_api_attr(obj, "OrderedEdges")


def wires(obj: apiShape) -> list[apiWire]:
    """Wires of the object"""
    return _get_api_attr(obj, "Wires")


def faces(obj: apiShape) -> list[apiFace]:
    """Faces of the object"""
    return _get_api_attr(obj, "Faces")


def shells(obj: apiShape) -> list[apiShell]:
    """Shells of the object"""
    return _get_api_attr(obj, "Shells")


def solids(obj: apiShape) -> list[apiSolid]:
    """Solids of the object"""
    return _get_api_attr(obj, "Solids")


def normal_at(face: apiFace, alpha_1: float = 0.0, alpha_2: float = 0.0) -> np.ndarray:
    """
    Get the normal vector of the face at a parameterised point in space. For
    planar faces, the normal is the same everywhere.
    """
    return np.array(face.normalAt(alpha_1, alpha_2))


# ======================================================================================
# Wire manipulation
# ======================================================================================
def wire_closure(wire: apiWire) -> apiWire:
    """Create a line segment wire that closes an open wire"""
    closure = None
    if not wire.isClosed():
        vertexes = wire.OrderedVertexes
        points = [v.Point for v in vertexes]
        closure = make_polygon([points[-1], points[0]])
    return closure


def close_wire(wire: apiWire) -> apiWire:
    """
    Closes a wire with a line segment, if not already closed.
    A new wire is returned.
    """
    if not wire.isClosed():
        vertexes = wire.OrderedVertexes
        points = [v.Point for v in vertexes]
        wline = make_polygon([points[-1], points[0]])
        wire = Part.Wire([wire, wline])
    return wire


def discretize(w: apiWire, ndiscr: int = 10, dl: Optional[float] = None) -> np.ndarray:
    """
    Discretize a wire.

    Parameters
    ----------
    w:
        wire to be discretized.
    ndiscr:
        number of points for the whole wire discretization.
    dl:
        target discretization length (default None). If dl is defined,
        ndiscr is not considered.

    Returns
    -------
    Array of points

    Raises
    ------
    ValueError:
        If ndiscr < 2
        If dl <= 0.0
    """
    if dl is None:
        if ndiscr < 2:  # noqa: PLR2004
            raise ValueError("ndiscr must be greater than 2.")
    elif dl <= 0.0:  # noqa: PLR2004
        raise ValueError("dl must be > 0.")
    else:
        # a dl is calculated for the discretisation of the different edges
        # NOTE: must discretise to at least two points.
        ndiscr = max(math.ceil(w.Length / dl + 1), 2)

    # discretization points array
    output = w.discretize(ndiscr)
    output = vector_to_numpy(output)

    if w.isClosed():
        output[-1] = output[0]
    return output


def discretize_by_edges(
    w: apiWire, ndiscr: int = 10, dl: Optional[float] = None
) -> np.ndarray:
    """
    Discretize a wire taking into account the edges of which it consists of.

    Parameters
    ----------
    w:
        Wire to be discretized.
    ndiscr:
        Number of points for the whole wire discretization.
    dl:
        Target discretization length (default None). If dl is defined,
        ndiscr is not considered.

    Returns
    -------
    Array of points

    Notes
    -----
    Final number of points can be slightly different due to edge discretization
    routine.
    """
    # discretization points array
    output = []

    if dl is None:
        # dl is calculated for the discretisation of the different edges
        dl = w.Length / float(ndiscr)
    elif dl <= 0.0:  # noqa: PLR2004
        raise ValueError("dl must be > 0.")

    # edges are discretised taking into account their orientation
    # Note: OrderedEdges already return a list of edges that considers the edge in the
    # correct sequence and orientation. No need for tricks after the discretization.
    for e in w.OrderedEdges:
        pointse = list(discretize(apiWire(e), dl=dl))
        output += pointse[:-1]

    if w.isClosed():
        output += [output[0]]
    else:
        output += [pointse[-1]]

    return np.array(output)


def dist_to_shape(
    shape1: apiShape, shape2: apiShape
) -> Tuple[float, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Find the minimum distance between two shapes

    Parameters
    ----------
    shape1:
        reference shape.
    shape2:
        target shape.

    Returns
    -------
    dist:
        Minimum distance
    vectors:
        List of tuples corresponding to the nearest points (numpy.ndarray)
        between shape1 and shape2. The distance between those points is the minimum
        distance given by dist.
    """
    dist, solution, info = shape1.distToShape(shape2)
    vectors = []
    for v1, v2 in solution:
        vectors.append((vector_to_numpy(v1), vector_to_numpy(v2)))
    return dist, vectors


def wire_value_at(wire: apiWire, distance: float) -> np.ndarray:
    """
    Get a point a given distance along a wire.

    Parameters
    ----------
    wire:
        Wire along which to get a point
    distance:
        Distance

    Returns
    -------
    Wire point value at distance
    """
    if distance == 0.0:  # noqa: PLR2004
        return start_point(wire)
    if distance == wire.Length:
        return end_point(wire)
    if distance < 0.0:  # noqa: PLR2004
        bluemira_warn("Distance must be greater than 0; returning start point.")
        return start_point(wire)
    if distance > wire.Length:
        bluemira_warn("Distance greater than the length of wire; returning end point.")
        return end_point(wire)

    length = 0
    for edge in wire.OrderedEdges:
        edge_length = edge.Length
        new_length = length + edge_length
        if new_length < distance:
            length = new_length
        elif new_length == distance:
            point = edge.valueAt(edge.LastParameter)
            break
        else:
            new_distance = distance - length
            parameter = edge.getParameterByLength(new_distance)
            point = edge.valueAt(parameter)
            break

    return np.array(point)


def wire_parameter_at(
    wire: apiWire, vertex: Iterable[float], tolerance: float = EPS * 10
) -> float:
    """
    Get the parameter value at a vertex along a wire.

    Parameters
    ----------
    wire:
        Wire along which to get the parameter
    vertex:
        Vertex for which to get the parameter
    tolerance:
        Tolerance within which to get the parameter

    Returns
    -------
    Parameter value along the wire at the vertex

    Raises
    ------
    FreeCADError:
        If the vertex is further away to the wire than the specified tolerance
    """
    split_wire_1, _ = split_wire(wire, vertex, tolerance)
    if split_wire_1:
        return split_wire_1.Length / wire.Length
    return 0.0


def split_wire(
    wire: apiWire, vertex: Iterable[float], tolerance: float
) -> Tuple[Union[None, apiWire], Union[None, apiWire]]:
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
    FreeCADError:
        If the vertex is further away to the wire than the specified tolerance
    """

    def warning_msg():
        bluemira_warn(
            "Wire split operation only returning one wire; you are splitting at an end."
        )

    vertex = apiVertex(*vertex)
    distance, points, _ = wire.distToShape(vertex)
    if distance > tolerance:
        raise FreeCADError(
            f"Vertex is not close enough to the wire, with a distance: {distance} >"
            f" {tolerance}"
        )

    edges = wire.OrderedEdges
    idx = _get_closest_edge_idx(wire, vertex)

    edges_1, edges_2 = [], []
    for i, edge in enumerate(edges):
        if i < idx:
            edges_1.append(edge)
        elif i == idx:
            parameter = edge.Curve.parameter(points[0][0])
            half_edge_1, half_edge_2 = _split_edge(edge, parameter)
            if half_edge_1:
                edges_1.append(half_edge_1)
            if half_edge_2:
                edges_2.append(half_edge_2)
        else:
            edges_2.append(edge)

    if edges_1:
        wire_1 = apiWire(edges_1)
    else:
        wire_1 = None
        warning_msg()

    if edges_2:
        wire_2 = apiWire(edges_2)
    else:
        wire_2 = None
        warning_msg()

    return wire_1, wire_2


def _split_edge(edge, parameter):
    p0, p1 = edge.ParameterRange[0], edge.ParameterRange[1]
    if parameter == p0:
        return None, edge
    if parameter == p1:
        return edge, None

    return edge.Curve.toShape(p0, parameter), edge.Curve.toShape(parameter, p1)


def _get_closest_edge_idx(wire, vertex):
    _, points, _ = wire.distToShape(vertex)
    closest_vector = points[0][0]
    closest_vertex = apiVertex(closest_vector)
    distances = [edge.distToShape(closest_vertex)[0] for edge in wire.OrderedEdges]
    return np.argmin(distances)


def slice_shape(
    shape: apiShape, plane_origin: Iterable[float], plane_axis: Iterable[float]
):
    """
    Slice a shape along a given plane

    TODO improve face-solid-shell interface

    Parameters
    ----------
    shape:
        shape to slice
    plane_origin:
        plane origin
    plane_axis:
        normal plane axis

    Notes
    -----
    Degenerate cases such as tangents to solid or faces do not return intersections
    if the shape and plane are acting at the Plane base.
    Further investigation needed.

    """
    if isinstance(shape, apiWire):
        return _slice_wire(shape, plane_axis, plane_origin)
    if not isinstance(shape, (apiFace, apiSolid)):
        bluemira_warn("The output structure of this function may not be as expected")
    shift = np.dot(np.array(plane_origin), np.array(plane_axis))
    return _slice_solid(shape, plane_axis, shift)


def _slice_wire(wire, normal_plane, shift, *, BIG_NUMBER=1e5):
    """
    Get the plane intersection points of any wire (possibly anything, needs testing)
    """
    circ = Part.Circle(
        Base.Vector(*shift), Base.Vector(*normal_plane), BIG_NUMBER
    ).toShape()
    plane = apiFace(apiWire(circ))
    intersect_obj = wire.section(plane)
    return np.array([[v.X, v.Y, v.Z] for v in intersect_obj.Vertexes])


def _slice_solid(obj, normal_plane, shift):
    """
    Get the plane intersection wires of a face or solid
    """
    return obj.slice(Base.Vector(*normal_plane), shift)


# ======================================================================================
# FreeCAD Configuration
# ======================================================================================
def _setup_document(
    parts: Iterable[apiShape], labels: Optional[Iterable[str]] = None
) -> Iterable[Part.Feature]:
    """
    Setup FreeCAD document.

    Converts shapes to FreeCAD Part.Features to enable saving and viewing
    """
    if not hasattr(FreeCADGui, "subgraphFromObject"):
        FreeCADGui.setupWithoutGUI()

    doc = FreeCAD.newDocument()

    if labels is None:
        # Empty string is the default argument for addObject
        labels = [""] * len(parts)

    elif len(labels) != len(parts):
        raise ValueError(
            f"Number of labels ({len(labels)}) != number of objects ({len(parts)})"
        )

    for part, label in zip(parts, labels):
        new_part = part.copy()
        new_part.rotate((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), -90.0)
        obj = doc.addObject("Part::FeaturePython", label)
        obj.Shape = new_part
        doc.recompute()
        yield obj


# ======================================================================================
# Save functions
# ======================================================================================
class CADFileType(enum.Enum):
    """
    FreeCAD standard export filetypes

    Notes
    -----
    Some filetypes my require additional dependencies see:
    https://wiki.freecad.org/Import_Export
    """

    # Commented out currently don't function
    ASCII_STEREO_MESH = ("ast", "Mesh")
    ADDITIVE_MANUFACTURING = ("amf", "Mesh")
    ASC = ("asc", "Points")
    AUTOCAD = ("dwg", "importDWG")
    AUTOCAD_DXF = ("dxf", "importDXF")
    # BDF = ("bdf", "feminout.exportNastranMesh")
    BINMESH = ("bms", "Mesh")
    BREP = ("brep", "Part")
    BREP_2 = ("brp", "Part")
    CSG = ("csg", "exportCSG")
    DAE = ("dae", "importDAE")
    # DAT = ("dat", "Fem")
    FREECAD = ("FCStd", None)
    # FENICS_FEM = ("xdmf", "feminout.importFenicsMesh")
    # FENICS_FEM_XML = ("xml", "feminout.importFenicsMesh")
    GLTRANSMISSION = ("gltf", "ImportGui")
    GLTRANSMISSION_2 = ("glb", "ImportGui")
    IFC_BIM = ("ifc", "exportIFC")
    IFC_BIM_JSON = ("ifcJSON", "exportIFC")
    IGES = ("iges", "ImportGui")
    IGES_2 = ("igs", "ImportGui")
    # INP = ("inp", "Fem")
    INVENTOR_V2_1 = ("iv", "Mesh")
    JSON = ("json", "importJSON")
    # JSON_MESH = ("$json", "feminout.importYamlJsonMesh")
    # MED = ("med", "Fem")
    # MESHJSON = ("meshjson", "feminout.importYamlJsonMesh")
    # MESHPY = ("meshpy", "feminout.importPyMesh")
    # MESHYAML = ("meshyaml", "feminout.importYamlJsonMesh")
    OBJ = ("obj", "Mesh")
    OBJ_WAVE = ("$obj", "importOBJ")
    OFF = ("off", "Mesh")
    OPENSCAD = ("scad", "exportCSG")
    # PCD = ("pcd", "Points")
    # PDF = ("pdf", "FreeCADGui")
    # PLY = ("ply", "Points")
    PLY_STANFORD = ("ply", "Mesh")
    SIMPLE_MODEL = ("smf", "Mesh")
    STEP = ("stp", "ImportGui")
    STEP_2 = ("step", "ImportGui")
    STEP_ZIP = ("stpZ", "stepZ")
    STL = ("stl", "Mesh")
    # SVG = ("svg", "DrawingGui")
    # SVG_FLAT = ("$svg", "importSVG")
    # TETGEN_FEM = ("poly", "feminout.convert2TetGen")
    THREED_MANUFACTURING = ("3mf", "Mesh")
    # UNV = ("unv", "Fem")
    # VRML = ("vrml", "FreeCADGui")
    # VRML_2 = ("wrl", "FreeCADGui")
    # VRML_ZIP = ("wrl.gz", "FreeCADGui")
    # VRML_ZIP_2 = ("wrz", "FreeCADGui")
    # VTK = ("vtk", "Fem")
    # VTU = ("vtu", "Fem")
    # WEBGL = ("html", "importWebGL")
    # WEBGL_X3D = ("xhtml", "FreeCADGui")
    # X3D = ("x3d", "FreeCADGui")
    # X3DZ = ("x3dz", "FreeCADGui")
    # YAML = ("yaml", "feminout.importYamlJsonMesh")
    # Z88_FEM_MESH = ("z88", "Fem")
    # Z88_FEM_MESH_2 = ("i1.txt", "feminout.importZ88Mesh")

    def __new__(cls, *args, **kwds):  # noqa: ARG003
        """Create Enum from first half of tuple"""
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _, module: str = ""):
        self.module = module

    @classmethod
    def unitless_formats(cls) -> Tuple[CADFileType, ...]:
        """CAD formats that don't need to be converted because they are unitless"""
        return (cls.OBJ_WAVE, *[form for form in cls if form.module == "Mesh"])

    @classmethod
    def manual_mesh_formats(cls) -> Tuple[CADFileType, ...]:
        """CAD formats that need to have meshed objects."""
        return (
            cls.GLTRANSMISSION,
            cls.GLTRANSMISSION_2,
            cls.PLY_STANFORD,
            cls.SIMPLE_MODEL,
        )

    @DynamicClassAttribute
    def exporter(self) -> ExporterProtocol:
        """Get exporter module for each filetype"""
        try:
            export_func = __import__(self.module).export
        except AttributeError:
            modlist = self.module.split(".")
            if len(modlist) > 1:
                return getattr(__import__(modlist[0]), modlist[1]).export
            raise FreeCADError(
                f"Unable to save to {self.value} please try through the main FreeCAD GUI"
            ) from None
        except TypeError:
            # Assume CADFileType.FREECAD
            def FreeCADwriter(objs, filename, **kwargs):  # noqa: ARG001
                doc = objs[0].Document
                doc.saveAs(filename)

            return FreeCADwriter
        else:
            if self in self.manual_mesh_formats():
                return meshed_exporter(self, export_func)
            return export_func


class ExporterProtocol(Protocol):
    """Typing for CAD exporter"""

    def __call__(self, objs: List[Part.Feature], filename: str, **kwargs):
        """Export CAD protocol"""


def meshed_exporter(
    cad_format: CADFileType, export_func: Callable[[Part.Feature, str], None]
) -> ExporterProtocol:
    """Meshing and then exporting CAD in certain formats."""

    @wraps(export_func)
    def wrapper(objs: Part.Feature, filename: str, *, tessellate: float = 0.5, **kwargs):
        """
        Tessellation should happen on a copied object
        """
        if cad_format in CADFileType.unitless_formats():
            for no, obj in enumerate(objs):
                objs[no].Shape = obj.Shape.copy()
        for ob in objs:
            ob.Shape.tessellate(tessellate)

        export_func(objs, filename, **kwargs)

    return wrapper


def save_as_STP(
    shapes: List[apiShape], filename: str = "test", unit_scale: str = "metre", **kwargs
):
    """
    Saves a series of Shape objects as a STEP assembly

    Parameters
    ----------
    shapes:
        Iterable of shape objects to be saved
    filename:
        Full path filename of the STP assembly
    unit_scale:
        The scale in which to save the Shape objects

    Notes
    -----
    This uses the legacy method to save to STP files.
    It doesnt require freecad documents but also doesnt allow much customisation.
    Part builds in millimetres therefore we need to scale to metres to be
    consistent with our units.

    """
    filename = force_file_extension(filename, [".stp", ".step"])

    if not isinstance(shapes, list):
        shapes = [shapes]

    if not all(not shape.isNull() for shape in shapes):
        raise FreeCADError("Shape is null.")

    compound = make_compound(shapes)

    if "scale" in kwargs:
        scale = kwargs["scale"]
        warn(
            "Using kwarg 'scale' is no longer supported. Please use 'unit_scale'",
            category=DeprecationWarning,
            stacklevel=2,
        )
    else:
        scale = raw_uc(1, unit_scale, "mm")

    if scale != 1:
        # scale the compound. Since the scale function modifies directly the shape,
        # a copy of the compound is made to avoid modification of the original shapes.
        compound = scale_shape(compound.copy(), scale)

    compound.exportStep(filename)


def _scale_obj(objs, scale: float = 1000):
    """
    Scale objects

    Notes
    -----
    Since the scale function modifies directly the shape,
    a copy of the shape is made to avoid modification of the original shapes.
    The scale of Part is in mm by default therefore we scale by 1000 to convert
    to metres.
    """
    if scale != 1:
        for no, obj in enumerate(objs):
            objs[no].Shape = scale_shape(obj.Shape.copy(), scale)


def save_cad(
    shapes: Iterable[apiShape],
    filename: str,
    cad_format: Union[str, CADFileType] = "stp",
    labels: Optional[Iterable[str]] = None,
    unit_scale: str = "metre",
    **kwargs,
):
    """
    Save CAD in a given file format

    Parameters
    ----------
    shapes:
        CAD shape objects to save
    filename:
        filename (file extension will be forced base on `cad_format`)
    cad_format:
        file cad_format
    labels:
        shape labels
    unit_scale:
        unit to save the objects as.
    kwargs:
        passed to freecad preferences configuration

    Notes
    -----
    Part builds in millimetres therefore we need to scale to metres to be
    consistent with our units
    """
    if kw_formatt := kwargs.pop("formatt", None):
        warn(
            "Using kwarg 'formatt' is no longer supported. Use cad_format instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        cad_format = kw_formatt

    try:
        cad_format = CADFileType(cad_format)
    except ValueError as ve:
        try:
            cad_format = CADFileType[cad_format.upper()]
        except (KeyError, AttributeError):
            raise ve from None

    filename = force_file_extension(filename, f".{cad_format.value.strip('$')}")

    _freecad_save_config(
        **{
            k: kwargs.pop(k)
            for k in kwargs.keys() & {"unit", "no_dp", "author", "stp_file_scheme"}
        }
    )

    objs = list(_setup_document(shapes, labels))

    # Part is always built in mm but some formats are unitless
    if cad_format not in CADFileType.unitless_formats():
        _scale_obj(objs, scale=raw_uc(1, unit_scale, "mm"))

    # Some exporters need FreeCADGui to be setup before their import,
    # this is achieved in _setup_document
    try:
        cad_format.exporter(objs, filename, **kwargs)
    except ImportError as imp_err:
        raise FreeCADError(
            f"Unable to save to {cad_format.value} please try through the main"
            " FreeCAD GUI"
        ) from imp_err

    if not Path(filename).exists():
        mesg = f"{filename} not created, filetype not written by FreeCAD."
        if cad_format is CADFileType.IFC_BIM:
            mesg += " FreeCAD requires `ifcopenshell` to save in this format."
        elif cad_format is CADFileType.DAE:
            mesg += " FreeCAD requires `pycollada` to save in this format."
        elif cad_format is CADFileType.IFC_BIM_JSON:
            mesg += (
                " FreeCAD requires `ifcopenshell` and"
                " IFCJSON module to save in this format."
            )
        elif cad_format is CADFileType.AUTOCAD:
            mesg += " FreeCAD requires `LibreDWG` to save in this format."

        raise FreeCADError(
            f"{mesg} Not able to save object with format:"
            f" '{cad_format.value.strip('$')}'"
        )


# # =============================================================================
# # Shape manipulations
# # =============================================================================
def scale_shape(shape: apiShape, factor: float) -> apiShape:
    """
    Apply scaling with factor to the shape

    Parameters
    ----------
    shape:
        The shape to be scaled
    factor:
        The scaling factor

    Returns
    -------
    The scaled shape
    """
    return shape.scale(factor)


def translate_shape(shape: apiShape, vector: Tuple[float, float, float]) -> apiShape:
    """
    Apply scaling with factor to the shape

    Parameters
    ----------
    shape:
        The shape to be scaled
    vector:
        The translation vector

    Returns
    -------
    The translated shape
    """
    return shape.translate(Base.Vector(vector))


def rotate_shape(
    shape: apiShape,
    base: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    degree: float = 180,
) -> apiShape:
    """
    Apply the rotation (base, dir, degree) to this shape

    Parameters
    ----------
    shape:
        The shape to be rotated
    base:
        Origin location of the rotation
    direction:
        The direction vector
    degree:
        rotation angle

    Returns
    -------
    The rotated shape
    """
    return shape.rotate(base, direction, degree)


def mirror_shape(
    shape: apiShape,
    base: Tuple[float, float, float],
    direction: Tuple[float, float, float],
) -> apiShape:
    """
    Mirror a shape about a plane.

    Parameters
    ----------
    shape:
        Shape to mirror
    base:
        Mirror plane base point
    direction:
        Mirror plane direction

    Returns
    -------
    The mirrored shape
    """
    base = Base.Vector(base)
    direction = Base.Vector(direction)
    mirrored_shape = shape.mirror(base, direction)
    if isinstance(shape, apiSolid):
        return mirrored_shape.Solids[0]
    if isinstance(shape, apiCompound):
        return mirrored_shape.Compounds[0]
    if isinstance(shape, apiFace):
        return mirrored_shape.Faces[0]
    if isinstance(shape, apiWire):
        return mirrored_shape.Wires[0]
    if isinstance(shape, apiShell):
        return mirrored_shape.Shells[0]
    return None


def revolve_shape(
    shape: apiShape,
    base: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    degree: float = 180.0,
) -> apiShape:
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
    base = Base.Vector(base)
    direction = Base.Vector(direction)
    return shape.revolve(base, direction, degree)


def extrude_shape(shape: apiShape, vec: Tuple[float, float, float]) -> apiShape:
    """
    Apply the extrusion along vec to this shape

    Parameters
    ----------
    shape:
        The shape to be extruded
    vec:
        The vector along which to extrude

    Returns
    -------
    The extruded shape.
    """
    vec = Base.Vector(vec)
    return shape.extrude(vec)


def _split_wire(wire):
    """
    Split a wire into two parts.
    """
    edges = wire.OrderedEdges
    if len(edges) == 1:
        # Only one edge in the wire, which we need to split
        edge = edges[0]
        p_start, p_end = edge.ParameterRange
        p_mid = 0.5 * (p_end - p_start)
        edges_1 = edge.Curve.toShape(p_start, p_mid)
        edges_2 = edge.Curve.toShape(p_mid, p_end)

    else:
        # We can just sub-divide the wire by its edges
        n_split = int(len(edges) / 2)
        edges_1, edges_2 = edges[:n_split], edges[n_split:]

    return apiWire(edges_1), apiWire(edges_2)


def sweep_shape(
    profiles: Iterable[apiWire], path: apiWire, solid: bool = True, frenet: bool = True
) -> Union[apiShell, apiSolid]:
    """
    Sweep a a set of profiles along a path.

    Parameters
    ----------
    profiles:
        Set of profiles to sweep
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

    closures = [p.isClosed() for p in profiles]
    all_closed = sum(closures) == len(closures)
    none_closed = sum(closures) == 0

    if not all_closed and not none_closed:
        raise FreeCADError("You cannot mix open and closed profiles when sweeping.")

    if none_closed and solid:
        bluemira_warn(
            "You cannot sweep open profiles and expect a Solid result. Disabling this."
        )
        solid = False

    if not _wire_edges_tangent(path):
        raise FreeCADError(
            "Sweep path contains edges that are not consecutively tangent. This will"
            " produce unexpected results."
        )

    result = path.makePipeShell(profiles, True, frenet)

    solid_result = apiSolid(result)
    if solid:
        return solid_result
    return solid_result.Shells[0]


def fillet_wire_2D(wire: apiWire, radius: float, chamfer: bool = False) -> apiWire:
    """
    Fillet or chamfer a two-dimensional wire, returning a new wire

    Parameters
    ----------
    wire:
        Wire to be filleted or chamfered
    radius:
        Radius of the fillet or chamfer operation
    chamfer: bool (default=False)
        Whether to chamfer or not

    Returns
    -------
    Resulting filleted or chamfered wire
    """
    # Temporarily suppress pesky print statement:
    # DraftGeomUtils.fillet: Warning: edges have same direction. Did nothing
    old_stdout = sys.stdout
    with open(os.devnull, "w") as fh:
        try:
            sys.stdout = fh
            result = DraftGeomUtils.filletWire(wire, radius, chamfer=chamfer)
        finally:
            sys.stdout = old_stdout

    return result


# ======================================================================================
# Boolean operations
# ======================================================================================
def boolean_fuse(shapes: Iterable[apiShape], remove_splitter: bool = True) -> apiShape:
    """
    Fuse two or more shapes together. Internal splitter are removed.

    Parameters
    ----------
    shapes:
        List of FreeCAD shape objects to be fused together. All the objects in the
        list must be of the same type.
    remove_splitter:
        if True, shape is refined removing extra edges.
        See(https://wiki.freecadweb.org/Part_RefineShape)


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

    _type = type(shapes[0])
    _check_shapes_same_type(shapes)

    if _is_wire_or_face(_type):
        _check_shapes_coplanar(shapes)
        if not _shapes_are_coaxis(shapes):
            bluemira_warn(
                "Boolean fuse on shapes that do not have the same planar axis."
                " Reversing."
            )
            _make_shapes_coaxis(shapes)

    try:
        if _type == apiWire:
            merged_shape = BOPTools.SplitAPI.booleanFragments(shapes, "Split")
            if len(merged_shape.Wires) > len(shapes):
                raise FreeCADError(  # noqa: TRY301
                    "Fuse wire creation failed. Possible "
                    "overlap or internal intersection of "
                    f"input shapes {shapes}."
                )
            merged_shape = merged_shape.fuse(merged_shape.Wires)
            return Part.Wire(merged_shape.Wires)

        if _type == apiFace:
            merged_shape = shapes[0].fuse(shapes[1:])
            if remove_splitter:
                merged_shape = merged_shape.removeSplitter()
            if len(merged_shape.Faces) > 1:
                raise FreeCADError(  # noqa: TRY301
                    f"Boolean fuse operation on {shapes} gives more than one face."
                )
            return merged_shape.Faces[0]

        if _type == apiSolid:
            merged_shape = shapes[0].fuse(shapes[1:])
            if remove_splitter:
                merged_shape = merged_shape.removeSplitter()
            if len(merged_shape.Solids) > 1:
                raise FreeCADError(  # noqa: TRY301
                    f"Boolean fuse operation on {shapes} gives more than one solid."
                )
            return merged_shape.Solids[0]

        raise ValueError(  # noqa: TRY301
            f"Fuse function still not implemented for {_type} instances."
        )
    except Exception as e:  # noqa: BLE001
        raise FreeCADError(str(e)) from e


def boolean_cut(
    shape: apiShape, tools: List[apiShape], split: bool = True
) -> List[apiShape]:
    """
    Difference of shape and a given (list of) topo shape cut(tools)

    Parameters
    ----------
    shape:
        the reference object
    tools:
        List of FreeCAD shape objects to be used as tools.
    split:
        If True, shape is split into pieces based on intersections with tools.

    Returns
    -------
    Result of the boolean operation.

    Raises
    ------
    error: GeometryError
        In case the boolean operation fails.
    """
    _type = type(shape)

    if not isinstance(tools, list):
        tools = [tools]

    if _is_wire_or_face(_type):
        _check_shapes_coplanar([shape, *tools])

    cut_shape = shape.cut(tools)
    if split:
        cut_shape = BOPTools.SplitAPI.slice(cut_shape, tools, mode="Split")

    if _type == apiWire:
        output = cut_shape.Wires
    elif _type == apiFace:
        output = cut_shape.Faces
    elif _type == apiShell:
        output = cut_shape.Shells
    elif _type == apiSolid:
        output = cut_shape.Solids
    else:
        raise ValueError(f"Cut function not implemented for {_type} objects.")
    return output


def boolean_fragments(
    shapes: List[apiSolid], tolerance: float = 0.0
) -> Tuple[apiCompound, List[apiSolid]]:
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
    """
    try:
        compound, fragment_map = shapes[0].generalFuse(shapes[1:], tolerance)
    except Exception as e:  # noqa: BLE001
        raise FreeCADError(f"Boolean fragments operation failed: {e!s}") from None
    return compound, fragment_map


def point_inside_shape(point: Iterable[float], shape: apiShape) -> bool:
    """
    Whether or not a point is inside a shape.

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
    vector = apiVector(*point)
    return shape.isInside(vector, EPS, True)


# ======================================================================================
# Geometry checking tools
# ======================================================================================


def _edges_tangent(edge_1, edge_2):
    """
    Check if two adjacent edges are tangent to one another.
    """
    angle = edge_1.tangentAt(edge_1.LastParameter).getAngle(
        edge_2.tangentAt(edge_2.FirstParameter)
    )
    return np.isclose(
        angle,
        0.0,
        rtol=1e-4,
        atol=1e-4,
    )


def _wire_edges_tangent(wire):
    """
    Check that all consecutive edges in a wire are tangent
    """
    if len(wire.Edges) <= 1:
        return True

    edges_tangent = []
    for i in range(len(wire.OrderedEdges) - 1):
        edge_1 = wire.OrderedEdges[i]
        edge_2 = wire.OrderedEdges[i + 1]
        edges_tangent.append(_edges_tangent(edge_1, edge_2))

    if wire.isClosed():
        # Check last and first edge tangency
        edges_tangent.append(_edges_tangent(wire.OrderedEdges[-1], wire.OrderedEdges[0]))

    return all(edges_tangent)


def _wire_is_planar(wire):
    """
    Check if a wire is planar.
    """
    try:
        face = Part.Face(wire)
    except Part.OCCError:
        return False
    return isinstance(face.Surface, Part.Plane)


def _wire_is_straight(wire):
    """
    Check if a wire is a straight line.
    """
    if len(wire.Edges) == 1:
        edge = wire.Edges[0]
        if len(edge.Vertexes) == 2:  # noqa: PLR2004
            straight = dist_to_shape(edge.Vertexes[0], edge.Vertexes[1])[0]
            if np.isclose(straight, wire.Length, rtol=EPS, atol=1e-8):
                return True
    return False


def _is_wire_or_face(shape_type):
    return shape_type in (apiWire, apiFace)


def _check_shapes_same_type(shapes):
    """
    Check that all the shapes are of the same type.
    """
    _type = type(shapes[0])
    if not all(isinstance(s, _type) for s in shapes):
        raise ValueError(f"All instances in {shapes} must be of the same type.")


def _check_shapes_coplanar(shapes):
    if not _shapes_are_coplanar(shapes):
        raise ValueError(
            "Shapes are not co-planar; this operation does not support non-co-planar"
            " wires or faces."
        )


def _shapes_are_coplanar(shapes):
    """
    Check if a list of shapes are all coplanar. First shape is taken as the reference.
    """
    return all(shapes[0].isCoplanar(other) for other in shapes[1:])


def _shapes_are_coaxis(shapes):
    """
    Check if a list of shapes are all co-axis. First shape is taken as the reference.
    """
    axis = shapes[0].findPlane().Axis
    for shape in shapes[1:]:
        other_axis = shape.findPlane().Axis
        if axis != other_axis:
            return False
    return True


def _make_shapes_coaxis(shapes):
    """
    Make a list of shapes co-axis by reversing. First shape is taken as the reference.
    """
    axis = shapes[0].findPlane().Axis
    for shape in shapes[1:]:
        other_axis = shape.findPlane().Axis
        if axis != other_axis:
            shape.reverse()


# ======================================================================================
# Geometry healing
# ======================================================================================


def fix_wire(wire: apiWire, precision: float = EPS, min_length: float = MINIMUM_LENGTH):
    """
    Fix a wire by removing any small edges and joining the remaining edges.

    Parameters
    ----------
    wire:
        Wire to fix
    precision:
        General precision with which to work
    min_length:
        Minimum edge length
    """
    wire.fix(precision, min_length, min_length)


# ======================================================================================
# Placement manipulations
# ======================================================================================
def make_placement(
    base: Iterable[float], axis: Iterable[float], angle: float
) -> apiPlacement:
    """
    Make a FreeCAD Placement

    Parameters
    ----------
    base: Iterable
        a vector representing the Placement local origin
    axis: Iterable
        axis of rotation
    angle:
        rotation angle in degree
    """
    base = Base.Vector(base)
    axis = Base.Vector(axis)

    return Base.Placement(base, axis, angle)


def make_placement_from_matrix(matrix: np.ndarray) -> apiPlacement:
    """
    Make a FreeCAD Placement from a 4 x 4 matrix.

    Parameters
    ----------
    matrix:
        4 x 4 matrix from which to make the placement

    Notes
    -----
    Matrix should be of the form:
        [cos_11, cos_12, cos_13, dx]
        [cos_21, cos_22, cos_23, dy]
        [cos_31, cos_32, cos_33, dz]
        [     0,      0,      0,  1]
    """
    if matrix.shape != (4, 4):
        raise FreeCADError(f"Matrix must be of shape (4, 4), not: {matrix.shape}")

    for i in range(3):
        row = matrix[i, :3]
        matrix[i, :3] = row / np.linalg.norm(row)
    matrix[-1, :] = [0, 0, 0, 1]

    matrix = Base.Matrix(*matrix.flat)
    return Base.Placement(matrix)


def move_placement(placement: apiPlacement, vector: Iterable[float]):
    """
    Moves the FreeCAD Placement along the given vector

    Parameters
    ----------
    placement:
        the FreeCAD placement to be modified
    vector:
        direction along which the placement is moved
    """
    placement.move(Base.Vector(vector))


def make_placement_from_vectors(
    base: Iterable[float] = [0, 0, 0],
    vx: Iterable[float] = [1, 0, 0],
    vy: Iterable[float] = [0, 1, 0],
    vz: Iterable[float] = [0, 0, 1],
    order: str = "ZXY",
) -> apiPlacement:
    """Create a placement from three directional vectors"""
    rotation = Base.Rotation(vx, vy, vz, order)
    return Base.Placement(base, rotation)


def change_placement(geo: apiShape, placement: apiPlacement):
    """
    Change the placement of a FreeCAD object

    Parameters
    ----------
    geo:
        the object to be modified
    placement:
        the FreeCAD placement to be modified
    """
    new_placement = geo.Placement.multiply(placement)
    new_base = placement.multVec(geo.Placement.Base)
    new_placement.Base = new_base
    geo.Placement = new_placement


# ======================================================================================
# Plane creation and manipulations
# ======================================================================================
def make_plane(
    base: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> apiPlane:
    """
    Creates a FreeCAD plane with a given location and normal

    Parameters
    ----------
    base:
        a reference point in the plane
    axis:
        normal vector to the plane

    Returns
    -------
    Plane from base and axis
    """
    base = Base.Vector(base)
    axis = Base.Vector(axis)

    return Part.Plane(base, axis)


def make_plane_from_3_points(
    point1: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    point2: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    point3: Tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> apiPlane:
    """
    Creates a FreeCAD plane defined by three non-linear points

    Parameters
    ----------
    point1:
        a reference point in the plane
    point2:
        a reference point in the plane
    point3:
        a reference point in the plane

    Returns
    -------
    Plane from three points
    """
    point1 = Base.Vector(point1)
    point2 = Base.Vector(point2)
    point3 = Base.Vector(point3)

    return Part.Plane(point1, point2, point3)


def face_from_plane(plane: apiPlane, width: float, height: float) -> apiFace:
    """
    Creates a FreeCAD face from a Plane with specified height and width.

    Note
    ----
    Face is centered on the Plane Position. With respect to the global coordinate
    system, the face placement is given by a simple rotation of the z axis.

    Parameters
    ----------
    plane:
        the reference plane
    width:
        output face width
    height:
        output face height

    Returns
    -------
    Face from plane
    """
    # as suggested in https://forum.freecadweb.org/viewtopic.php?t=46418
    corners = [
        Base.Vector(-width / 2, -height / 2, 0),
        Base.Vector(width / 2, -height / 2, 0),
        Base.Vector(width / 2, height / 2, 0),
        Base.Vector(-width / 2, height / 2, 0),
    ]
    # create the closed border
    border = Part.makePolygon([*corners, corners[0]])
    wall = Part.Face(border)

    wall.Placement = placement_from_plane(plane)

    return wall


def plane_from_shape(shape: apiShape) -> apiPlane:
    """Return a plane if the shape is planar"""
    return shape.findPlane()


def placement_from_plane(plane: apiPlane) -> apiPlacement:
    """
    Return a placement from a plane with the origin on the plane base and the z-axis
    directed as the plane normal.
    """
    axis = plane.Axis
    pos = plane.Position

    vx = plane.value(1, 0) - pos
    vy = plane.value(0, 1) - pos

    return make_placement_from_vectors(pos, vx, vy, axis, "ZXY")


# ======================================================================================
# Geometry visualisation
# ======================================================================================


def _colourise(node: coin.SoNode, options: Dict):
    if isinstance(node, coin.SoMaterial):
        rgb = colors.to_rgb(options["colour"])
        transparency = options["transparency"]
        node.ambientColor.setValue(coin.SbColor(*rgb))
        node.diffuseColor.setValue(coin.SbColor(*rgb))
        node.transparency.setValue(transparency)
    for child in node.getChildren() or []:
        _colourise(child, options)


def collect_verts_faces(
    solid: apiShape, tesselation: float = 0.1
) -> Tuple[Optional[np.ndarray], ...]:
    """
    Collects verticies and faces of parts and tessellates them
    for the CAD viewer

    Parameters
    ----------
    solid:
        FreeCAD Part
    tesselation:
        amount of tesselation for the mesh

    Returns
    -------
    vertices:
        Vertices
    faces:
        Faces
    """
    verts = []
    faces = []
    voffset = 0

    # collect
    for face in solid.Faces:
        # tesselation is likely to be the most expensive part of this
        v, f = face.tessellate(tesselation)

        if v != []:
            verts.append(np.array(v))
            if f != []:
                faces.append(np.array(f) + voffset)
            voffset += len(v)

    if len(solid.Faces) > 0:
        return np.vstack(verts), np.vstack(faces)
    return None, None


def collect_wires(solid: apiShape, **kwds) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collects verticies and edges of parts and discretizes them
    for the CAD viewer

    Parameters
    ----------
    solid:
        FreeCAD Part

    Returns
    -------
    vertices:
        Vertices
    edges:
        Edges
    """
    verts = []
    edges = []
    voffset = 0
    for wire in solid.Wires:
        v = wire.discretize(**kwds)
        verts.append(np.array(v))
        edges.append(np.arange(voffset, voffset + len(v) - 1))
        voffset += len(v)
    edges = np.concatenate(edges)[:, None]
    return np.vstack(verts), np.hstack([edges, edges + 1])


@dataclass
class DefaultDisplayOptions:
    """Freecad default display options"""

    colour: ColourDescriptor = ColourDescriptor()
    transparency: float = 0.0

    @property
    def color(self) -> str:
        """See colour"""
        return self.colour

    @color.setter
    def color(self, value: Union[str, Tuple[float, float, float], ColorPalette]):
        """See colour"""
        self.colour = value


def show_cad(
    parts: Union[apiShape, List[apiShape]],
    options: Union[Dict, List[Optional[Dict]]],
    labels: List[str],
    **kwargs,  # noqa: ARG001
):
    """
    The implementation of the display API for FreeCAD parts.

    Parameters
    ----------
    parts:
        The parts to display.
    options:
        The options to use to display the parts.
    labels:
        labels to use for each part object
    """
    if options is None:
        options = [None]

    if None in options:
        options = [DefaultDisplayOptions() if o is None else o for o in options]

    if len(options) != len(parts):
        raise FreeCADError(
            "If options for display are provided then there must be as many options as "
            "there are parts to display."
        )

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    root = coin.SoSeparator()

    for obj, option in zip(_setup_document(parts, labels), options):
        subgraph = FreeCADGui.subgraphFromObject(obj)
        _colourise(subgraph, option)
        root.addChild(subgraph)

    viewer = quarter.QuarterWidget()
    viewer.setBackgroundColor(coin.SbColor(1, 1, 1))
    viewer.setTransparencyType(coin.SoGLRenderAction.SCREEN_DOOR)
    viewer.setSceneGraph(root)

    viewer.setWindowTitle("Bluemira Display")
    viewer.show()
    app.exec_()


# # =============================================================================
# # Serialize and Deserialize
# # =============================================================================
def extract_attribute(func):
    """
    Decorator for serialize_shape. Convert the function output attributes string
    list to the corresponding object attributes.
    The first argument of func is the reference object.
    If an output is callable, the output result is returned.
    """

    def wrapper(*args, **kwargs):
        type_, attrs = func(*args, **kwargs)
        output = {}
        for k, v in attrs.items():
            if k == "type":
                output[k] = type(args[0])
            else:
                output[v] = getattr(args[0], k)
                if callable(output[v]):
                    output[v] = output[v]()
        return {type_: output}

    return wrapper


def serialize_shape(shape):
    """
    Serialize a FreeCAD topological data object.
    """
    type_ = type(shape)

    if type_ == Part.Wire:
        return {"Wire": [serialize_shape(edge) for edge in shape.OrderedEdges]}

    if type_ == Part.Edge:
        return serialize_shape(_convert_edge_to_curve(shape))

    if type_ in [Part.LineSegment, Part.Line]:
        return {
            "LineSegment": {
                "StartPoint": list(shape.StartPoint),
                "EndPoint": list(shape.EndPoint),
            },
        }

    if type_ == Part.BezierCurve:
        return {
            "BezierCurve": {
                "Poles": vector_to_list(shape.getPoles()),
                "FirstParameter": shape.FirstParameter,
                "LastParameter": shape.LastParameter,
            }
        }

    if type_ == Part.BSplineCurve:
        return {
            "BSplineCurve": {
                "Poles": vector_to_list(shape.getPoles()),
                "Mults": shape.getMultiplicities(),
                "Knots": shape.getKnots(),
                "isPeriodic": shape.isPeriodic(),
                "Degree": shape.Degree,
                "Weights": shape.getWeights(),
                "checkRational": shape.isRational(),
                "FirstParameter": shape.FirstParameter,
                "LastParameter": shape.LastParameter,
            }
        }

    if type_ == Part.ArcOfCircle:
        return {
            "ArcOfCircle": {
                "Radius": shape.Radius,
                "Center": list(shape.Center),
                "Axis": list(shape.Axis),
                "StartAngle": math.degrees(shape.FirstParameter),
                "EndAngle": math.degrees(shape.LastParameter),
                "StartPoint": list(shape.StartPoint),
                "EndPoint": list(shape.EndPoint),
            }
        }

    if type_ == Part.ArcOfEllipse:
        return {
            "ArcOfEllipse": {
                "Center": list(shape.Center),
                "MajorRadius": shape.MajorRadius,
                "MinorRadius": shape.MinorRadius,
                "MajorAxis": list(shape.XAxis),
                "MinorAxis": list(shape.YAxis),
                "StartAngle": math.degrees(shape.FirstParameter),
                "EndAngle": math.degrees(shape.LastParameter),
                "Focus1": list(shape.Ellipse.Focus1),
                "StartPoint": list(shape.StartPoint),
                "EndPoint": list(shape.EndPoint),
            }
        }

    raise NotImplementedError(f"Serialization non implemented for {type_}")


def deserialize_shape(buffer):
    """
    Deserialize a FreeCAD topological data object obtained from serialize_shape.

    Parameters
    ----------
    buffer:
        Object serialization as stored by serialize_shape

    Returns
    -------
        The deserialized FreeCAD object
    """
    for type_, v in buffer.items():
        if type_ == "Wire":
            return Part.Wire([deserialize_shape(edge) for edge in v])
        if type_ == "LineSegment":
            return make_polygon([v["StartPoint"], v["EndPoint"]])
        if type_ == "BezierCurve":
            return make_bezier(v["Poles"])
        if type_ == "BSplineCurve":
            return make_bspline(
                v["Poles"],
                v["Mults"],
                v["Knots"],
                v["isPeriodic"],
                v["Degree"],
                v["Weights"],
                v["checkRational"],
            )
        if type_ == "ArcOfCircle":
            return make_circle(
                v["Radius"], v["Center"], v["StartAngle"], v["EndAngle"], v["Axis"]
            )
        if type_ == "ArcOfEllipse":
            return make_ellipse(
                v["Center"],
                v["MajorRadius"],
                v["MinorRadius"],
                v["MajorAxis"],
                v["MinorAxis"],
                v["StartAngle"],
                v["EndAngle"],
            )
        raise NotImplementedError(f"Deserialization non implemented for {type_}")
    return None


def _convert_edge_to_curve(edge: apiEdge) -> Part.Curve:
    """
    Convert a Freecad Edge to the respective curve.

    Parameters
    ----------
    edge:
        FreeCAD Edge

    Returns
    -------
    FreeCAD Part curve object
    """
    curve = edge.Curve
    first = edge.FirstParameter
    last = edge.LastParameter
    if edge.Orientation == "Reversed":
        first, last = last, first
    output = None

    if isinstance(curve, Part.Line):
        output = Part.LineSegment(curve.value(first), curve.value(last))
    elif isinstance(curve, Part.Ellipse):
        output = Part.ArcOfEllipse(curve, first, last)
        if edge.Orientation == "Reversed":
            output.Axis = -output.Axis
            p0 = curve.value(first)
            p1 = curve.value(last)
            output = Part.ArcOfEllipse(
                output.Ellipse,
                output.Ellipse.parameter(p0),
                output.Ellipse.parameter(p1),
            )
    elif isinstance(curve, Part.Circle):
        output = Part.ArcOfCircle(curve, first, last)
        if edge.Orientation == "Reversed":
            output.Axis = -output.Axis
            p0 = curve.value(first)
            p1 = curve.value(last)
            output = Part.ArcOfCircle(
                output.Circle,
                output.Circle.parameter(p0),
                output.Circle.parameter(p1),
            )
    elif isinstance(curve, Part.BezierCurve):
        output = Part.BezierCurve()
        poles = curve.getPoles()
        if edge.Orientation == "Reversed":
            poles.reverse()
        output.setPoles(poles)
        output.segment(first, last)
    elif isinstance(curve, Part.BSplineCurve):
        output = curve
        # p = curve.discretize(100)
        # if edge.Orientation == "Reversed":
        #     p.reverse()
        # output = Part.BSplineCurve()
        # output.interpolate(p)
    elif isinstance(curve, Part.OffsetCurve):
        c = curve.toNurbs()
        if isinstance(c, Part.BSplineCurve) and edge.Orientation == "Reversed":
            c.reverse()
        output = _convert_edge_to_curve(Part.Edge(c))
    else:
        bluemira_warn(f"Conversion of {type(curve)} is still not supported!")

    return output
