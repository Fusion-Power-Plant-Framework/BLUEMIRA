# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Wrapper for FreeCAD Part.Wire objects
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import bluemira.codes._freecadapi as cadapi
from bluemira.base.look_and_feel import LOGGER, bluemira_warn
from bluemira.codes.error import FreeCADError
from bluemira.geometry.base import BluemiraGeo, _Orientation
from bluemira.geometry.constants import EPS_FREECAD
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.error import (
    GeometryError,
    MixedOrientationWireError,
    NotClosedWireError,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["BluemiraWire"]


class BluemiraWire(BluemiraGeo):
    """
    Bluemira Wire class.

    Parameters
    ----------
    boundary:
        List of wires from which to make the BluemiraWire
    label:
        Label to assign to the wire
    """

    def __init__(self, boundary: list[cadapi.apiWire | BluemiraWire], label: str = ""):
        boundary_classes = [type(self), cadapi.apiWire]
        super().__init__(boundary, label, boundary_classes)
        self._check_orientations()

    def _check_orientations(self):
        orientations = []
        for boundary in self.boundary:
            if isinstance(boundary, cadapi.apiWire):
                orient = boundary.Orientation
            elif isinstance(boundary, self.__class__):
                orient = boundary.shape.Orientation
            orientations.append(orient)

        if orientations.count(orientations[0]) != len(orientations):
            raise MixedOrientationWireError(
                "Cannot make a BluemiraWire from wires of mixed orientations:"
                f" {orientations}"
            )
        self._orientation = orientations[0]
        if self._orientation != _Orientation(self.shape.Orientation):
            self.shape.reverse()

    @staticmethod
    def _converter(func):
        def wrapper(*args, **kwargs):
            output = func(*args, **kwargs)
            if isinstance(output, cadapi.apiWire):
                output = BluemiraWire(output)
            return output

        return wrapper

    def _create_shape(self) -> cadapi.apiWire:
        """
        Returns
        -------
        apiWire:
            shape of the object as a single wire
        """
        return self._create_wire()

    def _create_wire(self, *, check_reverse: bool = True):
        wire = cadapi.apiWire(self._get_wires())

        if not cadapi.is_valid(wire):
            cadapi.fix_shape(wire)

        if check_reverse:
            return self._check_reverse(wire)
        return wire

    def _get_wires(self) -> list[cadapi.apiWire]:
        """
        Returns
        -------
        :
            List of wires of which the shape consists of.
        """
        wires = []
        for o in self.boundary:
            if isinstance(o, cadapi.apiWire):
                for w in o.Wires:
                    wire = cadapi.apiWire(w.OrderedEdges)
                    if self._orientation != _Orientation(wire.Orientation):
                        wire.reverse()
                    wires += [wire]
            else:
                wires += o._get_wires()
        return wires

    def __add__(self, other: BluemiraWire) -> BluemiraWire:
        """Add two wires

        Raises
        ------
        TypeError
            Must be BluemiraWire

        Returns
        -------
        :
            Wire resulting from combining two wires.

        """
        output = None
        if isinstance(other, BluemiraWire):
            output = BluemiraWire([self, other])
        else:
            raise TypeError(f"{type(other)} is not an instance of BluemiraWire.")
        return output

    def close(self, label: str = "") -> None:
        """
        Close the shape with a line segment between shape's end and start point.
        This function modifies the object boundary.

        Raises
        ------
        NotClosedWireError
            Wire has not been closed
        """
        if not self.is_closed():
            closure = BluemiraWire(cadapi.wire_closure(self.shape), label)
            self._boundary.append(closure)
            self._set_boundary(self.boundary)

        # check that the new boundary is closed
        if not self.is_closed():
            raise NotClosedWireError("The open boundary has not been closed.")

    def discretise(
        self, ndiscr: int = 100, *, byedges: bool = False, dl: float | None = None
    ) -> Coordinates:
        """
        Discretise the wire in ndiscr equidistant points or with a reference dl
        segment step.

        Parameters
        ----------
        ndiscr:
            Number of points to discretise to
        byedges:
            Whether or not to discretise by edges. If True, each edge is
            discretised separately using an approximated distance
            (wire.Length/ndiscr) or the specified dl. If True, it is
            possible that ndiscr is larger than specified.
        dl:
            Discretise by length, overriding ndiscr

        Returns
        -------
        Coordinates of the discretised points.
        """
        if byedges:
            points = cadapi.discretise_by_edges(self.shape, ndiscr=ndiscr, dl=dl)
        else:
            points = cadapi.discretise(self.shape, ndiscr=ndiscr, dl=dl)
        return Coordinates(points.T)

    def value_at(
        self, alpha: float | None = None, distance: float | None = None
    ) -> np.ndarray:
        """
        Get a point along the wire at a given parameterised length or length.

        Parameters
        ----------
        alpha:
            Parameterised distance along the wire length, in the range [0 .. 1]
        distance:
            Physical distance along the wire length

        Returns
        -------
        Point coordinates (w.r.t. BluemiraWire's BluemiraPlacement)

        Raises
        ------
        GeometryError
            Alpha or distance must be specified
        """
        if alpha is None and distance is None:
            raise GeometryError("Must specify one of alpha or distance.")
        if alpha is not None and distance is not None:
            raise GeometryError("Must specify either alpha or distance, not both.")

        if distance is None:
            if alpha < 0.0:
                bluemira_warn(
                    f"alpha must be between 0 and 1, not: {alpha}, setting to 0.0"
                )
                alpha = 0
            elif alpha > 1.0:
                bluemira_warn(
                    f"alpha must be between 0 and 1, not: {alpha}, setting to 1.0"
                )
                alpha = 1.0
            distance = alpha * self.length

        return cadapi.wire_value_at(self.shape, distance)

    def parameter_at(
        self, vertex: Iterable[float], tolerance: float = EPS_FREECAD
    ) -> float:
        """
        Get the parameter value at a vertex along a wire.

        Parameters
        ----------
        vertex:
            Vertex for which to get the parameter
        tolerance:
            Tolerance within which to get the parameter

        Returns
        -------
        Parameter value along the wire at the vertex

        Raises
        ------
        GeometryError
            If the vertex is further away to the wire than the specified tolerance
        """
        try:
            return cadapi.wire_parameter_at(
                self.shape, vertex=vertex, tolerance=tolerance
            )
        except FreeCADError as e:
            raise GeometryError(e.args[0]) from None

    def parameter_nearest_to_point(self, point: np.ndarray | Coordinates) -> float:
        """
        Parameters
        ----------
        point:
            point of interest, that we want to get as close to as possible.

        Returns
        -------
        :
            The parameter of the point on the wire itself, that's nearest to the point of
            interest.

        Raises
        ------
        GeometryError
            Reference point must be a single 3D point.
        """
        # Convert point into something that cadapi can use.
        if isinstance(point, Coordinates):
            if np.shape(point) != (3, 1):
                raise GeometryError("Can only measure distance to a single 3D point!")
            point = point.points[0]
        if len(point) != 3:  # noqa: PLR2004
            raise GeometryError("Points must be 3-dimensional.")
        cad_point = cadapi.apiVertex(*point)

        # dist_to_shape gives the (distance, (start point, end point)).
        nearest_to_point = cadapi.dist_to_shape(self, cad_point)[1][0]
        return self.parameter_at(nearest_to_point)

    def intersect_wire(self, other_wire: BluemiraWire) -> Coordinates:
        """Find the point at which other_wire intersects with self."""
        ...

    def start_point(self) -> Coordinates:
        """
        Returns
        -------
        :
            Get the coordinates of the start of the wire.
        """
        return Coordinates(cadapi.start_point(self.shape))

    def end_point(self) -> Coordinates:
        """
        Returns
        -------
        :
            Get the coordinates of the end of the wire.
        """
        return Coordinates(cadapi.end_point(self.shape))

    @property
    def vertexes(self) -> Coordinates:
        """
        The ordered vertexes of the wire.
        """
        vertexes = cadapi.ordered_vertexes(self.shape)
        if len(vertexes) == 3:  # noqa: PLR2004
            LOGGER.disabled = True
            coords = Coordinates(vertexes.T)
            LOGGER.disabled = False
            return coords

        return Coordinates(vertexes)

    @property
    def edges(self) -> tuple[BluemiraWire]:
        """
        The ordered edges of the wire.
        """
        return tuple(
            BluemiraWire(cadapi.apiWire(o)) for o in cadapi.ordered_edges(self.shape)
        )

    @property
    def wires(self) -> tuple[BluemiraWire]:
        """
        The wires of the wire. By definition a tuple of itself.
        """
        return (self,)

    @property
    def faces(self) -> tuple:
        """
        The faces of the wire. By definition an empty tuple.
        """
        return ()

    @property
    def shells(self) -> tuple:
        """
        The shells of the wire. By definition an empty tuple.
        """
        return ()

    @property
    def solids(self) -> tuple:
        """
        The solids of the wire. By definition an empty tuple.
        """
        return ()
