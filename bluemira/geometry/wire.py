# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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
Wrapper for FreeCAD Part.Wire objects
"""

from __future__ import annotations

from typing import List

import numpy

from bluemira.codes._freecadapi import (
    apiWire,
    change_plane,
    discretize,
    discretize_by_edges,
    rotate_shape,
    scale_shape,
    translate_shape,
    wire_closure,
)
from bluemira.display.plotter import BasePlotter, PointsPlotter, register_plotter
from bluemira.geometry.base import BluemiraGeo, _Orientation
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.error import MixedOrientationWireError, NotClosedWire

__all__ = ["BluemiraWire"]


class BluemiraWire(BluemiraGeo):
    """Bluemira Wire class."""

    def __init__(self, boundary, label: str = ""):
        boundary_classes = [self.__class__, apiWire]
        super().__init__(boundary, label, boundary_classes)
        self._check_orientations()

        # connection variable with BLUEPRINT Loop
        self._bp_loop = None

    def _check_orientations(self):
        orientations = []
        for boundary in self.boundary:
            if isinstance(boundary, apiWire):
                orient = boundary.Orientation
            elif isinstance(boundary, self.__class__):
                orient = boundary._shape.Orientation
            orientations.append(orient)

        if orientations.count(orientations[0]) != len(orientations):
            raise MixedOrientationWireError(
                f"Cannot make a BluemiraWire from wires of mixed orientations: {orientations}"
            )
        self._orientation = orientations[0]

    @staticmethod
    def _converter(func):
        def wrapper(*args, **kwargs):
            output = func(*args, **kwargs)
            if isinstance(output, apiWire):
                output = BluemiraWire(output)
            return output

        return wrapper

    @property
    def _shape(self) -> apiWire:
        """apiWire: shape of the object as a single wire"""
        return self._create_wire()

    def _create_wire(self, check_reverse=True):
        wire = apiWire(self._wires)
        if check_reverse:
            return self._check_reverse(wire)
        else:
            return wire

    @property
    def _wires(self) -> List[apiWire]:
        """list(apiWire): list of wires of which the shape consists of."""
        wires = []
        for o in self.boundary:
            if isinstance(o, apiWire):
                for w in o.Wires:
                    wire = apiWire(w.OrderedEdges)
                    if self._orientation != _Orientation(wire.Orientation):
                        edges = []
                        for edge in wire.OrderedEdges:
                            edge.reverse()
                            edges.append(edge)
                        wire = apiWire(edges)
                    wires += [wire]
            else:
                wires += o._wires
        return wires

    def get_single_wire(self) -> BluemiraWire:
        """Get a single wire representing the object"""
        return BluemiraWire(self._shape)

    def __add__(self, other):
        """Add two wires"""
        output = None
        if isinstance(other, BluemiraWire):
            output = BluemiraWire([self, other])
        else:
            raise TypeError(f"{type(other)} is not an instance of BluemiraWire.")
        return output

    def close(self) -> None:
        """
        Close the shape with a line segment between shape's end and start point.
        This function modifies the object boundary.
        """
        if not self.is_closed():
            closure = wire_closure(self._shape)
            if isinstance(self.boundary[0], apiWire):
                self.boundary.append(closure)
            else:
                self.boundary.append(BluemiraWire(closure))

        # check that the new boundary is closed
        if not self.is_closed():
            raise NotClosedWire("The open boundary has not been closed.")

    def discretize(
        self, ndiscr: int = 100, byedges: bool = False, dl: float = None
    ) -> numpy.ndarray:
        """
        Discretize the wire in ndiscr equidistant points or with a reference dl
        segment step.
        If byedges is True, each edges is discretized separately using an approximated
        distance (wire.Length/ndiscr) or the specified dl.

        Returns
        -------
        points: Coordinates
            a numpy array with the x,y,z coordinates of the discretized points.
        """
        if byedges:
            points = discretize_by_edges(self._shape, ndiscr=ndiscr, dl=dl)
        else:
            points = discretize(self._shape, ndiscr=ndiscr, dl=dl)
        return Coordinates(points)

    def scale(self, factor) -> None:
        """
        Apply scaling with factor to this object. This function modifies the self
        object.
        """
        for o in self.boundary:
            if isinstance(o, apiWire):
                scale_shape(o, factor)
            else:
                o.scale(factor)

    def translate(self, vector) -> None:
        """
        Translate this shape with the vector. This function modifies the self
        object.
        """
        for o in self.boundary:
            if isinstance(o, apiWire):
                translate_shape(o, vector)
            else:
                o.translate(vector)

    def rotate(
        self,
        base: tuple = (0.0, 0.0, 0.0),
        direction: tuple = (0.0, 0.0, 1.0),
        degree: float = 180,
    ):
        """
        Rotate this shape.

        Parameters
        ----------
        base: tuple (x,y,z)
            Origin location of the rotation
        direction: tuple (x,y,z)
            The direction vector
        degree: float
            rotation angle
        """
        for o in self.boundary:
            if isinstance(o, apiWire):
                rotate_shape(o, base, direction, degree)
            else:
                o.rotate(base, direction, degree)

    def change_plane(self, plane):
        """Apply a plane transformation to the wire"""
        for o in self.boundary:
            if isinstance(o, apiWire):
                change_plane(o, plane._shape)
            else:
                o.change_plane(plane)


class WirePlotter(BasePlotter):
    """
    Plotting class for bluemira wires.
    """

    _CLASS_PLOT_OPTIONS = {"show_points": False}

    def _check_obj(self, obj):
        if not isinstance(obj, BluemiraWire):
            raise ValueError(f"{obj} must be a BluemiraWire")
        return True

    def _check_options(self):
        # Check if nothing has to be plotted
        if not self.options.show_points and not self.options.show_wires:
            return False

        return True

    def _populate_data(self, wire: BluemiraWire):
        self._pplotter = PointsPlotter(self.options)
        new_wire = wire.deepcopy()
        # # change of plane integrated in PointsPlotter2D. Not necessary here.
        # new_wire.change_plane(self.options._options['plane'])
        pointsw = new_wire.discretize(
            ndiscr=self.options._options["ndiscr"],
            byedges=self.options._options["byedges"],
        ).T
        self._pplotter._populate_data(pointsw)
        self._data = pointsw
        self._data_to_plot = self._pplotter._data_to_plot

    def _make_plot_2d(self):
        if self.options.show_wires:
            self.ax.plot(*self._data_to_plot, **self.options.wire_options)

        if self.options.show_points:
            self._pplotter.ax = self.ax
            self._pplotter._make_plot_2d()
        self._set_aspect_2d()

    def _make_plot_3d(self):
        if self.options.show_wires:
            self.ax.plot(*self._data.T, **self.options.wire_options)

        if self.options.show_points:
            self._pplotter.ax = self.ax
            self._pplotter._make_plot_3d()
        self._set_aspect_3d()


register_plotter(BluemiraWire, WirePlotter)
