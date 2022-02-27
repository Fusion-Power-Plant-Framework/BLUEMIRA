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
Wrapper for FreeCAD Part.Face objects
"""

from __future__ import annotations

from typing import List

import mpl_toolkits.mplot3d as a3
import numpy as np

import bluemira.codes._freecadapi as cadapi
from bluemira.display.plotter import BasePlotter, get_plotter, register_plotter
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.error import DisjointedFace, NotClosedWire
from bluemira.geometry.wire import BluemiraWire

__all__ = ["BluemiraFace"]


class BluemiraFace(BluemiraGeo):
    """Bluemira Face class."""

    def __init__(self, boundary, label: str = ""):
        boundary_classes = [BluemiraWire]
        super().__init__(boundary, label, boundary_classes)
        self._create_face()

    @staticmethod
    def _converter(func):
        def wrapper(*args, **kwargs):
            output = func(*args, **kwargs)
            if isinstance(output, cadapi.apiWire):
                output = BluemiraWire(output)
            if isinstance(output, cadapi.apiFace):
                output = BluemiraFace._create(output)
            return output

        return wrapper

    def copy(self):
        """Make a copy of the BluemiraFace"""
        return BluemiraFace(self.boundary, self.label)

    def deepcopy(self):
        """Make a copy of the BluemiraFace"""
        boundary = []
        for o in self.boundary:
            boundary += [o.deepcopy()]
        return BluemiraFace(boundary, self.label)

    def _check_boundary(self, objs):
        """Check if objects in objs are of the correct type for this class"""
        if not hasattr(objs, "__len__"):
            objs = [objs]
        check = False
        for c in self._boundary_classes:
            check = check or (all(isinstance(o, c) for o in objs))
            if check:
                if all(o.is_closed() for o in objs):
                    return objs
                else:
                    raise NotClosedWire("Only closed BluemiraWire are accepted.")
        raise TypeError(
            f"Only {self._boundary_classes} objects can be used for {self.__class__}"
        )

    def _create_face(self, check_reverse=True):
        """Create the primitive face"""
        external: BluemiraWire = self.boundary[0]
        face = cadapi.apiFace(external._create_wire(check_reverse=False))

        if len(self.boundary) > 1:
            fholes = [cadapi.apiFace(h._shape) for h in self.boundary[1:]]
            face = face.cut(fholes)
            if len(face.Faces) == 1:
                face = face.Faces[0]
            else:
                raise DisjointedFace("Any or more than one face has been created.")

        if check_reverse:
            return self._check_reverse(face)
        else:
            return face

    @property
    def _shape(self) -> cadapi.apiFace:
        """Part.Face: shape of the object as a primitive face"""
        return self._create_face()

    @property
    def _wires(self) -> List[cadapi.apiWire]:
        """list(Part.Wire): list of wires of which the shape consists of."""
        wires = []
        for o in self.boundary:
            if isinstance(o, cadapi.apiWire):
                wires += o.Wires
            else:
                wires += o._wires
        return wires

    @classmethod
    def _create(cls, obj: cadapi.apiFace, label="") -> BluemiraFace:
        if isinstance(obj, cadapi.apiFace):
            orientation = obj.Orientation
            bmwires = []
            for w in obj.Wires:
                w_orientation = w.Orientation
                bm_wire = BluemiraWire(w)
                bm_wire._orientation = w_orientation
                bmwires += [bm_wire]
            bmface = cls(bmwires, label=label)
            bmface._orientation = orientation
            return bmface

        raise TypeError(f"Only Part.Face objects can be used to create a {cls} instance")

    def discretize(
        self, ndiscr: int = 100, byedges: bool = False, dl: float = None
    ) -> np.ndarray:
        """
        Make an array of the geometry.

        Parameters
        ----------
        ndiscr: int
            Number of points in the array
        byedges: bool
            Whether or not to discretise by edges

        Returns
        -------
        xyz: np.ndarray
            (M, (3, N)) array of point coordinates where M is the number of boundaries
            and N the number of discretization points.
        """
        points = []
        for w in self._shape.Wires:
            if byedges:
                points.append(cadapi.discretize_by_edges(w, ndiscr=ndiscr, dl=dl))
            else:
                points.append(cadapi.discretize(w, ndiscr=ndiscr, dl=dl))
        return points


class FacePlotter(BasePlotter):
    """
    Plotting class for bluemira faces.
    """

    _CLASS_PLOT_OPTIONS = {"show_points": False, "show_wires": False}

    def _check_obj(self, obj):
        if not isinstance(obj, BluemiraFace):
            raise ValueError(f"{obj} must be a BluemiraFace")
        return True

    def _check_options(self):
        # Check if nothing has to be plotted
        if (
            not self.options.show_points
            and not self.options.show_wires
            and not self.options.show_faces
        ):
            return False

        return True

    def _populate_data(self, face):
        self._data = []
        self._wplotters = []
        # TODO: the for must be done using face._shape.Wires because FreeCAD
        #  re-orient the Wires in the correct way for display. Find another way to do
        #  it (maybe adding this function to the freecadapi.
        for w in face._shape.Wires:
            boundary = BluemiraWire(w)
            wplotter = get_plotter(boundary)(self.options)
            self._wplotters.append(wplotter)
            wplotter._populate_data(boundary)
            self._data.extend(wplotter._data.tolist())
        self._data = np.array(self._data)

        self._data_to_plot = [[], []]
        for w in self._wplotters:
            self._data_to_plot[0] += w._data_to_plot[0].tolist() + [None]
            self._data_to_plot[1] += w._data_to_plot[1].tolist() + [None]

    def _make_plot_2d(self):
        if self.options.show_faces:
            self.ax.fill(*self._data_to_plot, **self.options.face_options)

        for w in self._wplotters:
            w.ax = self.ax
            w._make_plot_2d()
        self._set_aspect_2d()

    def _make_plot_3d(self):
        if self.options.show_faces:
            poly = a3.art3d.Poly3DCollection([self._data], **self.options.face_options)
            self.ax.add_collection3d(poly)

        for w in self._wplotters:
            w.ax = self.ax
            w._make_plot_3d()
        self._set_aspect_3d()


register_plotter(BluemiraFace, FacePlotter)
