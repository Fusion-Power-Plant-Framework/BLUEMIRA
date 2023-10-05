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
Base classes and functionality for the bluemira geometry module.
"""

from __future__ import annotations

import copy
import enum
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

if TYPE_CHECKING:
    import numpy as np

    from bluemira.geometry.coordinates import Coordinates
    from bluemira.geometry.placement import BluemiraPlacement

# import for abstract class
from abc import ABC, abstractmethod

# import freecad api
from bluemira.codes import _freecadapi as cadapi
from bluemira.geometry.bound_box import BoundingBox
from bluemira.mesh import meshing


class GeoMeshable(meshing.Meshable):
    """
    Extended Meshable class for BluemiraGeo objects.
    """

    def remove_mesh_options(self, recursive: bool = False):  # noqa: ARG002
        """
        Remove mesh options for this object.
        """
        super().remove_mesh_options()
        if hasattr(self, "boundary"):
            for obj in self.boundary:
                if isinstance(obj, GeoMeshable):
                    obj.remove_mesh_options(recursive=True)

    def print_mesh_options(self, recursive: bool = True):  # noqa: ARG002
        """
        Print the mesh options for this object.
        """
        # TODO: improve the output of this function
        output = []
        output.append(self.mesh_options)
        if hasattr(self, "boundary"):
            output.extend(
                [
                    obj.print_mesh_options(True)
                    for obj in self.boundary
                    if isinstance(obj, GeoMeshable)
                ]
            )
        return output


class _Orientation(enum.Enum):
    FORWARD = "Forward"
    REVERSED = "Reversed"


class BluemiraGeo(ABC, GeoMeshable):
    """
    Abstract base class for geometry.

    Parameters
    ----------
    boundary:
        shape's boundary
    label:
        identification label for the shape
    boundary_classes:
        list of allowed class types for shape's boundary
    """

    def __init__(
        self,
        boundary: Union[BluemiraGeo, List[BluemiraGeo]],
        label: str = "",
        boundary_classes: Optional[BluemiraGeo] = None,
    ):
        super().__init__()
        self._boundary_classes = boundary_classes
        self.__orientation = _Orientation.FORWARD
        self.label = label
        self._set_boundary(boundary)

    @property
    def _orientation(self):
        return self.__orientation

    @_orientation.setter
    def _orientation(self, value):
        self.__orientation = _Orientation(value)

    def _check_reverse(self, obj):
        if self._orientation != _Orientation(obj.Orientation):
            obj.reverse()
            self._orientation = _Orientation(obj.Orientation)
        return obj

    @staticmethod
    def _converter(func):
        """
        Function used in __getattr__ to modify the added functions.
        """
        return func

    def _check_boundary(self, objs):
        """
        Check if objects objs can be used as boundaries.

        Note: empty BluemiraGeo are allowed in case of objs == None.
        """
        if objs is None:
            return objs

        if not hasattr(objs, "__len__"):
            objs = [objs]

        check = False
        for c in self._boundary_classes:
            # # in case of obj = [], this check returns True instead of False
            # check = check or (all(isinstance(o, c) for o in objs))
            for o in objs:
                check = check or isinstance(o, c)
            if check:
                return objs
        raise TypeError(
            f"Only {self._boundary_classes} objects can be used for {self.__class__}"
        )

    @property
    def boundary(self) -> tuple:
        """
        The shape's boundary.
        """
        return tuple(self._boundary)

    def _set_boundary(self, objs, replace_shape: bool = True):
        self._boundary = self._check_boundary(objs)
        if replace_shape:
            if self._boundary is None:
                self._set_shape(None)
            else:
                self._set_shape(self._create_shape())

    @abstractmethod
    def _create_shape(self):
        """
        Create the shape from the boundary
        """
        # Note: this is the "hidden" connection with primitive shapes

    @property
    def shape(self) -> cadapi.apiShape:
        """
        The primitive shape of the object.
        """
        # Note: this is the "hidden" connection with primitive shapes
        return self._shape

    def _set_shape(self, value: cadapi.apiShape):
        self._shape = value

    @property
    def length(self) -> float:
        """
        The shape's length.
        """
        return cadapi.length(self.shape)

    @property
    def area(self) -> float:
        """
        The shape's area.
        """
        return cadapi.area(self.shape)

    @property
    def volume(self) -> float:
        """
        The shape's volume.
        """
        return cadapi.volume(self.shape)

    @property
    def center_of_mass(self) -> np.ndarray:
        """
        The shape's center of mass.
        """
        return cadapi.center_of_mass(self.shape)

    @property
    def bounding_box(self) -> BoundingBox:
        """
        The bounding box of the shape.

        Notes
        -----
        If your shape is complicated, this has the potential to not be very accurate.
        Consider using :meth:`~get_optimal_bounding_box`.
        """
        x_min, y_min, z_min, x_max, y_max, z_max = cadapi.bounding_box(self.shape)
        return BoundingBox(x_min, x_max, y_min, y_max, z_min, z_max)

    def get_optimal_bounding_box(self, tolerance: float = 1.0) -> BoundingBox:
        """
        Get the optimised bounding box of the shape, via tesselation of the underlying
        geometry.

        Parameters
        ----------
        tolerance:
            Tolerance with which to tesselate the BluemiraGeo before calculating the
            bounding box.
        """
        auto_copy = self.deepcopy()
        auto_copy._tessellate(tolerance)
        return auto_copy.bounding_box

    def is_null(self) -> bool:
        """
        Check if the shape is null.
        """
        return cadapi.is_null(self.shape)

    def is_closed(self) -> bool:
        """
        Check if the shape is closed.
        """
        return cadapi.is_closed(self.shape)

    def is_valid(self) -> bool:
        """
        Check if the shape is valid.
        """
        return cadapi.is_valid(self.shape)

    def is_same(self, obj: BluemiraGeo) -> bool:
        """
        Check if obj has the same shape as self
        """
        return cadapi.is_same(self.shape, obj.shape)

    def search(self, label: str) -> List[BluemiraGeo]:
        """
        Search for a shape with the specified label

        Parameters
        ----------
        label:
            Shape label

        Returns
        -------
        List of shapes that have the specified label
        """
        output = []
        if self.label == label:
            output.append(self)
        for o in self.boundary:
            if isinstance(o, BluemiraGeo):
                output += o.search(label)
        return output

    def scale(self, factor: float) -> None:
        """
        Apply scaling with factor to this object. This function modifies the self
        object.

        Note
        ----
        The operation is made on shape and boundary in order to maintain the consistency.
        Shape is then not reconstructed from boundary (in order to reduce the
        computational time and avoid problems due to api objects orientation).
        """
        for o in self.boundary:
            if isinstance(o, BluemiraGeo):
                o.scale(factor)
            else:
                cadapi.scale_shape(o, factor)
        cadapi.scale_shape(self.shape, factor)

    def _tessellate(self, tolerance: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tessellate the geometry object.

        Parameters
        ----------
        tolerance:
            Tolerance with which to tessellate the geometry

        Returns
        -------
        vertices:
            Array of the vertices (N, 3, dtype=float) from the tesselation operation
        indices:
            Array of the indices (M, 3, dtype=int) from the tesselation operation

        Notes
        -----
        Once tesselated, an object's properties may change. Tesselation cannot be
        reverted to a previous lower value, but can be increased (irreversibly).
        """
        return cadapi.tessellate(self.shape, tolerance)

    def translate(self, vector: Tuple[float, float, float]) -> None:
        """
        Translate this shape with the vector. This function modifies the self
        object.

        Note
        ----
        The operation is made on shape and boundary in order to maintain the consistency.
        Shape is then not reconstructed from boundary (in order to reduce the
        computational time and avoid problems due to api objects orientation).
        """
        for o in self.boundary:
            if isinstance(o, BluemiraGeo):
                o.translate(vector)
            else:
                cadapi.translate_shape(o, vector)
        cadapi.translate_shape(self.shape, vector)

    def rotate(
        self,
        base: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        degree: float = 180,
    ):
        """
        Rotate this shape.

        Parameters
        ----------
        base:
            Origin location of the rotation
        direction:
            The direction vector
        degree:
            rotation angle

        Note
        ----
        The operation is made on shape and boundary in order to maintain the consistency.
        Shape is then not reconstructed from boundary (in order to reduce the
        computational time and avoid problems due to api objects orientation).
        """
        for o in self.boundary:
            if isinstance(o, BluemiraGeo):
                o.rotate(base, direction, degree)
            else:
                cadapi.rotate_shape(o, base, direction, degree)
        cadapi.rotate_shape(self.shape, base, direction, degree)

    def change_placement(self, placement: BluemiraPlacement) -> None:
        """
        Change the placement of self
        Note
        ----
        The operation is made on shape and boundary in order to maintain the consistency.
        Shape is then not reconstructed from boundary (in order to reduce the
        computational time and avoid problems due to api objects orientation).
        """
        for o in self.boundary:
            if isinstance(o, BluemiraGeo):
                o.change_placement(placement)
            else:
                cadapi.change_placement(o, placement._shape)
        cadapi.change_placement(self.shape, placement._shape)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"([{type(self).__name__}] = Label: {self.label}, "
            f"length: {self.length}, "
            f"area: {self.area}, "
            f"volume: {self.volume})"
        )

    def copy(self, label: Optional[str] = None):
        """
        Make a copy of the BluemiraGeo.
        """
        geo_copy = copy.copy(self)
        if label is not None:
            geo_copy.label = label
        else:
            geo_copy.label = self.label
        return geo_copy

    def deepcopy(self, label: Optional[str] = None):
        """
        Make a deepcopy of the BluemiraGeo.
        """
        geo_copy = copy.deepcopy(self)
        if label is not None:
            geo_copy.label = label
        else:
            geo_copy.label = self.label
        return geo_copy

    @property
    @abstractmethod
    def vertexes(self) -> Coordinates:
        """
        The vertexes of the BluemiraGeo.
        """

    @property
    @abstractmethod
    def edges(self) -> tuple:
        """
        The edges of the BluemiraGeo.
        """

    @property
    @abstractmethod
    def wires(self) -> tuple:
        """
        The wires of the BluemiraGeo.
        """

    @property
    @abstractmethod
    def faces(self) -> tuple:
        """
        The faces of the BluemiraGeo.
        """

    @property
    @abstractmethod
    def shells(self) -> tuple:
        """
        The shells of the BluemiraGeo.
        """

    @property
    @abstractmethod
    def solids(self) -> tuple:
        """
        The solids of the BluemiraGeo.
        """
