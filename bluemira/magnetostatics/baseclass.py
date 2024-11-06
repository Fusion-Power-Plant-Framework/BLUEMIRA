# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Base classes for use in magnetostatics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.pyplot import Axes

import numpy as np
import numpy.typing as npt

from bluemira.geometry.bound_box import BoundingBox
from bluemira.geometry.coordinates import rotation_matrix
from bluemira.magnetostatics.error import MagnetostaticsError
from bluemira.utilities.plot_tools import Plot3D

__all__ = ["CrossSectionCurrentSource", "CurrentSource", "SourceGroup"]


class CurrentSource(ABC):
    """
    Abstract base class for a current source.
    """

    current: float

    def set_current(self, current: float):
        """
        Set the current inside each of the circuits.

        Parameters
        ----------
        current:
            The current of each circuit [A]
        """
        self.current = current

    @abstractmethod
    def field(
        self,
        x: float | npt.NDArray[np.float64],
        y: float | npt.NDArray[np.float64],
        z: float | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Calculate the magnetic field at a set of coordinates.

        Parameters
        ----------
        x:
            The x coordinate(s) of the points at which to calculate the field
        y:
            The y coordinate(s) of the points at which to calculate the field
        z:
            The z coordinate(s) of the points at which to calculate the field

        Returns
        -------
        :
            The magnetic field vector {Bx, By, Bz} in [T]
        """
        ...

    @abstractmethod
    def plot(self, ax: Axes | None, **kwargs):
        """
        Plot the CurrentSource.

        Parameters
        ----------
        ax:
            The matplotlib axes to plot on
        """

    @abstractmethod
    def rotate(self, angle: float, axis: np.ndarray | str):
        """
        Rotate the CurrentSource about an axis.

        Parameters
        ----------
        angle:
            The rotation degree [rad]
        axis:
            The axis of rotation
        """

    def copy(self):
        """
        Get a deepcopy of the CurrentSource.
        """
        return deepcopy(self)


class CrossSectionCurrentSource(CurrentSource):
    """
    Abstract class for a current source with a cross-section
    """

    _origin: np.array
    _dcm: np.array
    _points: np.array
    _rho: float
    _area: float

    def set_current(self, current: float):
        """
        Set the current inside the source, adjusting current density.

        Parameters
        ----------
        current:
            The current of the source [A]
        """
        super().set_current(current)
        self._rho = current / self._area

    def rotate(self, angle: float, axis: np.ndarray | str):
        """
        Rotate the CurrentSource about an axis.

        Parameters
        ----------
        angle:
            The rotation degree [degree]
        axis:
            The axis of rotation
        """
        r = rotation_matrix(np.deg2rad(angle), axis).T
        self._origin @= r
        self._points = np.array([p @ r for p in self._points], dtype=object)
        self._dcm @= r

    def _local_to_global(
        self, points: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Convert local x', y', z' point coordinates to global x, y, z point coordinates.
        """
        return np.array([self._origin + self._dcm.T @ p for p in points])

    def _global_to_local(
        self, points: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Convert global x, y, z point coordinates to local x', y', z' point coordinates.
        """
        return np.array([(self._dcm @ (p - self._origin)) for p in points])

    def plot(self, ax: Axes | None = None, *, show_coord_sys: bool = False):
        """
        Plot the CurrentSource.

        Parameters
        ----------
        ax: Union[None, Axes]
            The matplotlib axes to plot on
        show_coord_sys: bool
            Whether or not to plot the coordinate systems
        """
        if ax is None:
            ax = Plot3D()
            # If no ax provided, we assume that we want to plot only this source,
            # and thus set aspect ratio equality on this term only
            edge_points = np.concatenate(self._points)

            # Invisible bounding box to set equal aspect ratio plot
            xbox, ybox, zbox = BoundingBox.from_xyz(*edge_points.T).get_box_arrays()
            ax.plot(1.1 * xbox, 1.1 * ybox, 1.1 * zbox, "s", alpha=0)

        for points in self._points:
            ax.plot(*points.T, color="b", linewidth=1)

        # Plot local coordinate system
        if show_coord_sys:
            ax.scatter(*self._origin, color="k")
            ax.quiver(*self._origin, *self._dcm[0], length=1, color="r")
            ax.quiver(*self._origin, *self._dcm[1], length=1, color="r")
            ax.quiver(*self._origin, *self._dcm[2], length=1, color="r")


class PolyhedralCrossSectionCurrentSource(CrossSectionCurrentSource):
    """
    Abstract base class for a current source with a polyhedral cross-section.
    """

    _face_points: npt.NDArray[np.float64]
    _face_normals: npt.NDArray[np.float64]
    _mid_points: npt.NDArray[np.float64]

    def rotate(self, angle: float, axis: np.ndarray | str):
        """
        Rotate the CurrentSource about an axis.

        Parameters
        ----------
        angle:
            The rotation degree [degree]
        axis:
            The axis of rotation
        """
        super().rotate(angle, axis)
        r = rotation_matrix(np.deg2rad(angle), axis).T
        self._face_normals = np.array([n @ r for n in self._face_normals])
        self._face_points = np.array([p @ r for p in self._face_points])
        self._mid_points = np.array([p @ r for p in self._mid_points])


class PrismEndCapMixin:
    def _check_angle_values(self, alpha: float, beta: float):
        """
        Check that end-cap angles are acceptable.

        Raises
        ------
        MagnetostaticsError
            Endcap angles must have the same sign
            alpha and beta must be within [0, 180°)
        """
        sign_alpha = np.sign(alpha)
        sign_beta = np.sign(beta)
        one_zero = np.any(np.array([sign_alpha, sign_beta]) == 0)
        if not one_zero and sign_alpha != sign_beta:
            raise MagnetostaticsError(
                f"{self.__class__.__name__} instantiation error: end-cap angles "
                f"must have the same sign {alpha=:.3f}, {beta=:.3f}."
            )
        if not (0 <= abs(alpha) < 0.5 * np.pi):
            raise MagnetostaticsError(
                f"{self.__class__.__name__} instantiation error: {alpha=:.3f} is outside"
                " bounds of [0, 180°)."
            )
        if not (0 <= abs(beta) < 0.5 * np.pi):
            raise MagnetostaticsError(
                f"{self.__class__.__name__} instantiation error: {beta=:.3f} is outside "
                "bounds of [0, 180°)."
            )

    def _check_raise_self_intersection(
        self, length: float, breadth: float, alpha: float, beta: float
    ):
        """
        Check for bad combinations of source length and end-cap angles.

        Raises
        ------
        MagnetostaticsError
            Self intersection of trapezoidal segments
        """
        a = np.tan(alpha) * breadth
        b = np.tan(beta) * breadth
        if (a + b) > length:
            raise MagnetostaticsError(
                f"{self.__class__.__name__} instantiation error: source length and "
                "angles imply a self-intersecting trapezoidal prism."
            )


class SourceGroup(ABC):
    """
    Abstract base class for multiple current sources.
    """

    sources: list[CurrentSource]
    _points: np.array

    def __init__(self, sources: list[CurrentSource]):
        self.sources = sources
        self._points = np.vstack([np.vstack(s._points) for s in self.sources])

    def set_current(self, current: float):
        """
        Set the current inside each of the circuits.

        Parameters
        ----------
        current:
            The current of each circuit [A]
        """
        for source in self.sources:
            source.set_current(current)

    def field(
        self,
        x: float | npt.NDArray[np.float64],
        y: float | npt.NDArray[np.float64],
        z: float | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Calculate the magnetic field at a point.

        Parameters
        ----------
        x:
            The x coordinate(s) of the points at which to calculate the field
        y:
            The y coordinate(s) of the points at which to calculate the field
        z:
            The z coordinate(s) of the points at which to calculate the field

        Returns
        -------
        The magnetic field vector {Bx, By, Bz} in [T]
        """
        return np.sum([source.field(x, y, z) for source in self.sources], axis=0)

    def rotate(self, angle: float, axis: np.ndarray | str):
        """
        Rotate the CurrentSource about an axis.

        Parameters
        ----------
        angle:
            The rotation degree [rad]
        axis:
            The axis of rotation
        """
        for source in self.sources:
            source.rotate(angle, axis)
        self._points @= rotation_matrix(angle, axis)

    def plot(self, ax: Axes | None = None, *, show_coord_sys: bool = False):
        """
        Plot the MultiCurrentSource.

        Parameters
        ----------
        ax:
            The matplotlib axes to plot on
        show_coord_sys:
            Whether or not to plot the coordinate systems
        """
        if ax is None:
            ax = Plot3D()

        # Invisible bounding box to set equal aspect ratio plot
        xbox, ybox, zbox = BoundingBox.from_xyz(*self._points.T).get_box_arrays()
        ax.plot(1.1 * xbox, 1.1 * ybox, 1.1 * zbox, "s", alpha=0)

        for source in self.sources:
            source.plot(ax=ax, show_coord_sys=show_coord_sys)

    def copy(self):
        """
        Get a deepcopy of the SourceGroup.
        """
        return deepcopy(self)
