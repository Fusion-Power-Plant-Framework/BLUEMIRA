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
api for plotting using matplotlib
"""
from __future__ import annotations

import copy
import inspect
import pprint
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

import matplotlib.pyplot as plt
import numpy as np

from bluemira.display.error import DisplayError
from bluemira.display.palettes import BLUE_PALETTE

if TYPE_CHECKING:
    from bluemira.geometry.base import BluemiraGeo

DEFAULT_PLOT_OPTIONS = {
    # flags to enable points, wires, and faces plot
    "show_points": False,
    "show_wires": True,
    "show_faces": True,
    # matplotlib set of options to plot points, wires, and faces. If an empty dictionary
    # is specified, the default color plot of matplotlib is used.
    "point_options": {"s": 10, "facecolors": "red", "edgecolors": "black", "zorder": 30},
    "wire_options": {"color": "black", "linewidth": 0.5, "zorder": 20},
    "face_options": {"color": "blue", "zorder": 10},
    # projection plane
    "plane": "xz",
    # discretization properties for plotting wires (and faces)
    "ndiscr": 100,
    "byedges": True,
}

UNIT_LABEL = "[m]"
X_LABEL = f"x {UNIT_LABEL}"
Y_LABEL = f"y {UNIT_LABEL}"
Z_LABEL = f"z {UNIT_LABEL}"


def get_default_options():
    """
    Returns the instance as a dictionary.
    """
    from bluemira.geometry import plane as _plane

    output_dict = {}
    for k, v in DEFAULT_PLOT_OPTIONS.items():
        # FreeCAD Plane that is contained in BluemiraPlane cannot be deepcopied by
        # copy.deepcopy. For this reason, its deepcopy method must be used in
        # this case.
        if isinstance(v, _plane.BluemiraPlane):
            output_dict[k] = v.deepcopy()
        else:
            output_dict[k] = copy.deepcopy(v)
    return output_dict


class DisplayOptions:
    """
    The options that are available for displaying objects.
    """

    _options = None

    def as_dict(self):
        """
        Returns the instance as a dictionary.
        """
        return copy.deepcopy(self._options)

    def modify(self, **kwargs):
        """
        Function to override plotting options.
        """
        if kwargs:
            for k in kwargs:
                if k in self._options:
                    self._options[k] = kwargs[k]

    def __repr__(self):
        """
        Representation string of the DisplayOptions.
        """
        return f"{self.__class__.__name__}({pprint.pformat(self._options)}" + "\n)"


# Note: This class cannot be an instance of dataclasses 'cause it fails when inserting
# a field that is a Base.Placement (or something that contains a Base.Placement). The
# given error is "TypeError: can't pickle Base.Placement objects"
class PlotOptions(DisplayOptions):
    """
    The options that are available for plotting objects in 2D.

    Parameters
    ----------
    show_points: bool
        If True, points are plotted. By default True.
    show_wires: bool
        If True, wires are plotted. By default True.
    show_faces: bool
        If True, faces are plotted. By default True.
    point_options: Dict
        Dictionary with matplotlib options for points. By default  {"s": 10,
        "facecolors": "blue", "edgecolors": "black"}
    wire_options: Dict
        Dictionary with matplotlib options for wires. By default {"color": "black",
        "linewidth": "0.5"}
    face_options: Dict
        Dictionary with matplotlib options for faces. By default {"color": "red"}
    plane: [str, Plane]
        The plane on which the object is projected for plotting. As string, possible
        options are "xy", "xz", "zy". By default 'xz'.
    ndiscr: int
        The number of points to use when discretising a wire or face.
    byedges: bool
        If True then wires or faces will be discretised respecting their edges.
    """

    def __init__(self, **kwargs):
        self._options = get_default_options()
        self.modify(**kwargs)

    def as_dict(self):
        """
        Returns the instance as a dictionary.
        """
        from bluemira.geometry import plane as _plane

        output_dict = {}
        for k, v in self._options.items():
            # FreeCAD Plane that is contained in BluemiraPlane cannot be deepcopied by
            # copy.deepcopy. For this reason, its deepcopy method must to be used in
            # this case.
            if isinstance(v, _plane.BluemiraPlane):
                output_dict[k] = v.deepcopy()
            else:
                output_dict[k] = copy.deepcopy(v)
        return output_dict

    @property
    def show_points(self):
        """
        If true, points are plotted.
        """
        return self._options["show_points"]

    @show_points.setter
    def show_points(self, val):
        self._options["show_points"] = val

    @property
    def show_wires(self):
        """
        If True, wires are plotted.
        """
        return self._options["show_wires"]

    @show_wires.setter
    def show_wires(self, val):
        self._options["show_wires"] = val

    @property
    def show_faces(self):
        """
        If True, faces are plotted.
        """
        return self._options["show_faces"]

    @show_faces.setter
    def show_faces(self, val):
        self._options["show_faces"] = val

    @property
    def point_options(self):
        """
        Dictionary with matplotlib options for points.
        """
        return self._options["point_options"]

    @point_options.setter
    def point_options(self, val):
        self._options["point_options"] = val

    @property
    def wire_options(self):
        """
        Dictionary with matplotlib options for wires.
        """
        return self._options["wire_options"]

    @wire_options.setter
    def wire_options(self, val):
        self._options["wire_options"] = val

    @property
    def face_options(self):
        """
        Dictionary with matplotlib options for faces.
        """
        return self._options["face_options"]

    @face_options.setter
    def face_options(self, val):
        self._options["face_options"] = val

    @property
    def plane(self):
        """
        The plane on which the object is projected for plotting. As string, possible
        options are "xy", "xz", "zy".
        """
        return self._options["plane"]

    @plane.setter
    def plane(self, val):
        self._options["plane"] = val

    @property
    def ndiscr(self):
        """
        The number of points to use when discretising a wire or face.
        """
        return self._options["ndiscr"]

    @ndiscr.setter
    def ndiscr(self, val):
        self._options["ndiscr"] = val

    @property
    def byedges(self):
        """
        If True then wires or faces will be discretised respecting their edges.
        """
        return self._options["byedges"]

    @byedges.setter
    def byedges(self, val):
        self._options["byedges"] = val


# Note: when plotting points, it can happen that markers are not centred properly as
# described in https://github.com/matplotlib/matplotlib/issues/11836
class BasePlotter(ABC):
    """
    Base utility plotting class
    """

    _CLASS_PLOT_OPTIONS = {}

    def __init__(self, options: Optional[PlotOptions] = None, **kwargs):
        # discretization points representing the shape in global coordinate system
        self._data = []
        # modified discretization points for plotting (e.g. after plane transformation)
        self._data_to_plot = []
        self.ax = None
        self.options = (
            PlotOptions(**self._CLASS_PLOT_OPTIONS) if options is None else options
        )
        self.options.modify(**kwargs)
        self.set_plane(self.options._options["plane"])

    def set_plane(self, plane):
        """Set the plotting plane"""
        from bluemira.geometry import plane as _plane

        if plane == "xy":
            # Base.Placement(origin, axis, angle)
            self.options._options["plane"] = _plane.BluemiraPlane()
        elif plane == "xz":
            # Base.Placement(origin, axis, angle)
            self.options._options["plane"] = _plane.BluemiraPlane(
                axis=(1.0, 0.0, 0.0), angle=-90.0
            )
        elif plane == "zy":
            # Base.Placement(origin, axis, angle)
            self.options._options["plane"] = _plane.BluemiraPlane(
                axis=(0.0, 1.0, 0.0), angle=90.0
            )
        elif isinstance(plane, _plane.BluemiraPlane):
            self.options._options["plane"] = plane
        else:
            DisplayError(f"{plane} is not a valid plane")

    @abstractmethod
    def _check_obj(self, obj):
        """Internal function that check if obj is an instance of the correct class"""
        pass

    @abstractmethod
    def _check_options(self):
        """Internal function that check if it is needed to plot something"""
        pass

    def initialize_plot_2d(self, ax=None):
        """Initialize the plot environment"""
        if ax is None:
            fig = plt.figure()
            self.ax = fig.add_subplot()
        else:
            self.ax = ax

    def show(self):
        """Function to show a plot"""
        plt.show(block=True)

    @abstractmethod
    def _populate_data(self, obj):
        """
        Internal function that makes the plot. It fills self._data and
        self._data_to_plot
        """
        pass

    @abstractmethod
    def _make_plot_2d(self):
        """
        Internal function that makes the plot. It should use self._data and
        self._data_to_plot, so _populate_data should be called before.
        """
        pass

    def _set_aspect_2d(self):
        self.ax.set_aspect("equal")

    def _set_label_2d(self):
        # TODO: BluemiraPlane stuff.... I feel like the axis may not be the plane normal
        # and this makes me very sad. The workaround here is by creating the planes from
        # strs, but we're unlikely to recognise a user-input BluemiraPlane.
        # Note that the angle argument gets used to rotate the data for reasons I still
        # do not understand, hence why the axis here is not the plane normal..
        axis = np.abs(self.options.plane.axis)
        if np.allclose(axis, (1, 0, 0)):
            self.ax.set_xlabel(X_LABEL)
            self.ax.set_ylabel(Z_LABEL)
        elif np.allclose(axis, (0, 0, 1)):
            self.ax.set_xlabel(X_LABEL)
            self.ax.set_ylabel(Y_LABEL)
        elif np.allclose(axis, (0, 1, 0)):
            self.ax.set_xlabel(Y_LABEL)
            self.ax.set_ylabel(Z_LABEL)
        else:
            # Do not put x,y,z labels for planes we do not recognise
            self.ax.set_xlabel(UNIT_LABEL)
            self.ax.set_ylabel(UNIT_LABEL)

    def _set_aspect_3d(self):
        from bluemira.geometry import bound_box

        # This was the only way I found to get 3-D plots to look right in matplotlib
        x_bb, y_bb, z_bb = bound_box.BoundingBox.from_xyz(*self._data.T).get_box_arrays()
        for x, y, z in zip(x_bb, y_bb, z_bb):
            self.ax.plot([x], [y], [z], color="w")

    def _set_label_3d(self):
        offset = "\n\n"  # To keep labels from interfering with the axes
        self.ax.set_xlabel(offset + X_LABEL)
        self.ax.set_ylabel(offset + Y_LABEL)
        self.ax.set_zlabel(offset + Z_LABEL)

    def plot_2d(self, obj, ax=None, show: bool = True):
        """2D plotting method"""
        self._check_obj(obj)

        if not self._check_options():
            self.ax = ax
        else:
            self.initialize_plot_2d(ax)
            self._populate_data(obj)
            self._make_plot_2d()
            self._set_aspect_2d()
            self._set_label_2d()

            if show:
                self.show()
        return self.ax

    # # =================================================================================
    # # 3-D functions
    # # =================================================================================
    def initialize_plot_3d(self, ax=None):
        """Initialize the plot environment"""
        if ax is None:
            fig = plt.figure()
            self.ax = fig.add_subplot(projection="3d")
        else:
            self.ax = ax

    @abstractmethod
    def _make_plot_3d(self):
        """Internal function that makes the plot. It should use self._data and
        self._data_to_plot, so _populate_data should be called before.
        """
        pass

    def plot_3d(self, obj, ax=None, show: bool = True):
        """3D plotting method"""
        self._check_obj(obj)

        if not self._check_options():
            self.ax = ax
        else:
            self.initialize_plot_3d(ax=ax)
            # this function can be common to 2D and 3D plot
            # self._data is used for 3D plot
            # self._data_to_plot is used for 2D plot
            # TODO: probably better to rename self._data_to_plot into
            #  self._projected_data or self._data2d
            self._populate_data(obj)
            self._make_plot_3d()
            self._set_aspect_3d()
            self._set_label_3d()

            if show:
                self.show()

        return self.ax


class PointsPlotter(BasePlotter):
    """
    Base utility plotting class for points
    """

    _CLASS_PLOT_OPTIONS = {"show_points": True}

    def _check_obj(self, obj):
        # TODO: create a function that checks if the obj is a cloud of 3D or 2D points
        return True

    def _check_options(self):
        # Check if nothing has to be plotted
        if not self.options.show_points:
            return False
        return True

    def _populate_data(self, points):
        from bluemira.geometry.coordinates import _parse_to_xyz_array

        points = _parse_to_xyz_array(points).T
        self._data = points
        # apply rotation matrix given by options['plane']
        rot = self.options.plane.to_matrix().T
        temp_data = np.c_[self._data, np.ones(len(self._data))]
        self._data_to_plot = temp_data.dot(rot).T
        self._data_to_plot = self._data_to_plot[0:2]

    def _make_plot_2d(self):
        if self.options.show_points:
            self.ax.scatter(*self._data_to_plot, **self.options.point_options)
        self._set_aspect_2d()

    def _make_plot_3d(self, *args, **kwargs):
        if self.options.show_points:
            self.ax.scatter(*self._data.T, **self.options.point_options)
        self._set_aspect_3d()


class _PlotterRegistry:
    """
    Provides a registry of plotters that are associated with other bluemira classes.
    """

    _registry: Dict[Type[Any], Type[BasePlotter]] = {}

    @staticmethod
    def register_plotter(
        cls: Type[Any],
        plotter: Type[BasePlotter],
    ):
        """
        Register the provided plotter with the provided class.

        Parameters
        ----------
        cls: Type[Any]
            The class to register the plotter with.
        plotter: Type[BasePlotter]
            The plotter to use when plotting the class.
        """
        if not inspect.isclass(cls):
            cls = type(cls)

        if cls not in _PlotterRegistry._registry:
            _PlotterRegistry._registry[cls] = plotter
        else:
            raise ValueError(f"Class {cls} already registered to use plotter {plotter}")

    @staticmethod
    def get_plotter(cls: Union[Any, Type[Any]]) -> Type[BasePlotter]:
        """
        Get the plotter that has been registered for the provided class.

        Parameters
        ----------
        cls: Union[Any, Type[Any]]
            The class to get the plotter for. If a non-class object is passed in
            then the type of that object will be used to find the plotter.

        Raises
        ------
        ValueError
            If the provided class does not have a registered plotter.
        """
        if not inspect.isclass(cls):
            cls = type(cls)

        if cls in _PlotterRegistry._registry:
            return _PlotterRegistry._registry[cls]
        else:
            for parent_cls in cls.__bases__:
                return _PlotterRegistry.get_plotter(parent_cls)
            raise ValueError(f"Class {cls} has not been registered with a plotter.")


def register_plotter(cls: Type[Any], plotter: Type[BasePlotter]):
    """
    Add the provided class to the plotting registry and associate it with the provided
    plotter.

    Parameters
    ----------
    cls: Type[Any]
        The class to register the plotter with.
    plotter: Type[BasePlotter]
        The plotter to use when plotting the class.
    """
    _PlotterRegistry.register_plotter(cls, plotter)


def get_plotter(cls: Union[Any, Type[Any]]) -> Type[BasePlotter]:
    """
    Get the plotter associated with the provided class.

    Parameters
    ----------
    cls: Union[Any, Type[Any]]
        The class for which to get the plotter. If a non-class object is passed in then
        the type of that object will be used to find the plotter.
    """
    return _PlotterRegistry.get_plotter(cls)


def _validate_plot_inputs(parts, options):
    """
    Validate the lists of parts and options, applying some default options.
    """
    if not isinstance(parts, list):
        parts = [parts]

    if options is None:
        options = [None] * len(parts)
    elif not isinstance(options, list):
        options = [options] * len(parts)

    if len(options) != len(parts):
        raise DisplayError(
            "If options for plot are provided then there must be as many options as "
            "there are parts to plot."
        )
    return parts, options


def _get_plotter_class(part):
    """
    Get the plotting class for a BluemiraGeo object.
    """
    try:
        return get_plotter(part)
    except ValueError:
        raise DisplayError(
            f"{part} object cannot be plotted. No Plotter available for {type(part)}"
        )


def plot_2d(
    parts: Union[BluemiraGeo, List[BluemiraGeo]],
    options: Optional[Union[PlotOptions, List[PlotOptions]]] = None,
    ax=None,
    show: bool = True,
    **kwargs,
):
    """
    The implementation of the display API for BluemiraGeo parts.

    Parameters
    ----------
    parts: Union[Part.Shape, List[Part.Shape]]
        The parts to display.
    options: Optional[Union[PlotOptions, List[PlotOptions]]]
        The options to use to display the parts.
    ax: Optional[Axes]
        The axes onto which to plot
    show: bool
        Whether or not to show the plot immediately (default=True). Note
        that if using iPython or Jupyter, this has no effect; the plot is shown
        automatically.
    """
    parts, options = _validate_plot_inputs(parts, options)

    for part, option in zip(parts, options):
        plotter = _get_plotter_class(part)(option, **kwargs)
        ax = plotter.plot_2d(part, ax, show=False)

    if show:
        plotter.show()

    return ax


def plot_3d(
    parts: Union[BluemiraGeo, List[BluemiraGeo]],
    options: Optional[Union[PlotOptions, List[PlotOptions]]] = None,
    ax=None,
    show: bool = True,
    **kwargs,
):
    """
    The implementation of the display API for BluemiraGeo parts.

    Parameters
    ----------
    parts: Union[Part.Shape, List[Part.Shape]]
        The parts to display.
    options: Optional[Union[Plot3DOptions, List[Plot3Options]]]
        The options to use to display the parts.
    ax: Optional[Axes]
        The axes onto which to plot
    show: bool
        Whether or not to show the plot immediately in the console. (default=True). Note
        that if using iPython or Jupyter, this has no effect; the plot is shown
        automatically.
    """
    parts, options = _validate_plot_inputs(parts, options)

    for part, option in zip(parts, options):
        plotter = _get_plotter_class(part)(option, **kwargs)
        ax = plotter.plot_3d(part, ax, show=False)

    if show:
        plotter.show()

    return ax


class Plottable:
    """
    Mixin class to make a class plottable in 2D by imparting a plot2d method and
    options.

    Notes
    -----
    The implementing class must set the _plotter2D attribute to an instance of the
    appropriate Plotter2D class.
    """

    def __init__(self):
        super().__init__()
        self._plot_options: PlotOptions = PlotOptions()
        self._plot_options.face_options["color"] = next(BLUE_PALETTE)

    @property
    def plot_options(self) -> PlotOptions:
        """
        The options that will be used to display the object.
        """
        return self._plot_options

    @plot_options.setter
    def plot_options(self, value: PlotOptions):
        if not isinstance(value, PlotOptions):
            raise DisplayError("Display options must be set to a PlotOptions instance.")
        self._plot_options = value

    @property
    def _plotter(self) -> BasePlotter:
        """
        The options that will be used to display the object.
        """
        return _get_plotter_class(self)(self._plot_options)

    def plot_2d(self, ax=None, show: bool = True) -> None:
        """
        Default method to call display the object by calling into the Displayer's display
        method.

        Returns
        -------
        axes
            The axes that the plot has been displayed onto.
        """
        return self._plotter.plot_2d(self, ax=ax, show=show)

    def plot_3d(self, ax=None, show: bool = True) -> None:
        """
        Function to 3D plot a component.

        Returns
        -------
        axes
            The axes that the plot has been displayed onto.
        """
        return self._plotter.plot_3d(self, ax=ax, show=show)


register_plotter(list, PointsPlotter)
register_plotter(np.ndarray, PointsPlotter)
