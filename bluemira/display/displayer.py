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
api for plotting using CAD backend
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, ClassVar, List, Optional, Union

from bluemira.base.look_and_feel import bluemira_debug, bluemira_warn
from bluemira.display.error import DisplayError
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.display.tools import Options
from bluemira.utilities.tools import get_module

if TYPE_CHECKING:
    from bluemira.geometry.base import BluemiraGeo


class ViewerBackend(Enum):
    """CAD viewer backends."""

    FREECAD = "bluemira.codes._freecadapi"
    POLYSCOPE = "bluemira.codes._polyscope"

    @lru_cache(2)
    def get_module(self):
        """Load viewer module"""
        try:
            return get_module(self.value)
        except (ModuleNotFoundError, FileNotFoundError):
            if self.name != "FREECAD":
                name = self.name.lower()
                bluemira_warn(
                    f"Unable to import {name.capitalize()} viewer\n"
                    f"Please 'pip install {name}' to use, falling back to FreeCAD."
                )
                return get_module(type(self).FREECAD.value)
            raise


def get_default_options(backend=ViewerBackend.FREECAD):
    """
    Returns the default display options.
    """
    return backend.get_module().DefaultDisplayOptions()


class DisplayCADOptions(Options):
    """
    The options that are available for displaying objects in 3D

    Parameters
    ----------
    backend
        the backend viewer being used
    """

    __slots__ = ()

    def __init__(self, backend=ViewerBackend.FREECAD, **kwargs):
        self._options = get_default_options(backend)
        super().__init__(**kwargs)


# =======================================================================================
# Visualisation
# =======================================================================================


def _validate_display_inputs(parts, options, labels):
    """
    Validate the lists of parts and options, applying some default options.
    """
    if parts is None:
        bluemira_debug("No new parts to display")
        return [], [], []

    if not isinstance(parts, list):
        parts = [parts]

    if not isinstance(options, list) or options is None:
        options = [options] * len(parts)

    if labels is None:
        labels = ""

    if isinstance(labels, str):
        labels = [labels] * len(parts)

    if len(options) != len(parts):
        raise DisplayError(
            "If options for plot are provided then there must be as many options as "
            "there are parts to plot."
        )
    return parts, options, labels


def show_cad(
    parts: Optional[
        Union[BluemiraGeo, List[BluemiraGeo]]
    ] = None,  # avoiding circular deps
    options: Optional[Union[DisplayCADOptions, List[DisplayCADOptions]]] = None,
    labels: Optional[Union[str, List[str]]] = None,
    backend: Union[str, ViewerBackend] = ViewerBackend.FREECAD,
    **kwargs,
):
    """
    The CAD display API.

    Parameters
    ----------
    parts
        The parts to display.
    options
        The options to use to display the parts.
    labels
        Labels to use for each part object
    backend
        Viewer backend
    kwargs
        Passed on to modifications to the plotting style options and backend
    """
    if isinstance(backend, str):
        try:
            backend = ViewerBackend[backend.upper()]
        except KeyError:
            bluemira_warn(f"Unknown viewer backend '{backend}' defaulting to FreeCAD")
            backend = ViewerBackend.FREECAD

    parts, options, labels = _validate_display_inputs(parts, options, labels)

    new_options = []
    for o in options:
        if isinstance(o, DisplayCADOptions):
            temp = DisplayCADOptions(**o.as_dict(), backend=backend)
            temp.modify(**kwargs)
            new_options.append(temp)
        else:
            new_options.append(DisplayCADOptions(**kwargs, backend=backend))

    backend.get_module().show_cad(
        [part.shape for part in parts],
        [o.as_dict() for o in new_options],
        labels,
        **kwargs,
    )


class BaseDisplayer(ABC):
    """
    Displayer abstract class
    """

    _CLASS_DISPLAY_OPTIONS: ClassVar = {}

    def __init__(self, options: Optional[DisplayCADOptions] = None, **kwargs):
        self.options = (
            DisplayCADOptions(**self._CLASS_DISPLAY_OPTIONS)
            if options is None
            else options
        )
        self.options.modify(**kwargs)

    @abstractmethod
    def show_cad(self, objs, **kwargs):
        """
        Display a CAD object
        """


def _get_displayer_class(part):
    """
    Get the displayer class for an object.
    """
    import bluemira.base.components

    if isinstance(part, bluemira.base.components.Component):
        plot_class = ComponentDisplayer
    else:
        raise DisplayError(
            f"{part} object cannot be displayed. No Displayer available for {type(part)}"
        )
    return plot_class


class DisplayableCAD:
    """
    Mixin class to make a class displayable by imparting a show_cad method and options.
    """

    def __init__(self):
        self._display_cad_options: DisplayCADOptions = DisplayCADOptions()
        self._display_cad_options.colour = next(BLUE_PALETTE)

    @property
    def display_cad_options(self) -> DisplayCADOptions:
        """
        The options that will be used to display the object.
        """
        return self._display_cad_options

    @display_cad_options.setter
    def display_cad_options(self, value: DisplayCADOptions):
        if not isinstance(value, DisplayCADOptions):
            raise DisplayError(
                "Display options must be set to a DisplayCADOptions instance."
            )
        self._display_cad_options = value

    @property
    def _displayer(self) -> BaseDisplayer:
        """
        The options that will be used to display the object.
        """
        return _get_displayer_class(self)(self.display_cad_options)

    def show_cad(self, **kwargs) -> None:
        """
        Default method to call display the object by calling into the Displayer's display
        method.

        Returns
        -------
        axes
            The axes that the plot has been displayed onto.
        """
        return self._displayer.show_cad(self, **kwargs)


class ComponentDisplayer(BaseDisplayer):
    """
    CAD displayer for Components
    """

    @staticmethod
    def show_cad(
        comps,
        **kwargs,
    ):
        """
        Display the CAD of a component or iterable of components

        Parameters
        ----------
        comp: Union[Iterable[Component], Component]
            Component, or iterable of Components, to be displayed
        """
        import bluemira.base.components as bm_comp

        show_cad(
            *bm_comp.get_properties_from_components(
                comps, ("shape", "display_cad_options", "name")
            ),
            **kwargs,
        )
