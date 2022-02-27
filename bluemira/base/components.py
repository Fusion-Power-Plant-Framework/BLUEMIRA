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
Module containing the base Component class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Union

import anytree
from anytree import NodeMixin, RenderTree

from bluemira.base.error import ComponentError
from bluemira.display.displayer import DisplayableCAD
from bluemira.display.plotter import (
    BasePlotter,
    Plottable,
    get_plotter,
    register_plotter,
)

if TYPE_CHECKING:
    from bluemira.geometry.base import BluemiraGeo


class Component(NodeMixin, Plottable, DisplayableCAD):
    """
    The Component is the fundamental building block for a bluemira reactor design. It
    encodes the way that the corresponding part of the reactor will be built, along with
    any other derived properties that relate to that component.

    Components define a tree structure, based on the parent and children properties. This
    allows the nodes on that tree to be passed around within bluemira so that
    operations can be performed on the child branches of that structure.

    For example, a reactor design including just a TFCoilSystem may look as below:

    .. digraph:: base_component_tree

      "FusionPowerPlant" -> "TFCoilSystem" -> {"TFWindingPack" "TFCasing"}

    A Component cannot be used directly - only subclasses should be instantiated.
    """

    name: str

    def __init__(
        self,
        name: str,
        parent: Optional[Component] = None,
        children: Optional[List[Component]] = None,
    ):
        super().__init__()
        self.name = name
        self.parent = parent
        if children:
            self.children = children

    def __repr__(self) -> str:
        """
        The string representation of the instance
        """
        return self.name + " (" + self.__class__.__name__ + ")"

    def tree(self) -> str:
        """
        Get the tree of descendants of this instance.
        """
        return str(RenderTree(self))

    def get_component(
        self, name: str, first: bool = True, full_tree: bool = False
    ) -> Union["Component", List[Component]]:
        """
        Find the components with the specified name.

        Parameters
        ----------
        name: str
            The name of the component to search for.
        first: bool
            If True, only the first element is returned, by default True.
        full_tree: bool
            If True, searches the tree from the root, else searches from this node, by
            default False.

        Returns
        -------
        found_components: Union[Component, List[Component]]
            The first component of the search if first is True, else all components
            matching the search.

        Notes
        -----
            This function is just a wrapper of the anytree.search.findall_by_attr
            function.
        """
        if full_tree:
            found_components = anytree.search.findall_by_attr(self.root, name)
        else:
            found_components = anytree.search.findall_by_attr(self, name)

        if len(found_components) == 0:
            return None

        if first:
            return found_components[0]

        return found_components

    def add_child(self, child: Component) -> Component:
        """
        Add a single child to this node

        Parameters
        ----------
        child: Component
            The child to be added

        Returns
        -------
        self: Component
            This component.
        """
        # TODO: Support merge_trees here too.
        if child in self.children:
            raise ComponentError(f"Component {child} is already a child of {self}")
        self.children = list(self.children) + [child]

        return self

    def add_children(self, children: List[Component], merge_trees=False):
        """
        Add multiple children to this node

        Parameters
        ----------
        children: List[Component]
            The children to be added

        Returns
        -------
        self: Component
            This component.
        """
        if not isinstance(children, list) or len(children) == 0:
            child = children[0] if isinstance(children, list) else children
            return self.add_child(child)

        duplicates = []
        child: Component
        for idx, child in reversed(list(enumerate(children))):
            existing = self.get_component(child.name)
            if existing is not None:
                if merge_trees:
                    existing.children = list(existing.children) + list(child.children)
                    children.pop(idx)
                else:
                    duplicates += [child]
        if duplicates != []:
            raise ComponentError(
                f"Components {duplicates} are already a children of {self}"
            )
        self.children = list(self.children) + children

        return self


class PhysicalComponent(Component):
    """
    A physical component. It includes shape and materials.
    """

    def __init__(
        self,
        name: str,
        shape: BluemiraGeo,
        material: Any = None,
        parent: Component = None,
        children: Component = None,
    ):
        super().__init__(name, parent, children)
        self.shape = shape
        self.material = material

    @property
    def shape(self) -> BluemiraGeo:
        """
        The geometric shape of the Component.
        """
        return self._shape

    @shape.setter
    def shape(self, value: BluemiraGeo):
        self._shape = value

    @property
    def material(self):
        """
        The material that the Component is built from.
        """
        return self._material

    @material.setter
    def material(self, value):
        self._material = value


class MagneticComponent(PhysicalComponent):
    """
    A magnetic component. It includes a shape, a material, and a source conductor.
    """

    def __init__(
        self,
        name: str,
        shape: BluemiraGeo,
        material: Any = None,
        conductor: Any = None,
        parent: Component = None,
        children: Component = None,
    ):
        super().__init__(name, shape, material, parent, children)
        self.conductor = conductor

    @property
    def conductor(self):
        """
        The conductor used by current-carrying filaments.
        """
        return self._conductor

    @conductor.setter
    def conductor(self, value):
        self._conductor = value


class ComponentPlotter(BasePlotter):
    """
    Plotting class for bluemira components.
    """

    _CLASS_PLOT_OPTIONS = {"show_points": False, "show_wires": False}

    def _check_obj(self, obj):
        if not isinstance(obj, Component):
            raise ValueError(f"{obj} must be a BluemiraComponent")
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

    def _populate_data(self, comp: Component):
        self._cplotters: List[BasePlotter] = []

        def _populate_plotters(comp: Component):
            if comp.is_leaf and hasattr(comp, "shape"):
                options = (
                    self.options if comp.plot_options is None else comp.plot_options
                )
                plotter = get_plotter(comp.shape)(options)
                plotter._populate_data(comp.shape)
                self._cplotters.append(plotter)
            else:
                for child in comp.children:
                    _populate_plotters(child)

        _populate_plotters(comp)

    def _make_plot_2d(self):
        for plotter in self._cplotters:
            plotter.ax = self.ax
            plotter._make_plot_2d()
        self._set_aspect_2d()

    def _make_plot_3d(self):
        """
        Internal function that makes the plot. It should use self._data and
        self._data_to_plot, so _populate_data should be called before.
        """
        for plotter in self._cplotters:
            plotter.ax = self.ax
            plotter._make_plot_3d()

    def _set_aspect_3d(self):
        pass


register_plotter(Component, ComponentPlotter)
