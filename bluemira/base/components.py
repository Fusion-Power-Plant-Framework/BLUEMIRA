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
Module containing the base Component class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Tuple, Union

import anytree
from anytree import NodeMixin, RenderTree

from bluemira.base.error import ComponentError
from bluemira.display.displayer import DisplayableCAD
from bluemira.display.plotter import Plottable

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

        if parent is not None and name in (ch.name for ch in parent.children):
            raise ComponentError(f"Component {name} is already a child of {parent}")

        if children is not None:
            ch_names = [ch.name for ch in children]
            if len(ch_names) != len(set(ch_names)):
                raise ComponentError(
                    f"Children have duplicate names for Component {name}",
                )

        self.parent = parent

        if children:
            self.children = children

    def __repr__(self) -> str:
        """
        The string representation of the instance
        """
        return self.name + " (" + self.__class__.__name__ + ")"

    def filter_components(
        self,
        names: Iterable[str],
        component_filter: Optional[Callable[[Component], bool]] = None,
    ):
        """
        Removes all components from the tree, starting at this component,
        that are siblings of each component specified in `names`
        and that aren't in `names` themselves.

        Parameters
        ----------
        names:
            The list of names of each component to search for.
        component_filter:
            A callable to filter Components from the Component tree,
            returning True keeps the node False removes it

        Notes
        -----
            This function mutates components in the subtree
        """
        for n in names:
            descendent_comps = self.get_component(
                n,
                first=False,
            )

            if descendent_comps is None:
                continue
            if not isinstance(descendent_comps, Iterable):
                descendent_comps = [descendent_comps]

            # Filter out all siblings that are not in names
            for c in descendent_comps:
                for c_sib in c.siblings:
                    if c_sib.name not in names:
                        c_sib.parent = None

                for c_child in c.descendants:
                    if component_filter is not None and not component_filter(c_child):
                        c_child.parent = None

    def tree(self) -> str:
        """
        Get the tree of descendants of this instance.
        """
        return str(RenderTree(self))

    def copy(
        self,
        parent: Optional[Component] = None,
    ) -> Component:
        """
        Copies this component and its children (recursively)
        and sets `parent` as this copy's parent.
        This only creates copies of each Component,
        the shape and material instances (for a PhysicalComponent for ex.)
        are shared (i.e. are the same instance).

        Parameters
        ----------
        parent:
            The component to set as the copy's parent

        Returns
        -------
        The copied component

        Notes
        -----
            This function should be overridden by implementors
        """
        # Initially copy self with None children
        self_copy = Component(
            name=self.name,
            parent=parent,
            children=None,
        )
        self_copy._plot_options = self._plot_options
        self_copy._display_cad_options = self._display_cad_options
        # Attaches children to parent
        self.copy_children(parent=self_copy)

        return self_copy

    def copy_children(
        self,
        parent: Component,
    ) -> list[Component]:
        """
        Copies this component's children (recursively)
        and sets `parent` as the copied children's parent.

        Parameters
        ----------
        parent:
            The component to set as the copied children's parent

        Returns
        -------
        The copied children components

        Notes
        -----
            This function should *not* be overridden by implementors
        """
        return [] if len(self.children) == 0 else [c.copy(parent) for c in self.children]

    def get_component(
        self, name: str, first: bool = True, full_tree: bool = False
    ) -> Union[Component, Tuple[Component], None]:
        """
        Find the components with the specified name.

        Parameters
        ----------
        name:
            The name of the component to search for.
        first:
            If True, only the first element is returned, by default True.
        full_tree:
            If True, searches the tree from the root, else searches from this node, by
            default False.

        Returns
        -------
        The first component of the search if first is True, else all components
        matching the search.

        Notes
        -----
            This function is just a wrapper of the anytree.search.findall
            function.
        """
        return self._get_thing(
            lambda n: anytree.search._filter_by_name(n, "name", name), first, full_tree
        )

    def get_component_properties(
        self,
        properties: Union[Iterable[str], str],
        first: bool = True,
        full_tree: bool = False,
    ) -> Union[Tuple[List[Any]], List[Any], Any]:
        """
        Get properties from a component

        Parameters
        ----------
        properties:
            properties to extract from component tree
        first:
            If True, only the first element is returned, by default True.
        full_tree:
            If True, searches the tree from the root, else searches from this node, by
            default False.

        Returns
        -------
        If multiple properties specified returns a tuple of the list of properties,
        otherwise returns a list of the property.
        If only one node has the property returns the value(s).

        Notes
        -----
            This function is just a wrapper of the anytree.search.findall or find
            functions.
        """
        if isinstance(properties, str):
            properties = [properties]

        def filter_(node, properties):
            return all(hasattr(node, prop) for prop in properties)

        found_nodes = self._get_thing(lambda n: filter_(n, properties), first, full_tree)

        if found_nodes is None:
            return tuple([] for _ in properties)

        if not isinstance(found_nodes, Iterable):
            if len(properties) == 1:
                return getattr(found_nodes, properties[0])
            return [getattr(found_nodes, prop) for prop in properties]
        # Collect values by property instead of by node
        node_properties = [
            [getattr(node, prop) for prop in properties] for node in found_nodes
        ]
        return tuple(map(list, zip(*node_properties)))

    def _get_thing(
        self, filter_: Union[Callable, None], first: bool, full_tree: bool
    ) -> Union[Component, Tuple[Component], None]:
        found_nodes = anytree.search.findall(
            self.root if full_tree else self, filter_=filter_
        )
        if found_nodes in (None, ()):
            return None
        if first and isinstance(found_nodes, Iterable):
            found_nodes = found_nodes[0]
        return found_nodes

    def add_child(self, child: Component) -> Component:
        """
        Add a single child to this node

        Parameters
        ----------
        child:
            The child to be added

        Returns
        -------
        This component.
        """
        # TODO: Support merge_trees here too.
        if child in self.children or child.name in (ch.name for ch in self.children):
            raise ComponentError(f"Component {child} is already a child of {self}")
        self.children = [*list(self.children), child]

        return self

    def add_children(
        self,
        children: Optional[Union[Component, List[Component]]],
        merge_trees: bool = False,
    ) -> Optional[Component]:
        """
        Add multiple children to this node

        Parameters
        ----------
        children:
            The children to be added

        Returns
        -------
        This component.
        """
        if children is None:
            return None
        if isinstance(children, Component):
            return self.add_child(children)
        if not isinstance(children, list):
            return None
        if len(children) == 0:
            return None

        duplicates = []
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
                f"Components {duplicates} are already children of {self}"
            )
        self.children = list(self.children) + children

        return self

    def prune_child(self, name: str):
        """
        Remove the child with the given name, and all its children.
        """
        found_component = anytree.search.find_by_attr(self, name)
        if found_component:
            # Method of deleting a node suggested by library author
            # https://github.com/c0fec0de/anytree/issues/152
            found_component.parent = None


class PhysicalComponent(Component):
    """
    A physical component. It includes shape and materials.

    Parameters
    ----------
    name:
        Name of the PhysicalComponent
    shape:
        Geometry of the PhysicalComponent
    material:
        Material of the PhysicalComponent
    parent:
        Parent of the PhysicalComponent
    children:
        Children of the PhysicalComponent
    """

    def __init__(
        self,
        name: str,
        shape: BluemiraGeo,
        material: Any = None,
        parent: Optional[Component] = None,
        children: Optional[List[Component]] = None,
    ):
        super().__init__(name, parent, children)
        self.shape = shape
        self.material = material

    def copy(
        self,
        parent: Optional[Component] = None,
    ) -> Component:
        """
        Copies this component and its children (recursively)
        and sets `parent` as this copy's parent.
        This only creates copies of each Component,
        the shape and material instances (for a PhysicalComponent for ex.)
        are shared (i.e. are the same instance).
        """
        # Initially copy self with None children
        self_copy = PhysicalComponent(
            name=self.name,
            parent=parent,
            children=None,
            shape=self.shape,
            material=self.material,
        )
        self_copy._plot_options = self._plot_options
        self_copy._display_cad_options = self._display_cad_options
        # Attaches children to parent
        self.copy_children(parent=self_copy)

        return self_copy

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
        parent: Optional[Component] = None,
        children: Optional[List[Component]] = None,
    ):
        super().__init__(name, shape, material, parent, children)
        self.conductor = conductor

    def copy(
        self,
        parent: Optional[Component] = None,
    ) -> Component:
        """
        Copies this component and its children (recursively)
        and sets `parent` as this copy's parent.
        This only creates copies of each Component,
        the shape and material instances (for a PhysicalComponent for ex.)
        are shared (i.e. are the same instance).
        """
        # Initially copy self with None children
        self_copy = MagneticComponent(
            name=self.name,
            parent=parent,
            children=None,
            shape=self.shape,
            material=self.material,
            conductor=self.conductor,
        )
        self_copy._plot_options = self._plot_options
        self_copy._display_cad_options = self._display_cad_options

        # Attaches children to parent
        self.copy_children(parent=self_copy)

        return self_copy

    @property
    def conductor(self):
        """
        The conductor used by current-carrying filaments.
        """
        return self._conductor

    @conductor.setter
    def conductor(self, value):
        self._conductor = value


def get_properties_from_components(
    comps: Union[Component, Iterable[Component]], properties: Union[str, Iterable[str]]
) -> Union[Tuple[List[Any]], List[Any], Any]:
    """
    Get properties from Components

    Parameters
    ----------
    comps:
        A component or list of components
    properties:
        properties to collect

    Returns
    -------
    If multiple properties specified returns a tuple of the list of properties,
    otherwise returns a list of the property.
    If only one node has the property returns the value(s).
    """
    if isinstance(properties, str):
        properties = [properties]

    property_lists = tuple([] for _ in properties)

    if not isinstance(comps, Iterable):
        comps = [comps]

    for comp in comps:
        props = comp.get_component_properties(properties, first=False)
        if not isinstance(props, tuple):
            props = tuple(props)
        for i, prop in enumerate(props):
            property_lists[i].extend(prop)

    if len(property_lists[0]) == 1:
        property_lists = [p[0] for p in property_lists]

    if len(property_lists) == 1:
        property_lists = property_lists[0]

    return property_lists
