# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Base class for a bluemira reactor."""

from __future__ import annotations

import abc
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_type_hints

from rich.progress import track

from bluemira.base.components import Component, get_properties_from_components
from bluemira.base.error import ComponentError
from bluemira.base.look_and_feel import bluemira_print
from bluemira.builders.tools import circular_pattern_component
from bluemira.display.displayer import ComponentDisplayer
from bluemira.display.plotter import ComponentPlotter
from bluemira.geometry.tools import save_cad
from bluemira.materials.material import Material, Void

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from os import PathLike

    import bluemira.codes._freecadapi as cadapi
    from bluemira.base.components import ComponentT
    from bluemira.geometry.base import BluemiraGeoT

_PLOT_DIMS = ["xy", "xz"]
_CAD_DIMS = ["xy", "xz", "xyz"]


class BaseManager(abc.ABC):
    """
    A base wrapper around a component tree or component trees.

    The purpose of the classes deriving from this is to abstract away
    the structure of the component tree and provide access to a set of
    its features. This way a reactor build procedure can be completely
    agnostic of the structure of component trees.

    """

    @abc.abstractmethod
    def component(self) -> ComponentT:
        """
        Return the component tree wrapped by this manager.
        """

    @abc.abstractmethod
    def save_cad(
        self,
        components: ComponentT | Iterable[ComponentT],
        filename: str,
        cad_format: str | cadapi.CADFileType = "stp",
        **kwargs,
    ):
        """
        Save the CAD build of the component.

        Parameters
        ----------
        components:
            components to save
        filename:
            the filename to save
        cad_format:
            CAD file format
        """
        shapes: list[BluemiraGeoT]
        names: list[str]
        shapes, names = get_properties_from_components(components, ("shape", "name"))
        save_cad(shapes, filename, cad_format, names, **kwargs)

    @abc.abstractmethod
    def show_cad(
        self,
        *dims: str,
        component_filter: Callable[[ComponentT], bool] | None,
        **kwargs,
    ):
        """
        Show the CAD build of the component.

        Parameters
        ----------
        *dims:
            The dimension of the reactor to show, typically one of
            'xz', 'xy', or 'xyz'. (default: 'xyz')
        component_filter:
            A callable to filter Components from the Component tree,
            returning True keeps the node False removes it
        """

    @abc.abstractmethod
    def plot(self, *dims: str, component_filter: Callable[[ComponentT], bool] | None):
        """
        Plot the component.

        Parameters
        ----------
        *dims:
            The dimension(s) of the reactor to show, 'xz' and/or 'xy'.
            (default: 'xz')
        component_filter:
            A callable to filter Components from the Component tree,
            returning True keeps the node False removes it
        """

    def tree(self) -> str:
        """
        Get the component tree.

        Returns
        -------
        :
            The component tree as a string.
        """
        return self.component().tree()

    @staticmethod
    def _validate_cad_dims(*dims: str) -> tuple[str, ...]:
        """
        Validate showable CAD dimensions.

        Returns
        -------
        :
            The validated dimensions

        Raises
        ------
        ComponentError
            Unknown plot dimension
        """
        # give dims_to_show a default value
        dims_to_show = ("xyz",) if len(dims) == 0 else dims

        for dim in dims_to_show:
            if dim not in _CAD_DIMS:
                raise ComponentError(
                    f"Invalid plotting dimension '{dim}'. Must be one of {_CAD_DIMS!s}"
                )

        return dims_to_show

    @staticmethod
    def _validate_plot_dims(*dims) -> tuple[str, ...]:
        """
        Validate showable plot dimensions.

        Returns
        -------
        :
            The validated dimensions

        Raises
        ------
        ComponentError
            Unknown plot dimension
        """
        # give dims_to_show a default value
        dims_to_show = ("xz",) if len(dims) == 0 else dims

        for dim in dims_to_show:
            if dim not in _PLOT_DIMS:
                raise ComponentError(
                    f"Invalid plotting dimension '{dim}'. Must be one of {_PLOT_DIMS!s}"
                )

        return dims_to_show

    @staticmethod
    def _filter_tree(
        comp: ComponentT,
        dims_to_show: tuple[str, ...],
        component_filter: Callable[[ComponentT], bool] | None,
    ) -> ComponentT:
        """
        Filter a component tree.

        Returns
        -------
        :
            The filtered component tree.

        Notes
        -----
        A copy of the component tree is made
        as filtering would mutate the ComponentMangers' underlying component trees
        """
        comp_copy = comp.copy()
        comp_copy.filter_components(dims_to_show, component_filter)
        return comp_copy

    def _plot_dims(
        self,
        comp: ComponentT,
        dims_to_show: tuple[str, ...],
        component_filter: Callable[[ComponentT], bool] | None,
        show: bool = True,
    ):
        for i, dim in enumerate(dims_to_show):
            if show:
                sub_show = i == len(dims_to_show) - 1
            else:
                sub_show = False
            ComponentPlotter(view=dim).plot_2d(
                self._filter_tree(comp, dims_to_show, component_filter),
                show=sub_show,
            )


class FilterMaterial:
    """
    Filter nodes by material

    Parameters
    ----------
    keep_material:
       materials to include
    reject_material:
       materials to exclude

    """

    __slots__ = ("keep_material", "reject_material")

    def __init__(
        self,
        keep_material: type[Material] | tuple[type[Material]] | None = None,
        reject_material: type[Material] | tuple[type[Material]] | None = Void,
    ):
        super().__setattr__("keep_material", keep_material)
        super().__setattr__("reject_material", reject_material)

    def __call__(self, node: ComponentT) -> bool:
        """Filter node based on material include and exclude rules.

        Parameters
        ----------
        node:
            The node to filter.

        Returns
        -------
        :
            True if the node should be kept, False otherwise.
        """
        if hasattr(node, "material"):
            return self._apply_filters(node.material)
        return True

    def __setattr__(self, name: str, value: Any):
        """
        Override setattr to force immutability

        This method makes the class nearly immutable as no new attributes
        can be modified or added by standard methods.

        See #2236 discussion_r1191246003 for further details

        Raises
        ------
        AttributeError
            FilterMaterial is immutable
        """
        raise AttributeError(f"{type(self).__name__} is immutable")

    def _apply_filters(self, material: Material | tuple[Material]) -> bool:
        bool_store = True

        if self.keep_material is not None:
            bool_store = isinstance(material, self.keep_material)

        if self.reject_material is not None:
            bool_store = not isinstance(material, self.reject_material)

        return bool_store


class ComponentManager(BaseManager):
    """
    A wrapper around a component tree.

    The purpose of the classes deriving from this is to abstract away
    the structure of the component tree and provide access to a set of
    its features. This way a reactor build procedure can be completely
    agnostic of the structure of component trees, relying instead on
    a set of methods implemented on concrete `ComponentManager`
    instances.

    This class can also be used to hold 'construction geometry' that may
    not be part of the component tree, but was useful in construction
    of the tree, and could be subsequently useful (e.g., an equilibrium
    can be solved to get a plasma shape, the equilibrium is not
    derivable from the plasma component tree, but can be useful in
    other stages of a reactor build procedure).

    Parameters
    ----------
    component_tree:
        The component tree this manager should wrap.
    """

    def __init__(self, component_tree: ComponentT) -> None:
        self._component = component_tree

    def component(self) -> ComponentT:
        """
        Return the component tree wrapped by this manager.

        Returns
        -------
        :
            The underlying component, with all descendants.
        """
        return self._component

    def save_cad(
        self,
        *dims: str,
        component_filter: Callable[[ComponentT], bool] | None = FilterMaterial(),
        filename: str | None = None,
        cad_format: str | cadapi.CADFileType = "stp",
        directory: str | PathLike = "",
        **kwargs,
    ):
        """
        Save the CAD build of the component.

        Parameters
        ----------
        *dims:
            The dimension of the reactor to show, typically one of
            'xz', 'xy', or 'xyz'. (default: 'xyz')
        component_filter:
            A callable to filter Components from the Component tree,
            returning True keeps the node False removes it
        filename:
            the filename to save, will default to the component name
        cad_format:
            CAD file format
        directory:
            Directory to save into, defaults to the current directory
        kwargs:
            passed to the :func:`bluemira.geometry.tools.save_cad` function
        """
        comp = self.component()
        if filename is None:
            filename = comp.name

        super().save_cad(
            self._filter_tree(comp, self._validate_cad_dims(*dims), component_filter),
            filename=Path(directory, filename).as_posix(),
            cad_format=cad_format,
            **kwargs,
        )

    def show_cad(
        self,
        *dims: str,
        component_filter: Callable[[ComponentT], bool] | None = FilterMaterial(),
        **kwargs,
    ):
        """
        Show the CAD build of the component.

        Parameters
        ----------
        *dims:
            The dimension of the reactor to show, typically one of
            'xz', 'xy', or 'xyz'. (default: 'xyz')
        component_filter:
            A callable to filter Components from the Component tree,
            returning True keeps the node False removes it
        kwargs:
            passed to the `~bluemira.display.displayer.show_cad` function
        """
        ComponentDisplayer().show_cad(
            self._filter_tree(
                self.component(),
                self._validate_cad_dims(*dims),
                component_filter,
            ),
            **kwargs,
        )

    def plot(
        self,
        *dims: str,
        component_filter: Callable[[ComponentT], bool] | None = FilterMaterial(),
    ):
        """
        Plot the component.

        Parameters
        ----------
        *dims:
            The dimension(s) of the reactor to show, 'xz' and/or 'xy'.
            (default: 'xz')
        component_filter:
            A callable to filter Components from the Component tree,
            returning True keeps the node False removes it
        """
        self._plot_dims(
            self.component(), self._validate_plot_dims(*dims), component_filter
        )


class Reactor(BaseManager):
    """
    Base class for reactor definitions.

    Assign :obj:`bluemira.base.builder.ComponentManager` instances to
    fields defined on the reactor, and this class acts as a container
    to group those components' trees. It is also a place to define any
    methods to calculate/derive properties that require information
    about multiple reactor components.

    Components should be defined on the reactor as class properties
    annotated with a type (similar to a ``dataclass``). A type that
    subclasses ``ComponentManager`` must be given, or it will not be
    recognised as part of the reactor tree. Note that a declared
    component is not required to be set for the reactor to be valid.
    So it is possible to just add a reactor's plasma, but not its
    TF coils, for example.

    Parameters
    ----------
    name:
        The name of the reactor. This will be the label for the top
        level :obj:`bluemira.base.components.Component` in the reactor
        tree.
    n_sectors:
        Number of sectors in a reactor

    Example
    -------

    .. code-block:: python

        class MyReactor(Reactor):
            '''An example of how to declare a reactor structure.'''

            plasma: MyPlasma
            tf_coils: MyTfCoils

            def get_ripple(self):
                '''Calculate the ripple in the TF coils.'''

        reactor = MyReactor("My Reactor", n_sectors=1)
        reactor.plasma = build_plasma()
        reactor.tf_coils = build_tf_coils()
        reactor.show_cad()

    """

    def __init__(self, name: str, n_sectors: int):
        self.name = name
        self.n_sectors = n_sectors
        self.start_time = time.perf_counter()

    def component(
        self,
        with_components: list[ComponentManager] | None = None,
    ) -> Component:
        """Return the component tree.

        Parameters
        ----------
        with_components:
            The components to include in the tree. If None, all components

        Returns
        -------
        :
            The list of components.
        """
        return self._build_component_tree(with_components)

    def time_since_init(self) -> float:
        """
        Get time since initialisation.

        Returns
        -------
        :
            The time since initialisation.
        """
        return time.perf_counter() - self.start_time

    def _build_component_tree(
        self,
        with_components: list[ComponentManager] | None = None,
    ) -> Component:
        """Build the component tree from this class's annotations.

        Returns
        -------
        :
            The component tree.

        Raises
        ------
        ComponentError
            Initialising Reactor directly
        """
        if not hasattr(self, "__annotations__"):
            raise ComponentError(
                "This reactor is ill-defined. "
                "Make sure you have sub-classed Reactor and "
                "correctly defined component managers for it. "
                "Please see the examples for a template Reactor."
            )

        component = Component(self.name)
        comp_type: type
        for comp_name, comp_type in get_type_hints(type(self)).items():
            if not issubclass(comp_type, ComponentManager):
                continue
            try:
                component_manager = getattr(self, comp_name)
                if (
                    with_components is not None
                    and component_manager not in with_components
                ):
                    continue
            except AttributeError:
                # We don't mind if a reactor component is not set, it
                # just won't be part of the tree
                continue

            component.add_child(component_manager.component())
        return component

    def _construct_xyz_cad(
        self,
        reactor_component: ComponentT,
        with_components: list[ComponentManager] | None = None,
        n_sectors: int = 1,
    ):
        xyzs = reactor_component.get_component(
            "xyz",
            first=False,
        )
        xyzs = [xyzs] if isinstance(xyzs, Component) else xyzs

        comp_names = (
            "all"
            if not with_components
            else ", ".join([cm.component().name for cm in with_components])
        )
        bluemira_print(
            f"Constructing xyz CAD for display with {n_sectors} sectors and components:"
            f" {comp_names}"
        )
        for xyz in track(xyzs):
            xyz.children = circular_pattern_component(
                list(xyz.children),
                n_sectors,
                degree=(360 / self.n_sectors) * n_sectors,
            )

    def _filter_and_reconstruct(
        self,
        dims_to_show: tuple[str, ...],
        with_components: list[ComponentManager] | None,
        n_sectors: int | None,
        component_filter: Callable[[ComponentT], bool] | None,
    ) -> Component:
        # We filter because self.component (above) only creates
        # a new root node for this reactor, not a new component tree.
        comp_copy = self._filter_tree(
            self.component(with_components), dims_to_show, component_filter
        )
        if not comp_copy.children:
            raise ComponentError(
                "The reactor has no components defined for the given "
                "dimension(s) and/or filter."
            )
        # if "xyz" is requested, construct the 3d cad
        # from each xyz component in the tree,
        # as it's assumed that the cad is only built for 1 sector
        # and is sector symmetric, therefore can be patterned
        if "xyz" in dims_to_show:
            self._construct_xyz_cad(
                comp_copy,
                with_components,
                self.n_sectors if n_sectors is None else n_sectors,
            )
        return comp_copy

    def save_cad(
        self,
        *dims: str,
        with_components: list[ComponentManager] | None = None,
        n_sectors: int | None = None,
        component_filter: Callable[[ComponentT], bool] | None = FilterMaterial(),
        filename: str | None = None,
        cad_format: str | cadapi.CADFileType = "stp",
        directory: str | PathLike = "",
        **kwargs,
    ):
        """
        Save the CAD build of the reactor.

        Parameters
        ----------
        *dims:
            The dimension of the reactor to show, typically one of
            'xz', 'xy', or 'xyz'. (default: 'xyz')
        with_components:
            The components to construct when displaying CAD for xyz.
            Defaults to None, which means show "all" components.
        n_sectors:
            The number of sectors to construct when displaying CAD for xyz
            Defaults to None, which means show "all" sectors.
        component_filter:
            A callable to filter Components from the Component tree,
            returning True keeps the node False removes it
        filename:
            the filename to save, will default to the component name
        cad_format:
            CAD file format
        directory:
            Directory to save into, defaults to the current directory
        kwargs:
            passed to the :func:`bluemira.geometry.tools.save_cad` function
        """
        if filename is None:
            filename = self.name

        super().save_cad(
            self._filter_and_reconstruct(
                self._validate_cad_dims(*dims),
                with_components,
                n_sectors,
                component_filter,
            ),
            Path(directory, filename).as_posix(),
            cad_format,
            **kwargs,
        )

    def show_cad(
        self,
        *dims: str,
        with_components: list[ComponentManager] | None = None,
        n_sectors: int | None = None,
        component_filter: Callable[[ComponentT], bool] | None = FilterMaterial(),
        **kwargs,
    ):
        """
        Show the CAD build of the reactor.

        Parameters
        ----------
        *dims:
            The dimension of the reactor to show, typically one of
            'xz', 'xy', or 'xyz'. (default: 'xyz')
        with_components:
            The components to construct when displaying CAD for xyz.
            Defaults to None, which means show "all" components.
        n_sectors:
            The number of sectors to construct when displaying CAD for xyz
            Defaults to None, which means show "all" sectors.
        component_filter:
            A callable to filter Components from the Component tree,
            returning True keeps the node False removes it
        kwargs:
            passed to the `~bluemira.display.displayer.show_cad` function
        """
        ComponentDisplayer().show_cad(
            self._filter_and_reconstruct(
                self._validate_cad_dims(*dims),
                with_components,
                n_sectors,
                component_filter,
            ),
            **kwargs,
        )

    def plot(
        self,
        *dims: str,
        with_components: list[ComponentManager] | None = None,
        component_filter: Callable[[ComponentT], bool] | None = FilterMaterial(),
        show: bool = True,
    ):
        """
        Plot the reactor.

        Parameters
        ----------
        *dims:
            The dimension(s) of the reactor to show, 'xz' and/or 'xy'.
            (default: 'xz')
        with_components:
            The components to construct when displaying CAD for xyz.
            Defaults to None, which means show "all" components.
        component_filter:
            A callable to filter Components from the Component tree,
            returning True keeps the node False removes it
        show:
            Whether or not to immediately display the plot
        """
        self._plot_dims(
            self.component(with_components),
            self._validate_plot_dims(*dims),
            component_filter,
            show=show,
        )
