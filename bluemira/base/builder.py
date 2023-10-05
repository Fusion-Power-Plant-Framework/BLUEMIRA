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
Interfaces for builder classes.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Dict, List, Optional, Type, Union

from bluemira.base.components import Component
from bluemira.base.parameter_frame import ParameterFrame, make_parameter_frame
from bluemira.base.reactor import ComponentManager  # noqa: F401 for compatibility
from bluemira.base.tools import _timing
from bluemira.utilities.plot_tools import set_component_view

if TYPE_CHECKING:
    from bluemira.base.reactor_config import ConfigParams

BuildConfig = Dict[str, Union[int, float, str, "BuildConfig"]]
"""
Type alias for representing nested build configuration information.
"""


def _remove_suffix(s: str, suffix: str) -> str:
    # Python 3.9 has str.removesuffix()
    if suffix and s.endswith(suffix):
        return s[: -len(suffix)]
    return s


class Builder(abc.ABC):
    """
    Base class for component builders.

    Parameters
    ----------
    params:
        The parameters required by the builder.
    build_config:
        The build configuration for the builder.
    verbose:
        control how much logging the designer will output

    Notes
    -----
    If there are no parameters associated with a concrete builder, set
    `param_cls` to `None` and pass `None` into this class's constructor.
    """

    def __init__(
        self,
        params: Union[Dict, ParameterFrame, ConfigParams, None],
        build_config: Optional[Dict] = None,
        *,
        verbose=True,
    ):
        super().__init__()
        self.params = make_parameter_frame(params, self.param_cls)
        self.build_config = build_config if build_config is not None else {}
        self.name = self.build_config.get(
            "name", _remove_suffix(self.__class__.__name__, "Builder")
        )
        self.build = _timing(
            self.build, "Built in", f"Building {self.name}", debug_info_str=not verbose
        )

    @abc.abstractproperty
    def param_cls(self) -> Union[Type[ParameterFrame], None]:
        """The class to hold this Builders's parameters."""

    @abc.abstractmethod
    def build(self) -> Component:
        """Build the component."""

    def component_tree(
        self, xz: List[Component], xy: List[Component], xyz: List[Component]
    ) -> Component:
        """
        Adds views of components to an overall component tree.

        Parameters
        ----------
        xz:
            xz view of component
        xy:
            xy view of component
        xyz:
            xyz view of component
        """
        component = Component(self.name)
        component.add_child(Component("xz", children=xz))
        component.add_child(Component("xy", children=xy))
        component.add_child(Component("xyz", children=xyz))

        set_component_view(component.get_component("xz"), "xz")
        set_component_view(component.get_component("xy"), "xy")

        return component
