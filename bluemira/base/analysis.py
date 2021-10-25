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
Module containing the bluemira Analysis class.
"""

from typing import Any, Dict, List, Type

from bluemira.base.builder import Builder
from bluemira.base.components import ComponentManager
from bluemira.base.parameter import ParameterFrame
from bluemira.utilities.tools import get_module


class Analysis:
    """
    The Analysis class is the main bluemira class that controls the design study that is
    being performed. It allows a series of Builder objects to be created, which are
    configured using the input parameters and configuration and walked though to produce
    the analysis results and reactor components.
    """

    _required_params: List[str] = []
    _params: ParameterFrame
    _build_config: Dict[str, Dict[str, Any]]
    _builders: List[Builder]
    _component_manager: ComponentManager

    def __init__(self, params, build_config, trees=["xz", "xy", "xyz"]):
        self._params = params
        self._build_config = build_config
        self._extract_builders()
        self._component_manager = ComponentManager(trees)
        self._params = ParameterFrame.from_template(self._required_params)
        self._params.update_kw_parameters(params)

    @property
    def component_manager(self):
        return self._component_manager

    @property
    def params(self):
        return self._params

    def run(self):
        for builder in self._builders:
            build_result = builder.build(self._params)
            for result in build_result:
                self._component_manager.insert_at_path(result[0], result[1])
            self._params.update_kw_parameters(builder._params.to_dict())

    def _extract_builders(self):
        """
        Extracts the builders from the config, which must be an ordered dictionary
        mapping the name of the builder to the corresponding options.
        """

        def _get_builder_class(builder_class: str) -> Type[Builder]:
            module = "bluemira.builders"
            class_name = builder_class
            if "::" in class_name:
                module, class_name = class_name.split("::")
            return getattr(get_module(module), class_name)

        self._builders = []
        for key, val in self._build_config.items():
            class_name = val.pop("class")
            val["name"] = key
            builder_class = _get_builder_class(class_name)
            self._builders += [builder_class(self._params, val)]
            self._required_params += self._builders[-1]._required_params
