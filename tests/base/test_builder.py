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

from typing import ClassVar

from bluemira.base.builder import Builder
from bluemira.base.components import Component


class ParamClass:
    def __init__(self, param_1, param_2) -> None:
        self.param_1 = param_1
        self.param_2 = param_2

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class StubBuilder(Builder):
    param_cls = ParamClass

    def build(self):
        return super().build()


class TestBuilder:
    _params: ClassVar = {
        "param_1": {"unit": "m", "value": 1},
        "param_2": {"unit": "T", "value": 2},
    }

    def test_default_name_is_class_name_sans_builder(self):
        builder = StubBuilder(self._params, {})

        assert builder.name == "Stub"

    def test_default_name_is_class_name_sans_build_config(self):
        builder = StubBuilder(self._params, None)

        assert builder.name == "Stub"

    def test_component_tree(self):
        builder = StubBuilder(self._params, {})
        component = builder.component_tree(
            xz=[Component("p1")], xy=[Component("p2")], xyz=[Component("p3")]
        )
        assert len(component.descendants) == 6
        assert len(component.children) == 3
        assert [ch.name for ch in component.children] == ["xz", "xy", "xyz"]
        # xyz child's view is not changed from the default
        assert [
            desc.plot_options.view_placement.label for desc in component.descendants
        ] == [
            "xzy",
            "xzy",
            "xyz",
            "xyz",
            "xzy",
            "xzy",
        ]
