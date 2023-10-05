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

import copy
from typing import ClassVar

import pytest

from bluemira.base.parameter_frame import Parameter


class TestParameter:
    SERIALIZED_PARAM: ClassVar = {
        "name": "my_param",
        "value": 100,
        "description": "A parameter for testing on.",
        "source": "test",
        "unit": "dimensionless",
        "long_name": "My parameter",
    }

    def test_fields_set_on_init(self):
        param = Parameter(**self.SERIALIZED_PARAM)

        assert param.name == "my_param"
        assert param.value == 100
        assert param.description == "A parameter for testing on."
        assert param.long_name == "My parameter"
        assert param.source == "test"
        assert param.unit == ""

    def test_initial_value_added_to_history_on_init(self):
        param = Parameter(**self.SERIALIZED_PARAM)

        history = param.history()
        assert len(history) == 1
        assert history[0].value == 100
        assert history[0].source == "test"

    def test_value_added_to_history_on_being_set(self):
        param = Parameter(**self.SERIALIZED_PARAM)

        param.value = 200
        history = param.history()

        assert len(history) == 2
        assert history[1].value == 200
        assert history[1].source == ""

    def test_value_added_to_history_given_set_value_called(self):
        param = Parameter(**self.SERIALIZED_PARAM)

        param.set_value(150, "test_case")
        history = param.history()

        assert len(history) == 2
        assert history[1].value == 150
        assert history[1].source == "test_case"

    @pytest.mark.parametrize(
        ("arg", "value"),
        [
            ("name", 1),
            ("unit", 0),
            ("source", [100]),
            ("description", 0.5),
            ("long_name", ["a", "b"]),
        ],
    )
    def test_TypeError_on_init_given_arg_incorrect_type(self, arg, value):
        kwargs = copy.deepcopy(self.SERIALIZED_PARAM)
        kwargs[arg] = value

        with pytest.raises(TypeError):
            Parameter(**kwargs)

    def test_to_dict_returns_equal_dict_as_used_in_init(self):
        param = Parameter(**self.SERIALIZED_PARAM)

        output = param.to_dict()

        assert output == self.SERIALIZED_PARAM

    @pytest.mark.parametrize("value_type", [list, str])
    def test_TypeError_given_value_type_different_to_arg_type(self, value_type):
        with pytest.raises(TypeError):
            Parameter(**self.SERIALIZED_PARAM, _value_types=(value_type,))

    def test_int_values_converted_to_float_given_float_value_type(self):
        param = Parameter(**self.SERIALIZED_PARAM, _value_types=(float,))

        assert isinstance(param.value, float)

    def test_repr_contains_name_value_and_no_unit(self):
        param = Parameter(**self.SERIALIZED_PARAM)
        assert "(my_param=100 )" in repr(param)

    def test_repr_contains_name_value_and_unit(self):
        s_param = copy.deepcopy(self.SERIALIZED_PARAM)
        s_param["unit"] = "metre"
        param = Parameter(**s_param)
        assert "(my_param=100 m)" in repr(param)

    def test_value_as_conversion(self):
        s_param = copy.deepcopy(self.SERIALIZED_PARAM)
        param = Parameter(**s_param)

        with pytest.raises(ValueError):  # noqa: PT011
            param.value_as("W")

        s_param["unit"] = "metre"
        param = Parameter(**s_param)
        assert param.value_as("km") == 1e-3 * s_param["value"]

    def test_value_as_conversion_for_None(self):
        s_param = copy.deepcopy(self.SERIALIZED_PARAM)
        s_param["value"] = None
        param = Parameter(**s_param)

        with pytest.raises(ValueError):  # noqa: PT011
            param.value_as("W")

        s_param["unit"] = "metre"
        param = Parameter(**s_param)
        assert param.value_as("km") is None

    def test_value_as_conversion_for_bool(self):
        s_param = copy.deepcopy(self.SERIALIZED_PARAM)
        s_param["value"] = False

        param = Parameter(**s_param)
        with pytest.raises(TypeError):
            param.value_as("km")

    @pytest.mark.parametrize(
        ("param1", "param2"),
        [
            (
                {"name": "p", "value": 10, "unit": "dimensionless"},
                {"name": "p", "value": 10, "unit": "dimensionless"},
            ),
            (
                {"name": "p", "value": 1, "unit": "m"},
                {"name": "p", "value": 100, "unit": "cm"},
            ),
        ],
    )
    def test_params_with_same_name_and_values_are_equal(self, param1, param2):
        p1 = Parameter(**param1)
        p2 = Parameter(**param2)

        assert p1 == p2

    @pytest.mark.parametrize(
        ("param1", "param2"),
        [
            (
                {"name": "p", "value": 10, "unit": "m"},
                {"name": "p", "value": 10, "unit": "dimensionless"},
            ),
            (
                {"name": "p", "value": 100.5, "unit": "m"},
                {"name": "p", "value": 100, "unit": "m"},
            ),
        ],
    )
    def test_params_with_different_values_are_not_equal(self, param1, param2):
        p1 = Parameter(**param1)
        p2 = Parameter(**param2)

        assert p1 != p2

    def test_params_with_different_names_are_not_equal(self):
        p1 = Parameter(  # noqa: PIE804
            **{"name": "p1", "value": 10, "unit": "dimensionless"}
        )
        p2 = Parameter(  # noqa: PIE804
            **{"name": "p2", "value": 10, "unit": "dimensionless"}
        )

        assert p1 != p2
