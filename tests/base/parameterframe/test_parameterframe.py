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
import io
from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar, Union
from unittest import mock

import pint
import pytest

from bluemira.base.constants import EPS
from bluemira.base.parameter_frame import Parameter, ParameterFrame, make_parameter_frame
from bluemira.base.parameter_frame._parameter import ParamDictT

FRAME_DATA = {
    "height": {"value": 180.5, "unit": "cm"},
    "age": {"value": 30, "unit": "years"},
}


@dataclass
class BasicFrame(ParameterFrame):
    height: Parameter[float]
    age: Parameter[int]


class NoDataclassFrame(ParameterFrame):
    height: Parameter[float]
    age: Parameter[int]


@dataclass
class BrokenFrame(ParameterFrame):
    height: float


class TestParameterFrame:
    def setup_method(self):
        self.frame = BasicFrame.from_dict(deepcopy(FRAME_DATA))

    def test_frame_defined_with_dataclass_annotation(self, caplog):
        NoDataclassFrame()
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "WARNING"

    def test_init_from_dict_sets_valid_entries(self):
        assert self.frame.height.value == pytest.approx(1.805, rel=0, abs=EPS)
        assert self.frame.height.unit == "m"
        assert (
            self.frame.age.value
            == pint.Quantity(FRAME_DATA["age"]["value"], FRAME_DATA["age"]["unit"])
            .to("s")
            .magnitude
        )
        assert self.frame.age.unit == "s"

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("name", 100),
            ("value", "wrong type"),
            ("value", 30.532423),
            ("unit", 0.5),
        ],
    )
    def test_from_dict_TypeError_given_invalid_type(self, name, value):
        data = {
            "height": {"value": 180.5, "unit": "cm"},
            "age": {"value": 30, "unit": "years"},
        }
        data["age"][name] = value

        with pytest.raises(TypeError):
            BasicFrame.from_dict(data)

    def test_TypeError_raise_for_non_parameter_type(self):
        frame_data = {"height": FRAME_DATA["height"]}
        with pytest.raises(TypeError):
            BrokenFrame.from_dict(frame_data)

    def test_from_dict_ValueError_given_unknown_parameter(self):
        data = {
            "height": {"value": 180.5, "unit": "m"},
            "age": {"value": 30, "unit": "years"},
            "weight": {"value": 60, "unit": "kg"},
        }

        with pytest.raises(ValueError):  # noqa: PT011
            BasicFrame.from_dict(data)

    def test_from_dict_ValueError_given_missing_parameter(self):
        data = {"height": {"value": 180.5, "unit": "m"}}

        with pytest.raises(ValueError):  # noqa: PT011
            BasicFrame.from_dict(data)

    def test_to_dict_returns_input_to_from_dict(self):
        frame_data = deepcopy(FRAME_DATA)
        frame_data["height"]["unit"] = "m"
        frame_data["age"]["unit"] = "s"
        frame = BasicFrame.from_dict(frame_data)
        out_dict = frame.to_dict()

        assert out_dict == frame_data

    def test_initialising_using_untyped_generic_parameter_is_allowed(self):
        @dataclass
        class GenericFrame(ParameterFrame):
            x: Parameter

        frame = GenericFrame.from_dict({"x": {"value": 10, "unit": "m"}})

        assert frame.x.value == 10

    @pytest.mark.parametrize("value", ["OK", ["OK"]])
    def test_TypeError_given_field_has_Union_Parameter_type(self, value):
        @dataclass
        class GenericFrame(ParameterFrame):
            x: Parameter[Union[str, list]]

        with pytest.raises(TypeError):
            GenericFrame.from_dict({"x": {"value": value, "unit": "m"}})

    def test_TypeError_given_field_does_not_have_Parameter_type(self):
        @dataclass
        class BadFrame(ParameterFrame):
            x: int

        with pytest.raises(TypeError):
            BadFrame.from_dict({"x": {"value": 10}})

    def test_a_default_frame_is_empty(self):
        @dataclass
        class GenericFrame(ParameterFrame):
            ...

        assert len(GenericFrame().to_dict()) == 0

    def test_from_json_reads_json_string(self):
        json_str = """{
            "height": {"value": 180.5, "unit": "cm"},
            "age": {"value": 30, "unit": "years"}
        }"""

        frame = BasicFrame.from_json(json_str)

        assert frame.height.value == pytest.approx(1.805, rel=0, abs=EPS)
        assert frame.height.unit == "m"
        assert frame.age.value == pint.Quantity(30, "year").to("s").magnitude
        assert frame.age.unit == "s"

    def test_from_json_reads_json_io(self):
        json_str = """{
            "height": {"value": 180.5, "unit": "cm"},
            "age": {"value": 30, "unit": "years"}
        }"""
        json_io = io.StringIO(json_str)

        frame = BasicFrame.from_json(json_io)

        assert frame.height.value == pytest.approx(1.805, rel=0, abs=EPS)
        assert frame.height.unit == "m"
        assert frame.age.value == pint.Quantity(30, "year").to("s").magnitude
        assert frame.age.unit == "s"

    def test_from_json_reads_from_file(self):
        json_str = """{
            "height": {"value": 180.5, "unit": "cm"},
            "age": {"value": 30, "unit": "years"}
        }"""
        with mock.patch(
            "builtins.open", new_callable=mock.mock_open, read_data=json_str
        ):
            frame = BasicFrame.from_json("./some/path")

        assert frame.height.value == pytest.approx(1.805, rel=0, abs=EPS)
        assert frame.height.unit == "m"
        assert frame.age.value == pint.Quantity(30, "year").to("s").magnitude
        assert frame.age.unit == "s"

    def test_parameter_frames_with_eq_parameters_are_equal(self):
        assert self.frame == BasicFrame.from_dict(FRAME_DATA)

    def test_parameter_frames_with_different_parameters_are_not_equal(self):
        @dataclass
        class OtherFrame(ParameterFrame):
            height: Parameter[float]
            age: Parameter[int]
            weight: Parameter[float]

        frame2 = OtherFrame.from_dict(
            {**FRAME_DATA, "weight": {"value": 58.2, "unit": "kg"}}
        )

        assert self.frame != frame2

    @pytest.mark.parametrize("x", [1, "str", Parameter("x", 0.1, "m")])
    def test_frame_ne_to_non_frame(self, x):
        assert self.frame != x
        assert x != self.frame

    def test_update_values_edits_frames_values(self):
        self.frame.update_values({"height": 160.4}, source="a test")

        assert self.frame.height.value == pytest.approx(160.4, rel=0, abs=EPS)
        assert self.frame.height.source == "a test"
        assert (
            self.frame.age.value
            == pint.Quantity(FRAME_DATA["age"]["value"], FRAME_DATA["age"]["unit"])
            .to("s")
            .magnitude
        )
        assert self.frame.age.source != "a test"

    def test_update_using_values_edits_frames_values(self):
        self.frame.update({"height": 160.4})

        assert self.frame.height.value == pytest.approx(160.4, rel=0, abs=EPS)
        assert (
            self.frame.age.value
            == pint.Quantity(FRAME_DATA["age"]["value"], FRAME_DATA["age"]["unit"])
            .to("s")
            .magnitude
        )

    @pytest.mark.parametrize("func", ["update_from_dict", "update"])
    def test_update_from_dict(self, func):
        getattr(self.frame, func)(
            {
                "height": {
                    "name": "height",
                    "value": 160.4,
                    "unit": "m",
                    "source": "a test",
                },
                "age": {"value": 20, "unit": "years"},
            }
        )

        assert self.frame.height.value == pytest.approx(160.4, rel=0, abs=EPS)
        assert self.frame.height.source == "a test"
        assert self.frame.age.value == pint.Quantity(20, "years").to("s").magnitude
        assert self.frame.age.source != "a test"

    @pytest.mark.parametrize("func", ["update_from_frame", "update"])
    def test_update_from_frame(self, func):
        update_frame = BasicFrame.from_dict(
            {
                "height": {
                    "value": 160.4,
                    "unit": "m",
                    "source": "a test",
                },
                "age": {"value": 20, "unit": "years"},
            }
        )
        getattr(self.frame, func)(update_frame)

        assert self.frame.height.value == pytest.approx(160.4, rel=0, abs=EPS)
        assert self.frame.height.source == "a test"
        assert self.frame.age.value == pint.Quantity(20, "years").to("s").magnitude
        assert self.frame.age.source != "a test"

    @pytest.mark.parametrize("func", ["update_from_frame", "update"])
    def test_update_from_frame_with_None(self, func):
        update_frame = BasicFrame.from_dict(
            {
                "height": {
                    "value": 160.4,
                    "unit": "m",
                    "source": "a test",
                },
                "age": {"value": None, "unit": "years"},
            }
        )
        getattr(self.frame, func)(update_frame)

        assert self.frame.height.value == pytest.approx(160.4, rel=0, abs=EPS)
        assert self.frame.height.source == "a test"
        assert self.frame.age.value is None
        assert self.frame.age.source != "a test"

    @pytest.mark.parametrize("func", ["update_from_dict", "update"])
    def test_update_from_dict_with_None(self, func):
        getattr(self.frame, func)(
            {
                "height": {
                    "name": "height",
                    "value": 160.4,
                    "unit": "m",
                    "source": "a test",
                },
                "age": {"value": None, "unit": "years"},
            }
        )

        assert self.frame.height.value == pytest.approx(160.4, rel=0, abs=EPS)
        assert self.frame.height.source == "a test"
        assert self.frame.age.value is None
        assert self.frame.age.source != "a test"

    def test_get_values(self):
        assert self.frame.get_values("height", "age") == (
            self.frame.height.value,
            self.frame.age.value,
        )
        assert self.frame.height.value_as("cm") == pytest.approx(
            FRAME_DATA["height"]["value"]
        )
        assert self.frame.age.value_as("yr") == pytest.approx(FRAME_DATA["age"]["value"])

    def test_get_values_raises_AttributeError(self):
        with pytest.raises(AttributeError) as ae:
            self.frame.get_values("I dont", "exist", "height")

        assert "['I dont', 'exist']" in ae.value.args[0]

    def _call_tabulate(self, head_keys):
        frame_data = deepcopy(FRAME_DATA)
        frame_data["height"]["unit"] = "m"
        frame_data["age"]["unit"] = "s"
        with mock.patch("bluemira.base.parameter_frame._frame.tabulate") as m_tb:
            BasicFrame.from_dict(frame_data).tabulate(keys=head_keys)

        (table_rows,), call_kwargs = m_tb.call_args

        return call_kwargs["headers"], table_rows, frame_data

    @pytest.mark.parametrize(
        ("head_keys", "result"),
        zip(
            [None, ["name", "value"]],
            [ParamDictT.__annotations__.keys(), ["name", "value"]],
        ),
    )
    def test_tabulate_headers(self, head_keys, result):
        headers, *_ = self._call_tabulate(head_keys)
        assert set(headers) == set(result)

    def _get_data_keys_and_values(self, frame_data, head_keys):
        # The columns and rows of the parameterframe are sorted
        data_keys = sorted(frame_data.keys())

        if head_keys is not None:
            fd_keys_list = list(set(head_keys) - set(frame_data.keys()))
        else:
            fd_keys_list = list(frame_data.keys())

        data_values = list(frame_data.values())
        data_values_index = sorted(
            range(len(fd_keys_list)), key=fd_keys_list.__getitem__
        )

        return data_keys, data_values, data_values_index

    @pytest.mark.parametrize("head_keys", [None, ["name", "value"]])
    def test_tabulate_method_columns_have_correct_num_of_NA(self, head_keys):
        # Number of 'N/A' equal to headers without name - number of filled keys

        nn_headers, table_rows, frame_data = self._call_tabulate(head_keys)
        nn_headers.pop(nn_headers.index("name"))

        _, data_values, data_values_index = self._get_data_keys_and_values(
            frame_data, head_keys
        )

        for tr, dvi in zip(table_rows, data_values_index):
            assert len([i for i, x in enumerate(tr) if x == "N/A"]) == len(
                nn_headers - data_values[dvi].keys()
            )

    @pytest.mark.parametrize("head_keys", [None, ["name", "value"]])
    def test_tabulate_method_columns_have_correct_data(self, head_keys):
        headers, table_rows, frame_data = self._call_tabulate(head_keys)

        (
            data_keys,
            data_values,
            data_values_index,
        ) = self._get_data_keys_and_values(frame_data, head_keys)

        for no, (tr, dvi) in enumerate(zip(table_rows, data_values_index)):
            # name is correct
            assert tr[0] == data_keys[no]
            for ind in data_values[dvi]:
                try:
                    assert tr[headers.index(ind)] == frame_data[data_keys[no]][ind]
                except ValueError as ve:  # noqa: PERF203
                    if ind in head_keys:
                        raise
                except AssertionError as ae:
                    if ind == "value":
                        assert (
                            tr[headers.index(ind)]
                            == f"{frame_data[data_keys[no]][ind]: 5g}"
                        )
                    else:
                        raise

    def test_iterating_returns_parameters_in_declaration_order(self):
        frame = BasicFrame.from_dict(FRAME_DATA)
        params = list(frame)

        assert all(isinstance(p, Parameter) for p in params)
        assert [p.name for p in params] == ["height", "age"]

    def test_ValueError_creating_frame_from_non_superset_frame(self):
        @dataclass
        class OtherFrame(ParameterFrame):
            height: Parameter[float]
            age: Parameter[int]
            weight: Parameter[float]

        basic_frame = BasicFrame.from_dict(FRAME_DATA)

        with pytest.raises(ValueError):  # noqa: PT011
            OtherFrame.from_frame(basic_frame)

    def test_from_json_TypeError_given_non_string_or_buffer(self):
        with pytest.raises(TypeError) as error:
            BasicFrame.from_json(["x"])
        assert "Cannot read JSON" in str(error)

    def test_init_of_ParameterFrame_raises_TypeError(self):
        with pytest.raises(TypeError):
            ParameterFrame()

    def test_init_with_non_Parameters_raises_TypeError(self):
        @dataclass
        class GenericFrame(ParameterFrame):
            A: float

        with pytest.raises(TypeError):
            GenericFrame(5)

    def test_definition_with_non_Parameters_raises_TypeError(self):
        @dataclass
        class GenericFrame(ParameterFrame):
            A: float

        with pytest.raises(TypeError):
            GenericFrame(Parameter("A", 5))

    def test_init_Parameter_has_wrong_name_raises_TypeError(self):
        @dataclass
        class GenericFrame(ParameterFrame):
            A: Parameter[float]

        with pytest.raises(TypeError):
            GenericFrame(Parameter("B", 5))

    def test_init_SI_unit_enforcement(self):
        @dataclass
        class GenericFrame(ParameterFrame):
            A: Parameter[float]

        frm = GenericFrame(Parameter("A", 5, "cm"))

        assert frm.A.unit == "m"
        assert frm.A.value == pytest.approx(0.05)
        assert frm.A.source == "unit enforcement"

        frm = GenericFrame(Parameter("A", 5, "m"))
        assert frm.A.unit == "m"
        assert frm.A.value == pytest.approx(5)
        assert frm.A.source == ""


@dataclass
class UnitFrame1(ParameterFrame):
    length: Parameter[float]
    time: Parameter[int]
    mass: Parameter[int]
    flag: Parameter[bool]
    string: Parameter[str]


@dataclass
class UnitFrame2(ParameterFrame):
    magfield: Parameter[float]
    energy: Parameter[float]
    pressure: Parameter[int]


@dataclass
class UnitFrame3(ParameterFrame):
    damagepertime: Parameter[float]
    timeperdamage: Parameter[float]
    damage: Parameter[float]
    perdamage: Parameter[float]
    on_time: Parameter[float]
    per_on_time: Parameter[float]


@dataclass
class UnitFrame4(ParameterFrame):
    angle: Parameter[float]
    angle2: Parameter[float]
    angleperthing: Parameter[float]
    thingperangle: Parameter[float]


@dataclass
class UnitFrame5(ParameterFrame):
    wtf1: Parameter[float]
    wtf2: Parameter[float]
    wtf3: Parameter[float]


class TestParameterFrameUnits:
    SIMPLE_FRAME_DATA: ClassVar = {
        "length": {"value": 180.5, "unit": "in"},
        "time": {"value": 30, "unit": "day"},
        "mass": {"value": 1, "unit": "tonne"},
        "flag": {"value": False, "unit": ""},
        "string": {"value": "Hello 👋", "unit": ""},
    }

    COMPLEX_FRAME_DATA: ClassVar = {
        "magfield": {"value": 5000, "unit": "gamma"},
        "energy": {"value": 30, "unit": "keV"},
        "pressure": {"value": 1, "unit": "atm"},
    }

    WEIRD_FRAME_DATA: ClassVar = {
        "damagepertime": {"value": 30, "unit": "dpa/fpy"},
        "timeperdamage": {"value": 30, "unit": "fpy/dpa"},
        "damage": {"value": 30, "unit": "dpa"},
        "perdamage": {"value": 30, "unit": "1/dpa"},
        "on_time": {"value": 1, "unit": "fpy"},
        "per_on_time": {"value": 1, "unit": "1/fpy"},
    }

    ANGLE_FRAME_DATA: ClassVar = {
        "angle": {"value": 5, "unit": "radian"},
        "angle2": {"value": 5, "unit": "grade"},
        "angleperthing": {"value": 5, "unit": "radian/m"},
        "thingperangle": {"value": 5, "unit": "W/turn"},
    }

    WTF_FRAME_DATA: ClassVar = {
        "wtf1": {"value": 5, "unit": "m^2/grade.W/(Pa.fpy)"},
        "wtf2": {"value": 5, "unit": "dpa.m^2/rad.W/(Pa.fpy)"},
        "wtf3": {"value": 5, "unit": "dpa^-1.m^2/turn.W/(Pa.fpy)"},
    }

    def test_simple_units_to_defaults(self):
        frame = UnitFrame1.from_dict(self.SIMPLE_FRAME_DATA)
        assert frame.length.unit == "m"
        assert frame.length.value == 4.5847
        assert frame.time.unit == "s"
        assert frame.time.value == 2592000
        assert frame.mass.unit == "kg"
        assert frame.mass.value == 1000
        assert frame.flag.value is False
        assert frame.flag.unit == ""
        assert frame.string.value == "Hello 👋"
        assert frame.string.unit == ""

    def test_complex_units_to_defaults(self):
        frame = UnitFrame2.from_dict(self.COMPLEX_FRAME_DATA)
        assert frame.magfield.unit == "T"
        assert frame.magfield.value == pytest.approx(5e-6)
        assert frame.energy.unit == "J"
        assert frame.energy.value == pytest.approx(4.8065299e-15)
        assert frame.pressure.unit == "Pa"
        assert frame.pressure.value == 101325

    def test_weird_units_to_defaults_dpa_fpy(self):
        frame = UnitFrame3.from_dict(self.WEIRD_FRAME_DATA)
        assert frame.damage.value == 30
        assert frame.damage.unit == "dpa"
        assert frame.on_time.value == 1
        assert frame.on_time.unit == "fpy"

        assert frame.perdamage.value == 30
        assert frame.perdamage.unit == "1/dpa"
        assert frame.per_on_time.value == 1
        assert frame.per_on_time.unit == "1/fpy"
        assert frame.damagepertime.value == 30
        assert frame.damagepertime.unit == "dpa/fpy"
        assert frame.timeperdamage.value == 30
        assert frame.timeperdamage.unit == "fpy/dpa"

    def test_angle_units_to_defaults(self):
        frame = UnitFrame4.from_dict(self.ANGLE_FRAME_DATA)
        assert frame.angle.value == pytest.approx(286.4789)
        assert frame.angle.unit == "deg"
        assert frame.angle2.value == pytest.approx(4.5)
        assert frame.angle2.unit == "deg"
        assert frame.angleperthing.value == pytest.approx(286.4789)
        assert frame.angleperthing.unit == "deg/m"
        assert frame.thingperangle.value == pytest.approx(0.0138888888)
        assert frame.thingperangle.unit == "W/deg"

    def test_wtf_units(self):
        frame = UnitFrame5.from_dict(self.WTF_FRAME_DATA)
        assert frame.wtf1.value == pytest.approx(5.555555)
        assert frame.wtf1.unit == "m⁵/deg/fpy/s"
        assert frame.wtf2.value == pytest.approx(0.0872664)
        assert frame.wtf2.unit == "dpa·m⁵/deg/fpy/s"
        assert frame.wtf3.value == pytest.approx(0.01388888)
        assert frame.wtf3.unit == "m⁵/deg/dpa/fpy/s"

    def test_bad_unit(self):
        frame_data = deepcopy(self.SIMPLE_FRAME_DATA)
        frame_data["length"]["unit"] = "I_am_a_unit"
        with pytest.raises(ValueError):  # noqa: PT011
            frame = UnitFrame1.from_dict(frame_data)


class TestParameterSetup:
    def test_params_None(self):
        with pytest.raises(ValueError):  # noqa: PT011
            make_parameter_frame(FRAME_DATA, None)
        with pytest.raises(TypeError):
            make_parameter_frame(None, BasicFrame)
        params = make_parameter_frame(None, None)
        assert params is None

    @pytest.mark.parametrize(
        "frame",
        [
            FRAME_DATA,
            BasicFrame.from_dict(FRAME_DATA),
        ],
    )
    def test_params_type(self, frame):
        params = make_parameter_frame(frame, BasicFrame)
        assert isinstance(params, BasicFrame)


def test_changes_to_parameters_are_propagated_between_frames():
    """
    This tests a key mechanic that updates to parameters in a frame
    made from some base frame, are propagated to the base frame.
    This allows builders/designers to update parameters at the reactor
    level, whilst still only working with their specific frame.
    """

    @dataclass
    class SubFrame(ParameterFrame):
        height: Parameter[float]

    base_frame = BasicFrame.from_dict(FRAME_DATA)
    slim_frame = SubFrame.from_frame(base_frame)

    slim_frame.height.value = 200.5

    assert base_frame.height.value == 200.5
