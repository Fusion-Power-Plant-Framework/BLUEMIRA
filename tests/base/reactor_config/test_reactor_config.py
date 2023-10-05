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

from dataclasses import dataclass
from pathlib import Path

import pytest

from bluemira.base.constants import EPS, raw_uc
from bluemira.base.error import ReactorConfigError
from bluemira.base.logs import get_log_level, set_log_level
from bluemira.base.parameter_frame import (
    EmptyFrame,
    Parameter,
    ParameterFrame,
    make_parameter_frame,
)
from bluemira.base.reactor_config import ReactorConfig


@dataclass
class TestGlobalParams(ParameterFrame):
    __test__ = False

    only_global: Parameter[int]
    height: Parameter[float]
    age: Parameter[int]
    extra_global: Parameter[int]


@dataclass
class TestCompADesignerParams(ParameterFrame):
    __test__ = False

    only_global: Parameter[int]
    height: Parameter[float]
    age: Parameter[int]
    name: Parameter[str]
    location: Parameter[str]


test_config_path = Path(__file__).parent / "data" / "reactor_config.test.json"
empty_config_path = Path(__file__).parent / "data" / "reactor_config.empty.json"
nested_config_path = Path(__file__).parent / "data" / "reactor_config.nested_config.json"
nested_params_config_path = (
    Path(__file__).parent / "data" / "reactor_config.nested_params.json"
)
nesting_config_path = Path(__file__).parent / "data" / "reactor_config.nesting.json"


class TestReactorConfigClass:
    """
    Tests for the Reactor Config class functionality.
    """

    def setup_method(self):
        self.old_log_level = get_log_level()
        set_log_level("DEBUG")

    def teardown_method(self):
        set_log_level(self.old_log_level)

    def test_file_loading_with_empty_config(self, caplog):
        reactor_config = ReactorConfig(empty_config_path, EmptyFrame)

        # want to know explicitly if it is an EmptyFrame
        assert type(reactor_config.global_params) is EmptyFrame

        p_dne = reactor_config.params_for("dne")
        c_dne = reactor_config.config_for("dne")

        assert len(caplog.records) == 2
        for record in caplog.records:
            assert record.levelname == "DEBUG"

        assert len(p_dne.local_params) == 0
        assert len(c_dne) == 0

    def test_incorrect_global_config_type_empty_config(self):
        with pytest.raises(ValueError):  # noqa: PT011
            ReactorConfig(empty_config_path, TestGlobalParams)

    def test_incorrect_global_config_type_non_empty_config(self):
        with pytest.raises(ValueError):  # noqa: PT011
            ReactorConfig(test_config_path, EmptyFrame)

    def test_throw_on_too_specific_arg(self):
        reactor_config = ReactorConfig(test_config_path, TestGlobalParams)

        with pytest.raises(ReactorConfigError):
            reactor_config.config_for("comp A", "config_a", "a_value")

    def test_set_global_params(self, caplog):
        reactor_config = ReactorConfig(test_config_path, TestGlobalParams)

        cp = reactor_config.params_for("comp A", "designer")

        assert len(caplog.records) == 1
        for record in caplog.records:
            assert record.levelname == "DEBUG"

        cpf = make_parameter_frame(cp, TestCompADesignerParams)

        # instance checks
        assert cpf.only_global is reactor_config.global_params.only_global
        assert cpf.height is reactor_config.global_params.height
        assert cpf.age is reactor_config.global_params.age

        self._compa_designer_param_value_checks(cpf)

        cpf.only_global.value = raw_uc(2, "years", "s")
        assert cpf.only_global.value == raw_uc(2, "years", "s")
        assert reactor_config.global_params.only_global.value == raw_uc(2, "years", "s")
        assert cpf.only_global is reactor_config.global_params.only_global

    def _compa_designer_param_value_checks(self, cpf):
        assert cpf.only_global.value == raw_uc(1, "years", "s")
        assert cpf.height.value == pytest.approx(1.8, rel=0, abs=EPS)
        assert cpf.age.value == raw_uc(30, "years", "s")
        assert cpf.name.value == "Comp A"
        assert cpf.location.value == "here"

    def test_params_for_warnings_make_param_frame_type_value_overrides(self, caplog):
        reactor_config = ReactorConfig(
            test_config_path,
            TestGlobalParams,
        )

        cp = reactor_config.params_for("comp A", "designer")

        assert len(caplog.records) == 1
        for record in caplog.records:
            assert record.levelname == "DEBUG"

        cpf = make_parameter_frame(cp, TestCompADesignerParams)

        self._compa_designer_param_value_checks(cpf)

        # instance checks
        assert cpf.only_global is reactor_config.global_params.only_global
        assert cpf.height is reactor_config.global_params.height
        assert cpf.age is reactor_config.global_params.age

    def test_config_for_warnings_value_overrides(self, caplog):
        reactor_config = ReactorConfig(
            test_config_path,
            TestGlobalParams,
        )

        cf_comp_a = reactor_config.config_for("comp A")
        cf_comp_a_des = reactor_config.config_for("comp A", "designer")

        assert len(caplog.records) == 1
        for record in caplog.records:
            assert record.levelname == "DEBUG"

        assert cf_comp_a["config_a"] == cf_comp_a_des["config_a"]
        assert cf_comp_a["config_b"] == cf_comp_a_des["config_b"]
        assert cf_comp_a_des["config_c"]["c_value"] == "c_value"

    def test_no_params_warning(self, caplog):
        reactor_config = ReactorConfig(
            {
                "comp A": {
                    "designer": {},
                },
            },
            EmptyFrame,
        )

        cp = reactor_config.params_for("comp A", "designer")
        cp_dne = reactor_config.params_for("comp A", "designer", "dne")

        assert len(caplog.records) == 2
        for record in caplog.records:
            assert record.levelname == "DEBUG"

        assert len(cp.local_params) == 0
        assert len(cp_dne.local_params) == 0

    def test_no_config_warning(self, caplog):
        reactor_config = ReactorConfig(
            {
                "comp A": {
                    "designer": {},
                },
            },
            EmptyFrame,
        )

        cf_comp_a = reactor_config.config_for("comp A")
        cf_comp_a_des = reactor_config.config_for("comp A", "designer")
        cf_comp_a_des_dne = reactor_config.config_for("comp A", "designer", "dne")

        assert len(caplog.records) == 2
        for record in caplog.records:
            assert record.levelname == "DEBUG"

        assert len(cf_comp_a) == 1
        assert len(cf_comp_a_des) == 0
        assert len(cf_comp_a_des_dne) == 0

    def test_invalid_rc_initialization(self):
        with pytest.raises(ReactorConfigError):
            ReactorConfig(
                ["wrong"],
                EmptyFrame,
            )

    def test_args_arent_str(self):
        reactor_config = ReactorConfig(
            {
                "comp A": {
                    "designer": {},
                },
            },
            EmptyFrame,
        )

        with pytest.raises(ReactorConfigError):
            reactor_config.config_for("comp A", 1)

    def test_file_path_loading_in_json_nested_params(self):
        out_dict = {
            "height": {"value": 1.8, "unit": "m"},
            "age": {"value": 946728000, "unit": "s"},
            "only_global": {"value": 31557600, "unit": "s"},
            "extra_global": {"value": 1, "unit": "s"},
        }
        reactor_config = ReactorConfig(nested_params_config_path, EmptyFrame)
        pf = make_parameter_frame(reactor_config.params_for("Tester"), TestGlobalParams)
        assert pf == TestGlobalParams.from_dict(out_dict)

    def test_file_path_loading_in_json_nested_config(self):
        reactor_config = ReactorConfig(nested_config_path, EmptyFrame)

        pf = make_parameter_frame(
            reactor_config.params_for("Tester", "comp A", "designer"),
            TestCompADesignerParams,
        )

        self._compa_designer_param_value_checks(pf)

        compa_designer_config = reactor_config.config_for("Tester", "comp A", "designer")
        assert compa_designer_config["config_a"] == {"a_value": "overridden_value"}
        assert compa_designer_config["config_b"] == {"b_value": "b_value"}
        assert compa_designer_config["config_c"] == {"c_value": "c_value"}

    def test_deeply_nested_files(self):
        reactor_config = ReactorConfig(nesting_config_path, EmptyFrame)

        assert reactor_config.config_for("nest_a")["a_val"] == "nest_a"
        assert reactor_config.config_for("nest_b")["a_val"] == "nest_b"

        assert reactor_config.params_for("nest_a").local_params["a_param"] == "nest_a"
        assert reactor_config.params_for("nest_b").local_params["a_param"] == "nest_b"
