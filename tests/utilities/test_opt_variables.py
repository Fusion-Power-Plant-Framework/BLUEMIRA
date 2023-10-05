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

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from bluemira.base.constants import EPS
from bluemira.utilities.opt_variables import (
    OptVariable,
    OptVariablesError,
    OptVariablesFrame,
    ov,
)


class TestOptVariable:
    def test_initialisation(self):
        v1 = OptVariable("a", 2, 0, 3, description="test")
        assert v1.name == "a"
        assert v1.value == 2
        assert v1.lower_bound == 0
        assert v1.upper_bound == 3
        assert v1.description == "test"

        v2 = OptVariable("b", 0, -1, 1)
        assert v2.name == "b"
        assert v2.value == 0
        assert v2.lower_bound == -1
        assert v2.upper_bound == 1
        assert v2.description is None

        with pytest.raises(OptVariablesError):
            v3 = OptVariable("a", 2, 2.5, 3)

    def test_normalised_value(self):
        v1 = OptVariable("a", 2, 0, 4)
        assert v1.normalised_value == pytest.approx(0.5, rel=0, abs=EPS)

    def test_adjust(self):
        v1 = OptVariable("a", 2, 0, 4)
        v1.adjust(value=3, lower_bound=2, upper_bound=4)
        assert v1.value == 3
        assert v1.lower_bound == 2
        assert v1.upper_bound == 4

    def test_adjust_bad(self):
        v1 = OptVariable("a", 2, 0, 4)
        with pytest.raises(OptVariablesError):
            v1.adjust(value=0, lower_bound=1, upper_bound=2)
        with pytest.raises(OptVariablesError):
            v1.adjust(lower_bound=2, upper_bound=-1)
        with pytest.raises(OptVariablesError):
            v1.adjust(value=-2, lower_bound=-4, upper_bound=-3)

    def test_adjust_fixed(self):
        v1 = OptVariable("a", 2, 0, 4, fixed=True)
        assert v1.value == 2
        with pytest.raises(OptVariablesError):
            v1.adjust(value=3)
        with pytest.raises(OptVariablesError):
            v1.adjust(upper_bound=9)
        with pytest.raises(OptVariablesError):
            v1.value = 3


@dataclass
class TestOptVariablesOptVariables(OptVariablesFrame):
    __test__ = False

    a: OptVariable = ov("a", 2, 0, 3)
    b: OptVariable = ov("b", 0, -1, 1)
    c: OptVariable = ov("c", -1, -10, 10)


class TestOptVariables:
    def setup_method(self):
        self.vars = TestOptVariablesOptVariables()

    def test_init(self):
        assert self.vars.n_free_variables == 3
        assert len(self.vars.values) == 3
        assert np.allclose(self.vars.values, np.array([2, 0, -1]))

    def test_fix(self):
        self.vars.fix_variable("a", value=100)
        assert self.vars.values[0] == 100
        assert self.vars.n_free_variables == 2
        assert np.allclose(self.vars.values, np.array([100, 0, -1]))

    def test_getitem(self):
        assert self.vars["a"] == self.vars.a

    def test_adjust(self):
        self.vars.adjust_variable("b", fixed=True)
        assert self.vars.b.fixed

    def test_not_strict_bounds(self):
        self.vars.adjust_variables({"a": {"value": -2}}, strict_bounds=False)

        assert self.vars.a.value == -2
        assert self.vars.a.lower_bound == -2

    def test_not_strict_bounds2(self):
        self.vars.adjust_variables(
            {"a": {"value": -2, "lower_bound": -1, "upper_bound": 3}},
            strict_bounds=False,
        )
        assert self.vars.a.value == -2
        assert self.vars.a.lower_bound == -2

        with pytest.raises(OptVariablesError):
            self.vars.adjust_variables(
                {"a": {"value": -2, "lower_bound": -1, "upper_bound": -3}},
                strict_bounds=True,
            )

    def test_read_write(self):
        tempdir = tempfile.mkdtemp()
        try:
            the_path = Path(tempdir, "opt_var_test.json")
            self.vars.to_json(the_path)
            new_vars = TestOptVariablesOptVariables.from_json(the_path)
            assert new_vars.as_dict() == self.vars.as_dict()
        finally:
            shutil.rmtree(tempdir)

    def test_tabulate_displays_column_values(self):
        table = self.vars.tabulate()

        table_pattern = "\n".join(  # noqa: FLY002
            [
                "TestOptVariablesOptVariables",
                "╒════════╤═════════╤═══════════════╤═══════════════╤═════════╤═══════════════╕",
                "│ name   │   value │   lower_bound │   upper_bound │ fixed   │ description   │",
                "╞════════╪═════════╪═══════════════╪═══════════════╪═════════╪═══════════════╡",
                "│ a      │       2 │             0 │             3 │ False   │               │",
                "├────────┼─────────┼───────────────┼───────────────┼─────────┼───────────────┤",
                "│ b      │       0 │            -1 │             1 │ False   │               │",
                "├────────┼─────────┼───────────────┼───────────────┼─────────┼───────────────┤",
                "│ c      │      -1 │           -10 │            10 │ False   │               │",
                "╘════════╧═════════╧═══════════════╧═══════════════╧═════════╧═══════════════╛",
            ]
        )
        assert len(table.split("\n")) == 10
        assert table == table_pattern
