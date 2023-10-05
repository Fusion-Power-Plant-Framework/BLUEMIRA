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
import pytest

from bluemira.base.designer import Designer


class SimpleDesigner(Designer):
    param_cls = None

    def run(self) -> int:
        return 10

    def mock(self) -> int:
        return 11

    def read(self) -> int:
        return 12

    def custom_run_mode(self) -> int:
        return 13


class TestDesigner:
    def test_execute_calls_run_if_no_run_mode_in_build_config(self):
        designer = SimpleDesigner(None, {})

        assert designer.execute() == 10

    @pytest.mark.parametrize(
        ("mode", "output"),
        [("run", 10), ("mock", 11), ("read", 12), ("custom_run_mode", 13)],
    )
    def test_execute_calls_function_given_by_run_mode(self, mode, output):
        designer = SimpleDesigner(None, {"run_mode": mode})

        assert designer.execute() == output

    def test_ValueError_on_init_given_unknown_run_mode(self):
        with pytest.raises(ValueError):  # noqa: PT011
            SimpleDesigner(None, {"run_mode": "not_a_mode"}).execute()
