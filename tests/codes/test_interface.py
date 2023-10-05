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
from typing import ClassVar

from bluemira.base.parameter_frame import Parameter
from bluemira.codes import interface
from bluemira.codes.interface import BaseRunMode, NoOpTask
from bluemira.codes.params import MappedParameterFrame, ParameterMapping


class NoOpRunMode(BaseRunMode):
    RUN = 0


class NoOpSolver(interface.CodesSolver):
    name = "MyTestSolver"
    setup_cls = NoOpTask
    run_cls = NoOpTask
    teardown_cls = NoOpTask
    run_mode_cls = NoOpRunMode


@dataclass
class Params(MappedParameterFrame):
    param1: Parameter[float]
    param2: Parameter[int]

    _mappings: ClassVar = {
        "param1": ParameterMapping("ext1", send=True, recv=True, unit="MW"),
        "param2": ParameterMapping("ext2", send=False, recv=False),
    }

    @property
    def mappings(self):
        return self._mappings

    @classmethod
    def from_defaults(cls):
        return super().from_defaults({})


class TestCodesSolver:
    def test_modify_mappings_updates_send_recv_values_of_params(self):
        params = Params.from_dict(
            {
                "param1": {"value": 0.1, "unit": "m"},
                "param2": {"value": 5, "unit": "dimensionless"},
            }
        )
        solver = NoOpSolver(params)

        solver.modify_mappings(
            {
                "param1": {"recv": True, "send": False},
                "param2": {"recv": False, "send": True},
            }
        )

        assert solver.params.mappings["param1"].send is False
        assert solver.params.mappings["param1"].recv is True
        assert solver.params.mappings["param2"].send is True
        assert solver.params.mappings["param2"].recv is False

    def test_no_defaults_are_set_to_None(self):
        params = Params.from_defaults()
        assert params.param1.value is None
        assert params.param2.value is None
        assert params.param1.unit == "W"
        assert params.param2.unit == ""
