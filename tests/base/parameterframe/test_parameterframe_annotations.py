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

from __future__ import annotations

from dataclasses import dataclass

from bluemira.base.parameter_frame import Parameter, ParameterFrame


@dataclass
class PFrame(ParameterFrame):
    a: Parameter[float]
    b: Parameter[int]


def test_future_annotations_are_typed():
    """
    The purpose of the test is to check we can still do type validation
    when we've imported annotations (because of delayed evaluations).
    """

    d = {
        "a": {"value": 3.14, "unit": ""},
        "b": {"value": 1, "unit": ""},
    }

    f = PFrame.from_dict(d)

    # importing annotations converts the typing to a string
    assert isinstance(PFrame.__annotations__["a"], str)
    assert not isinstance(f._types["a"], str)
