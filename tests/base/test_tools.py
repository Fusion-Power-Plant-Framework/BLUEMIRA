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

import logging

import pytest

from bluemira.base.tools import timing


def dummy(a, *, b=4):
    return (a, b)


@pytest.mark.parametrize(
    "print_name, records", [[True, ["INFO", "DEBUG"]], [False, ["DEBUG", "DEBUG"]]]
)
def test_timing(print_name, records, caplog):
    caplog.set_level(logging.INFO)
    assert timing(dummy, "debug", "print", print_name=print_name)(1, b=2) == (1, 2)
    assert len(caplog.records) == 2
    assert [r.levelname for r in caplog.records] == records
    for msg, exp in zip(caplog.messages, ("print", "debug")):
        assert exp in msg
