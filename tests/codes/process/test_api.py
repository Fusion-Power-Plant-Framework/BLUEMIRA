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

from pathlib import Path
from unittest.mock import patch

from bluemira.codes.process import api

PROCESS_OBS_VAR = {
    "ni": "ni wang",
    "ni wang": "ni peng",
    "garden": "shrubbery",
}


@patch("bluemira.codes.process.api.OBS_VARS", PROCESS_OBS_VAR)
def test_update_obsolete_vars():
    str1 = api.update_obsolete_vars("ni")
    str2 = api.update_obsolete_vars("garden")
    assert str1 == "ni peng"
    assert str2 == "shrubbery"


@patch.object(api, "imp_data")
def test_impurities(imp_data_mock):
    imp_data_mock.__file__ = "./__init__.py"
    assert api.Impurities["H"] == api.Impurities.H
    assert api.Impurities(1) == api.Impurities.H
    assert api.Impurities(1).id() == "fimp(01"
    assert api.Impurities(10).id() == "fimp(10"
    assert api.Impurities(1).file() == Path("./H_Lzdata.dat")
    assert api.Impurities(10).file() == Path("./FeLzdata.dat")


def test_INVariable_works_with_floats():
    inv = api._INVariable("name", 5, "a 'type'", "a 'group'", "a comment")

    assert inv.name == "name"
    assert inv._value == inv.get_value == 5
    assert inv.value == "5"

    inv.value = 6

    assert inv._value == inv.get_value == 6
    assert inv.value == "6"

    assert inv.v_type == "a 'type'"
    assert inv.parameter_group == "a 'group'"
    assert inv.comment == "a comment"


def test_INVariable_works_with_list_and_dict():
    inv = api._INVariable("name", [5], "a 'type'", "a 'group'", "a comment")

    assert inv.name == "name"
    assert inv._value == inv.value == inv.get_value == [5]

    inv.value = {"a": 6}

    assert inv._value == inv.value == inv.get_value == {"a": 6}

    assert inv.v_type == "a 'type'"
    assert inv.parameter_group == "a 'group'"
    assert inv.comment == "a comment"
