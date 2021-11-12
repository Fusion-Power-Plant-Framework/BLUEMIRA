# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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

"""
Tests for utilities for external code integration
"""

import pytest

from bluemira.base.parameter import ParameterFrame, ParameterMapping

from bluemira.codes.utilities import get_recv_mapping, get_send_mapping


class TestMappings:
    params = ParameterFrame(
        [
            [
                "a_param",
                "_a_param",
                1.0,
                "m",
                None,
                "Input",
                {"codeA": ParameterMapping("aParam", recv=True, send=False)},
            ],
            [
                "the_param",
                "_the_param",
                2.0,
                "m",
                None,
                "Input",
                {"codeB": ParameterMapping("ParamB", recv=False, send=True)},
            ],
            [
                "other_param",
                "_other_param",
                2.0,
                "m",
                None,
                "Input",
                {"CodeC": ParameterMapping("PaRaM", recv=True, send=True)},
            ],
            [
                "another_param",
                "_another_param",
                None,
                "N/A",
                None,
                "Input",
                {"CodeC": ParameterMapping("pArAm", recv=True, send=True)},
            ],
            [
                "last_param",
                "_last_param",
                2.0,
                "m",
                None,
                "Input",
                {"codeB": ParameterMapping("ParamD", recv=False, send=False)},
            ],
        ]
    )

    @pytest.mark.parametrize(
        "code,recv_all,expected",
        [
            ("codeA", False, {"aParam": "a_param"}),
            ("codeA", True, {"aParam": "a_param"}),
            ("codeB", False, {}),
            ("codeB", True, {"ParamB": "the_param", "ParamD": "last_param"}),
            ("CodeC", False, {"PaRaM": "other_param", "pArAm": "another_param"}),
            ("CodeC", True, {"PaRaM": "other_param", "pArAm": "another_param"}),
        ],
    )
    def test_get_recv_mapping(self, code, recv_all, expected):
        mapping = get_recv_mapping(self.params, code, recv_all=recv_all)
        assert mapping == expected

    @pytest.mark.parametrize(
        "code,send_all,expected",
        [
            ("codeA", False, {}),
            ("codeA", True, {"aParam": "a_param"}),
            ("codeB", False, {"ParamB": "the_param"}),
            ("codeB", True, {"ParamB": "the_param", "ParamD": "last_param"}),
            ("CodeC", False, {"PaRaM": "other_param", "pArAm": "another_param"}),
            ("CodeC", True, {"PaRaM": "other_param", "pArAm": "another_param"}),
        ],
    )
    def test_get_send_mapping(self, code, send_all, expected):
        mapping = get_send_mapping(self.params, code, send_all=send_all)
        assert mapping == expected
