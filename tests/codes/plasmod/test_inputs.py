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

import pytest

from bluemira.codes.error import CodesError
from bluemira.codes.plasmod.api._inputs import MODEL_MAP, PlasmodInputs


class TestPlasmodInputs:
    def test_write_writes_a_line_for_every_parameter(self):
        stream = io.StringIO()
        params = PlasmodInputs()

        params.write(stream)

        stream.seek(0)
        output = stream.read()
        lines = [line for line in output.split("\n") if line.strip()]
        params_dict = vars(params)
        assert len(lines) == len(params_dict)
        for param in params_dict:
            assert any(param in line for line in lines)

    @pytest.mark.parametrize(("model", "enum_cls"), MODEL_MAP.items())
    def test_model_is_converted_to_enum_on_init(self, model, enum_cls):
        # Just get the first member of the enum to test with
        enum_member = next(iter(enum_cls.__members__.values()))
        values = {model: enum_member.value}

        params = PlasmodInputs(**values)

        assert getattr(params, model) == getattr(enum_cls, enum_member.name)

    @pytest.mark.parametrize(("model", "enum_cls"), MODEL_MAP.items())
    def test_model_is_converted_to_enum_on_init_using_name(self, model, enum_cls):
        # Just get the first member enum name to test with
        enum_name = next(iter(enum_cls.__members__.keys()))
        values = {model: enum_name}

        params = PlasmodInputs(**values)

        assert getattr(params, model) == getattr(enum_cls, enum_name)

    @pytest.mark.parametrize(("model", "enum_cls"), MODEL_MAP.items())
    def test_CodesError_if_model_not_convertible_to_enum(
        self, model, enum_cls  # noqa: ARG002
    ):
        values = {model: "NOT_AN_ENUM_VALUE"}

        with pytest.raises(CodesError):
            PlasmodInputs(**values)
