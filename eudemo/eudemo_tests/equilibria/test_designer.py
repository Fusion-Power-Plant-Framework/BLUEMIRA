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

import json
from pathlib import Path
from typing import Dict

import pytest

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria import Equilibrium
from eudemo.equilibria import EquilibriumDesigner
from eudemo.equilibria._designer import EquilibriumDesignerParams


class TestEquilibriumDesigner:
    EQDSK_FILE = Path(
        get_bluemira_path("equilibria", subfolder="data"), "EU-DEMO_EOF.json"
    )
    DATA_DIR = Path(Path(__file__).parent, "test_data")

    @classmethod
    def setup_class(cls):
        cls.param_dict = cls._read_json(Path(cls.DATA_DIR, "params.json"))

    def test_params_converted_to_parameter_frame(self):
        designer = EquilibriumDesigner(self.param_dict)

        assert isinstance(designer.params, EquilibriumDesignerParams)

    @pytest.mark.longrun
    def test_designer_converges_on_run(self):
        designer = EquilibriumDesigner(self.param_dict, {"plot_optimisation": True})

        eq = designer.execute()

        assert eq.get_LCFS()
        # check parameters have been updated
        assert designer.params != EquilibriumDesignerParams.from_dict(self.param_dict)

    def test_designer_reads_file_in_read_mode(self):
        eqdsk = self.EQDSK_FILE
        designer = EquilibriumDesigner(
            self.param_dict, {"run_mode": "read", "file_path": eqdsk}
        )

        eq = designer.execute()

        ref_eq = Equilibrium.from_eqdsk(eqdsk)
        assert eq.analyse_plasma() == ref_eq.analyse_plasma()

    def test_ValueError_on_init_given_read_mode_and_no_file_path(self):
        with pytest.raises(ValueError):  # noqa: PT011
            EquilibriumDesigner(self.param_dict, {"run_mode": "read"})

    @staticmethod
    def _read_json(file_path: str) -> Dict:
        with open(file_path) as f:
            return json.load(f)
