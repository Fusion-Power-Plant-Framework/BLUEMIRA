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

import filecmp
import re
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from bluemira.codes.error import CodesError
from bluemira.codes.process import ENABLED
from bluemira.codes.process._solver import RunMode, Solver
from bluemira.codes.process.params import ProcessSolverParams
from tests.codes.process import utilities as utils


class TestSolver:
    MODULE_REF = "bluemira.codes.process._solver"

    @classmethod
    def setup_class(cls):
        cls._mfile_patch = mock.patch(
            "bluemira.codes.process._teardown.MFile", new=utils.FakeMFile
        )
        cls.mfile_mock = cls._mfile_patch.start()

    @classmethod
    def teardown_class(cls):
        cls._mfile_patch.stop()

    def setup_method(self):
        self.params = ProcessSolverParams.from_json(utils.PARAM_FILE)

    def teardown_method(self):
        self.mfile_mock.reset_data()

    @mock.patch(f"{MODULE_REF}.bluemira_warn")
    def test_bluemira_warning_if_build_config_has_unknown_arg(self, bm_warn_mock):
        build_config = {"not_an_arg": 0, "also_not_an_arg": 0}

        Solver(self.params, build_config)

        bm_warn_mock.assert_called_once()
        call_args, _ = bm_warn_mock.call_args
        assert re.match(
            ".* unknown .* arguments: 'not_an_arg', 'also_not_an_arg'", call_args[0]
        )

    def test_none_mode_does_not_alter_parameters(self):
        solver = Solver(self.params, {})

        solver.execute(RunMode.NONE)

        assert solver.params.to_dict() == self.params.to_dict()

    def test_get_raw_variables_retrieves_parameters(self):
        solver = Solver(self.params, {"read_dir": utils.DATA_DIR})
        solver.execute(RunMode.READ)

        assert solver.get_raw_variables("kappa_95") == [1.65]

    def test_get_raw_variables_CodesError_given_solver_not_run(self):
        solver = Solver(self.params, {"read_dir": utils.DATA_DIR})

        with pytest.raises(CodesError):
            solver.get_raw_variables("kappa_95")

    def test_get_species_fraction_retrieves_parameter_value(self):
        solver = Solver(self.params, {"read_dir": utils.DATA_DIR})
        solver.execute(RunMode.READ)

        assert solver.get_species_fraction("H") == pytest.approx(0.74267)
        assert solver.get_species_fraction("W") == pytest.approx(5e-5)


@pytest.mark.skipif(not ENABLED, reason="PROCESS is not installed on the system.")
class TestSolverIntegration:
    DATA_DIR = Path(Path(__file__).parent, "test_data")

    def setup_method(self):
        self.params = ProcessSolverParams.from_json(utils.PARAM_FILE)

    @pytest.mark.longrun
    def test_run_mode_outputs_process_files(self):
        run_dir = tempfile.TemporaryDirectory()
        build_config = {"run_dir": run_dir.name}
        solver = Solver(self.params, build_config)

        solver.execute(RunMode.RUN)

        assert Path(run_dir.name, "IN.DAT").exists()
        assert Path(run_dir.name, "MFILE.DAT").exists()

    @pytest.mark.parametrize("run_mode", [RunMode.READ, RunMode.READALL])
    def test_read_mode_updates_params_from_mfile(self, run_mode):
        # Assert here to check the parameter is actually changing
        assert self.params.r_tf_in_centre.value != pytest.approx(2.6354)
        build_config = {"read_dir": self.DATA_DIR}

        solver = Solver(self.params, build_config)
        solver.execute(run_mode)

        # Expected value comes from ./test_data/MFILE.DAT
        assert solver.params.r_tf_in_centre.value == pytest.approx(2.6354)

    @pytest.mark.parametrize("run_mode", [RunMode.READ, RunMode.READALL])
    def test_derived_radial_build_params_are_updated(self, run_mode):
        build_config = {"read_dir": self.DATA_DIR}

        solver = Solver(self.params, build_config)
        solver.execute(run_mode)

        # Expected values come from derivation (I added the numbers up by hand)
        assert solver.params.r_tf_in.value == pytest.approx(1.89236)
        assert solver.params.r_ts_ib_in.value == pytest.approx(3.47836)
        assert solver.params.r_vv_ib_in.value == pytest.approx(4.09836)
        assert solver.params.r_fw_ib_in.value == pytest.approx(4.89136)
        assert solver.params.r_fw_ob_in.value == pytest.approx(12.67696)
        assert solver.params.r_vv_ob_in.value == pytest.approx(13.69696)

    @pytest.mark.longrun
    def test_runinput_mode_does_not_edit_template(self):
        run_dir = tempfile.TemporaryDirectory()
        template_path = Path(self.DATA_DIR, "IN.DAT")
        build_config = {
            "run_dir": run_dir.name,
            "template_in_dat": template_path,
        }

        solver = Solver(self.params, build_config)
        solver.execute(RunMode.RUN)

        assert Path(run_dir.name, "IN.DAT").is_file()
        filecmp.cmp(Path(run_dir.name, "IN.DAT"), template_path)
        assert Path(run_dir.name, "MFILE.DAT").is_file()

    def test_get_species_data_returns_row_vectors(self):
        temp, loss_f, z_eff = Solver.get_species_data("H")

        assert isinstance(temp.size, int) == 1
        assert temp.size > 0
        assert isinstance(loss_f.size, int) == 1
        assert loss_f.size > 0
        assert isinstance(z_eff.size, int) == 1
        assert z_eff.size > 0
