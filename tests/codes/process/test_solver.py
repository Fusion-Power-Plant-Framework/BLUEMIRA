# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import contextlib
import filecmp
import json
import re
from pathlib import Path
from unittest import mock

import pytest

from bluemira.codes.error import CodesError
from bluemira.codes.process import ENABLED
from bluemira.codes.process._inputs import ProcessInputs
from bluemira.codes.process._solver import RunMode, Solver
from bluemira.codes.process.params import ProcessSolverParams
from tests._helpers import file_exists
from tests.codes.process import utilities as utils


class TestSolver:
    MODULE_REF = "bluemira.codes.process._solver"
    TEARDOWN_MODULE_REF = "bluemira.codes.process._teardown"

    IS_FILE_REF = f"{TEARDOWN_MODULE_REF}.Path.is_file"

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
            r".* unknown .* arguments: 'not_an_arg', 'also_not_an_arg'", call_args[0]
        )

    def test_none_mode_does_not_alter_parameters(self):
        solver = Solver(self.params, {})

        solver.execute(RunMode.NONE)

        assert solver.params.to_dict() == self.params.to_dict()

    def test_get_raw_variables_retrieves_parameters(self):
        solver = Solver(self.params, {"read_dir": utils.DATA_DIR})
        with (
            mock.patch(f"{self.MODULE_REF}.ENABLED", new=True),
            mock.patch(f"{self.TEARDOWN_MODULE_REF}._MFileWrapper", new=utils.mfw()),
            file_exists(Path(utils.READ_DIR, "MFILE.DAT"), self.IS_FILE_REF),
        ):
            solver.execute(RunMode.READ)

        assert solver.get_raw_variables("kappa_95") == [1.65]

    def test_get_raw_variables_CodesError_given_solver_not_run(self):
        solver = Solver(self.params, {"read_dir": utils.DATA_DIR})

        with pytest.raises(CodesError, match="solver has not been"):
            solver.get_raw_variables("kappa_95")

    def test_get_species_fraction_retrieves_parameter_value(self):
        solver = Solver(self.params, {"read_dir": utils.DATA_DIR})
        with (
            mock.patch(f"{self.MODULE_REF}.ENABLED", new=True),
            mock.patch(f"{self.TEARDOWN_MODULE_REF}._MFileWrapper", new=utils.mfw()),
            file_exists(Path(utils.READ_DIR, "MFILE.DAT"), self.IS_FILE_REF),
        ):
            solver.execute(RunMode.READ)

        assert solver.get_species_fraction("H") == pytest.approx(0.74267)
        assert solver.get_species_fraction("W") == pytest.approx(5e-5)


@pytest.mark.skipif(not ENABLED, reason="PROCESS is not installed on the system.")
class TestSolverIntegration:
    DATA_DIR = Path(Path(__file__).parent, "test_data")
    MODULE_REF = "bluemira.codes.process._setup"

    TEARDOWN_MODULE_REF = "bluemira.codes.process._teardown"
    IS_FILE_REF = f"{TEARDOWN_MODULE_REF}.Path.is_file"

    def setup_method(self):
        self.params = ProcessSolverParams.from_json(utils.PARAM_FILE)

        self._indat_patch = mock.patch(f"{self.MODULE_REF}.InDat")

    def teardown_method(self):
        self._indat_patch.stop()

    @pytest.mark.longrun
    def test_run_mode_outputs_process_files(self, tmp_path):
        solver = Solver(self.params, {"run_dir": tmp_path})

        with contextlib.suppress(CodesError):
            solver.execute(RunMode.RUNINPUT)

        assert Path(tmp_path, "IN.DAT").exists()
        assert Path(tmp_path, "MFILE.DAT").exists()

    @pytest.mark.parametrize("run_mode", [RunMode.READ, RunMode.READALL])
    def test_read_mode_updates_params_from_mfile(self, run_mode):
        # Assert here to check the parameter is actually changing
        assert self.params.r_tf_in_centre.value != pytest.approx(2.6354)

        solver = Solver(self.params, {"read_dir": self.DATA_DIR})

        with (
            mock.patch(f"{self.MODULE_REF}.ENABLED", new=True),
            mock.patch(f"{self.TEARDOWN_MODULE_REF}._MFileWrapper", new=utils.mfw()),
            file_exists(Path(utils.READ_DIR, "MFILE.DAT"), self.IS_FILE_REF),
        ):
            solver.execute(run_mode)

        # Expected value comes from ./test_data/MFILE.DAT
        assert solver.params.r_tf_in_centre.value == pytest.approx(2.6354)

    @pytest.mark.parametrize("run_mode", [RunMode.READ, RunMode.READALL])
    def test_derived_radial_build_params_are_updated(self, run_mode):
        solver = Solver(self.params, {"read_dir": self.DATA_DIR})
        with (
            mock.patch(f"{self.MODULE_REF}.ENABLED", new=True),
            mock.patch(
                f"{self.TEARDOWN_MODULE_REF}._MFileWrapper",
                new=utils.mfw(radial_override=False),
            ),
            file_exists(Path(utils.READ_DIR, "MFILE.DAT"), self.IS_FILE_REF),
        ):
            solver.execute(run_mode)

        # Expected values come from derivation (I added the numbers up by hand)
        assert solver.params.r_tf_in.value == pytest.approx(1.89236)
        assert solver.params.r_ts_ib_in.value == pytest.approx(3.47836)
        assert solver.params.r_vv_ib_in.value == pytest.approx(4.09836)
        assert solver.params.r_fw_ib_in.value == pytest.approx(4.89136)
        assert solver.params.r_fw_ob_in.value == pytest.approx(12.67696)
        assert solver.params.r_vv_ob_in.value == pytest.approx(13.69696)

    @pytest.mark.longrun
    def test_runinput_mode_does_not_edit_template(self, tmp_path):
        template_path = Path(self.DATA_DIR, "IN.DAT")
        build_config = {
            "run_dir": tmp_path,
            "template_in_dat": template_path,
        }

        solver = Solver(self.params, build_config)
        with contextlib.suppress(CodesError):
            solver.execute(RunMode.RUN)
        assert Path(tmp_path, "IN.DAT").is_file()
        filecmp.cmp(Path(tmp_path, "IN.DAT"), template_path)
        assert Path(tmp_path, "MFILE.DAT").is_file()

    def test_get_species_data_returns_row_vectors(self):
        temp, loss_f, z_eff = Solver.get_species_data("H", confinement_time_ms=1.0)

        assert isinstance(temp.size, int) == 1
        assert temp.size > 0
        assert isinstance(loss_f.size, int) == 1
        assert loss_f.size > 0
        assert isinstance(z_eff.size, int) == 1
        assert z_eff.size > 0

    def test_run_inits_writer_with_template_file_if_file_exists(self, tmp_path):
        build_config = {
            "run_dir": tmp_path,
            "template_in_dat": "template/path/in.dat",
        }

        class BLANK:
            get_value = 0.0

        with (
            self._indat_patch as indat_cls_mock,
            file_exists("template/path/in.dat", f"{self.MODULE_REF}.Path.is_file"),
        ):
            indat_cls_mock.return_value.data = {"casthi": BLANK}
            Solver(self.params, build_config)

        indat_cls_mock.assert_called_once_with(filename="template/path/in.dat")

    def test_run_inits_writer_without_template_returns_default_filled_data(self):
        with self._indat_patch as indat_cls_mock:
            solver = Solver(self.params, {})
            solver.run_cls = lambda *_, **_kw: None
            solver.teardown_cls = lambda *_, **_kw: None
            solver.execute("run")
        assert indat_cls_mock.return_value.data == self.params.template_defaults

    def test_run_raises_CodesError_given_no_data_in_template_file(self):
        build_config = {
            "template_in_dat": "template/path/in.dat",
        }

        with pytest.raises(CodesError):
            Solver(self.params, build_config)

    @pytest.mark.parametrize(("pf_n", "pf_v"), [(None, None), ("tk_sh_in", 3)])
    @pytest.mark.parametrize(
        ("template", "result"),
        [
            (ProcessInputs(bore=5, shldith=5, i_tf_wp_geom=2), (5, 5, 2)),
        ],
    )
    def test_indat_creation_with_template(self, template, result, pf_n, pf_v, tmp_path):
        if pf_n is None:
            pf = {}
        else:
            with open(utils.PARAM_FILE) as pf_h:
                pf = {pf_n: json.load(pf_h)[pf_n]}
            pf[pf_n]["value"] = pf_v
            result = (result[0], pf_v, result[2])
        path = tmp_path / "IN.DAT"
        build_config = {
            "in_dat_path": path,
            "template_in_dat": template,
        }

        solver = Solver(pf, build_config)
        solver.params.mappings["tk_sh_in"].send = True
        solver.run_cls = lambda *_, **_kw: None
        solver.teardown_cls = lambda *_, **_kw: None
        solver.execute("run")

        assert f"bore     = {result[0]}" in open(path).read()  # noqa: SIM115
        assert f"shldith  = {result[1]}" in open(path).read()  # noqa: SIM115
        assert f"i_tf_wp_geom = {result[2]}" in open(path).read()  # noqa: SIM115
