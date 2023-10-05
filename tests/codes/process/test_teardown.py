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
import copy
from pathlib import Path
from typing import ClassVar
from unittest import mock

import numpy as np
import pytest

from bluemira.codes.error import CodesError
from bluemira.codes.process._teardown import Teardown, _MFileWrapper
from bluemira.codes.process.params import ProcessSolverParams
from tests._helpers import file_exists
from tests.codes.process import utilities as utils


class TestTeardown:
    MODULE_REF = "bluemira.codes.process._teardown"
    IS_FILE_REF = f"{MODULE_REF}.Path.is_file"

    @classmethod
    def setup_class(cls):
        cls._pf = ProcessSolverParams.from_json(utils.PARAM_FILE)
        cls._mfile_patch = mock.patch(f"{cls.MODULE_REF}.MFile", new=utils.FakeMFile)

    def setup_method(self):
        self.default_pf = copy.deepcopy(self._pf)
        self.mfile_mock = self._mfile_patch.start()

    def teardown_method(self):
        self.mfile_mock.reset_data()
        self._mfile_patch.stop()

    @pytest.mark.parametrize("run_func", ["run", "runinput"])
    def test_run_func_updates_bluemira_params_from_mfile(self, run_func):
        teardown = Teardown(self.default_pf, utils.RUN_DIR, None)

        with file_exists(Path(utils.RUN_DIR, "MFILE.DAT"), self.IS_FILE_REF):
            getattr(teardown, run_func)()

        # Expected value comes from ./test_data/mfile_data.json
        assert teardown.params.tau_e.value == pytest.approx(4.3196)

    @pytest.mark.parametrize("run_func", ["read", "readall"])
    def test_read_func_updates_bluemira_params_from_mfile(self, run_func):
        teardown = Teardown(self.default_pf, None, utils.READ_DIR)

        with file_exists(Path(utils.READ_DIR, "MFILE.DAT"), self.IS_FILE_REF):
            getattr(teardown, run_func)()

        # Expected value comes from ./test_data/mfile_data.json
        assert teardown.params.tau_e.value == pytest.approx(4.3196)
        # auto unit conversion
        assert teardown.params.P_el_net.value == pytest.approx(5e8)

    def test_read_unknown_outputs_set_to_nan(self):
        """
        If a variable no longer exists in PROCESS we assume its OBS_VARS
        entry is None. If a user asks for that var we set its value to np.nan
        Tested for a value with units.
        """

        class MFile:
            data: ClassVar = {"enbeam": {"var_mod": "some info", "scan01": 1234}}

        class MFW(_MFileWrapper):
            # Overwrite some methods because data doesnt exist in 'mfile'

            def __init__(self, path, name):  # noqa: ARG002
                self.mfile = MFile()
                self._name = name

            def _derive_radial_build_params(self, data):  # noqa: ARG002
                return {}

        teardown = Teardown(self.default_pf, None, utils.READ_DIR)

        # Less Warnings
        for m in teardown.params.mappings.values():
            m.recv = False

        teardown.params.mappings["e_nbi"].recv = True

        # Test
        with mock.patch(f"{self.MODULE_REF}._MFileWrapper", new=MFW), file_exists(
            Path(utils.READ_DIR, "MFILE.DAT"), self.IS_FILE_REF
        ), mock.patch("bluemira.codes.process.api.OBS_VARS", new={"enbeam": None}):
            teardown.read()

        assert np.isnan(teardown.params.e_nbi.value)

    def test_CodesError_on_bad_output(self):
        class MFile:
            def __init__(self, file):  # noqa: ARG002
                self.data = {"ifail": {"scan01": 2}}

        with pytest.raises(CodesError), mock.patch(
            "bluemira.codes.process._teardown.Path"
        ), mock.patch("bluemira.codes.process._teardown.MFile", new=MFile):
            _MFileWrapper(None)

    @pytest.mark.parametrize(
        ("run_func", "data_dir"),
        [("runinput", utils.RUN_DIR), ("readall", utils.READ_DIR)],
    )
    def test_run_mode_updates_params_from_mfile_given_recv_False(
        self, run_func, data_dir
    ):
        # The two run modes in this test are expected to ignore the
        # 'recv = False' Parameter attribute
        teardown = Teardown(self.default_pf, utils.RUN_DIR, utils.READ_DIR)
        teardown.params.update_mappings({"r_tf_in_centre": {"recv": False}})

        with file_exists(Path(data_dir, "MFILE.DAT"), self.IS_FILE_REF):
            getattr(teardown, run_func)()

        # Expected value comes from ./test_data/mfile_data.json
        assert teardown.params.r_tf_in_centre.value == pytest.approx(2.6354)

    @pytest.mark.parametrize(
        ("run_func", "data_dir"), [("run", utils.RUN_DIR), ("read", utils.READ_DIR)]
    )
    def test_run_mode_does_not_update_params_from_mfile_given_recv_False(
        self, run_func, data_dir
    ):
        teardown = Teardown(self.default_pf, utils.RUN_DIR, utils.READ_DIR)
        teardown.params.update_mappings({"r_tf_in_centre": {"recv": False}})

        with file_exists(Path(data_dir, "MFILE.DAT"), self.IS_FILE_REF):
            getattr(teardown, run_func)()

        # Non-expected value comes from ./test_data/mfile_data.json
        assert teardown.params.r_tf_in_centre.value != pytest.approx(2.6354)

    def test_mock_updates_params_from_mockPROCESS_json_file(self):
        teardown = Teardown(self.default_pf, None, utils.READ_DIR)

        teardown.mock()

        # Expected values come from ./test_data/read/mockPROCESS.json
        assert teardown.params.delta_95.value == pytest.approx(0.33333)
        assert teardown.params.B_0.value == pytest.approx(5.2742)
        assert teardown.params.r_fw_ib_in.value == pytest.approx(5.953487)
        assert teardown.params.v_burn.value == pytest.approx(0.032175)

    def test_mock_raises_CodesError_if_mock_file_does_not_exist(self):
        teardown = Teardown(self.default_pf, None, "./not/a/dir")

        with pytest.raises(CodesError):
            teardown.mock()

    @pytest.mark.parametrize("run_func", ["run", "runinput", "read", "readall"])
    def test_run_func_raises_CodesError_given_mfile_does_not_exist(self, run_func):
        teardown = Teardown(self.default_pf, "./not/a/dir", "./not/a/dir")

        assert not Path("./not/a/dir/MFILE.DAT").is_file()
        with pytest.raises(CodesError) as codes_err:
            getattr(teardown, run_func)()
        assert "is not a file" in str(codes_err)

    @pytest.mark.parametrize("run_func", ["run", "runinput", "read", "readall"])
    @pytest.mark.parametrize("i_fail", [-1, 0, 2, 100])
    def test_run_func_raises_CodesError_given_process_infeasible_soln(
        self, run_func, i_fail
    ):
        self.mfile_mock.data["ifail"]["scan01"] = i_fail
        teardown = Teardown(self.default_pf, "./some/dir", "./some/dir")

        with file_exists("./some/dir/MFILE.DAT", self.IS_FILE_REF):
            with pytest.raises(CodesError) as codes_err:
                getattr(teardown, run_func)()
            assert "did not find a feasible solution" in str(codes_err)

    def test_obsolete_vars_with_multiple_new_names_all_have_mappings(self):
        def fake_uov(param: str):
            if param == "thshield":
                return ["thshield_ib", "thshield_ob", "thshield_vb"]
            return param

        teardown = Teardown(self.default_pf, None, utils.READ_DIR)
        with mock.patch(
            f"{self.MODULE_REF}.update_obsolete_vars", new=fake_uov
        ), file_exists(Path(utils.READ_DIR, "MFILE.DAT"), self.IS_FILE_REF):
            teardown.read()

        outputs = teardown.get_raw_outputs(["thshield_ib", "thshield_ob", "thshield_vb"])
        # value from the 'thshield' param in ./test_data/mfile_data.json
        assert outputs == [0.05, 0.05, 0.05]

    def test_CodesError_if_process_parameter_missing_from_radial_build_calculation(self):
        teardown = Teardown(self.default_pf, None, utils.READ_DIR)
        del self.mfile_mock.data["bore"]

        with file_exists(
            Path(utils.READ_DIR, "MFILE.DAT"), self.IS_FILE_REF
        ), pytest.raises(CodesError) as exc:
            teardown.read()

        assert "bore" in str(exc)
