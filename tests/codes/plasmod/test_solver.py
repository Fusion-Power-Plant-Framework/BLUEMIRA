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
import re
import tempfile
from pathlib import Path
from typing import List
from unittest import mock

import numpy as np
import pytest
from scipy.interpolate import interp1d

from bluemira.base.constants import EPS
from bluemira.codes import plasmod
from bluemira.codes.error import CodesError
from bluemira.codes.plasmod.api import Run, Setup, Teardown
from bluemira.codes.plasmod.params import PlasmodSolverParams
from tests._helpers import combine_text_mock_write_calls

SOLVER_MODULE_REF = "bluemira.codes.plasmod.api"
RUN_SUBPROCESS_REF = "bluemira.codes.interface.run_subprocess"
DATA_DIR = Path(Path(__file__).parent, "data")
PARAMS_FILE = Path(DATA_DIR, "params.json").as_posix()


class TestPlasmodSetup:
    MODULE_REF = f"{SOLVER_MODULE_REF}._setup"

    def setup_method(self):
        self.default_pf = PlasmodSolverParams.from_json(PARAMS_FILE)
        self.input_file = "/path/to/input.dat"

    def test_inputs_updated_from_problem_settings_on_init(self):
        problem_settings = {
            "v_loop": -1.5e-3,
            "q_heat": 1.5,
            "nx": 25,
        }

        setup = Setup(self.default_pf, problem_settings, self.input_file)

        assert setup.inputs.v_loop == pytest.approx(-1.5e-3, rel=0, abs=EPS)
        assert setup.inputs.q_heat == pytest.approx(1.5, rel=0, abs=EPS)
        assert setup.inputs.nx == pytest.approx(25, rel=0, abs=EPS)

    def test_update_inputs_changes_input_values(self):
        new_inputs = {
            "v_loop": -1.5e-3,
            "q_heat": 1.5,
            "nx": 25,
        }

        setup = Setup(self.default_pf, {}, self.input_file)

        setup.update_inputs(new_inputs)

        assert setup.inputs.v_loop == pytest.approx(-1.5e-3, rel=0, abs=EPS)
        assert setup.inputs.q_heat == pytest.approx(1.5, rel=0, abs=EPS)
        assert setup.inputs.nx == pytest.approx(25, rel=0, abs=EPS)

    def test_update_inputs_shows_warning_if_input_unknown(self):
        new_inputs = {"not_a_param": -1.5e-3}
        setup = Setup(self.default_pf, {}, self.input_file)

        with mock.patch(f"{self.MODULE_REF}.bluemira_warn") as bm_warn:
            setup.update_inputs(new_inputs)

        bm_warn.assert_called_once()

    def test_run_writes_plasmod_dat_file(self):
        problem_settings = {"v_loop": -1.5e-3, "q_heat": 1.5, "nx": 25}
        setup = Setup(self.default_pf, problem_settings, "/some/file/path.dat")

        with mock.patch("builtins.open", new_callable=mock.mock_open) as open_mock:
            setup.run()

        open_mock.assert_called_once_with("/some/file/path.dat", "w")
        output = combine_text_mock_write_calls(open_mock)
        assert re.search(r"^ *v_loop +-0.150+E-02\n", output, re.MULTILINE)
        assert re.search(r"^ *q_heat +0.150+E\+01\n", output, re.MULTILINE)
        assert re.search(r"^ *nx +25\n", output, re.MULTILINE)

    @mock.patch("builtins.open", new_callable=mock.mock_open)
    def test_CodesError_if_writing_to_plasmod_dat_file_fails(self, open_mock):
        problem_settings = {"v_loop": -1.5e-3, "q_heat": 1.5, "nx": 25}
        setup = Setup(
            self.default_pf,
            problem_settings=problem_settings,
            plasmod_input_file="/some/file/path.dat",
        )
        open_mock.side_effect = OSError

        with pytest.raises(CodesError):
            setup.run()

    def test_mock_does_not_write_dat_file(self):
        setup = Setup(self.default_pf, {}, self.input_file)

        with mock.patch("builtins.open", new_callable=mock.mock_open) as open_mock:
            setup.mock()

        open_mock.assert_not_called()

    def test_read_does_not_write_dat_file(self):
        setup = Setup(self.default_pf, {}, self.input_file)

        with mock.patch("builtins.open", new_callable=mock.mock_open) as open_mock:
            setup.read()

        open_mock.assert_not_called()


class TestPlasmodRun:
    def setup_method(self):
        self._run_subprocess_patch = mock.patch(RUN_SUBPROCESS_REF)
        self.run_subprocess_mock = self._run_subprocess_patch.start()
        self.run_subprocess_mock.return_value = 0

        self.default_pf = PlasmodSolverParams.from_json(PARAMS_FILE)

    def teardown_method(self):
        self._run_subprocess_patch.stop()

    @pytest.mark.parametrize(
        ("arg", "arg_num"),
        [
            ("plasmod_binary", 0),
            ("input.dat", 1),
            ("output.dat", 2),
            ("profiles.dat", 3),
        ],
    )
    def test_run_calls_subprocess_with_argument_in_position(self, arg, arg_num):
        run = Run(
            self.default_pf,
            "input.dat",
            "output.dat",
            "profiles.dat",
            binary="plasmod_binary",
        )

        run.run()

        self.run_subprocess_mock.assert_called_once()
        args, _ = self.run_subprocess_mock.call_args
        assert args[0][arg_num] == arg

    def test_run_raises_CodesError_given_run_subprocess_raises_OSError(self):
        self.run_subprocess_mock.side_effect = OSError
        run = Run(self.default_pf, "input.dat", "output.dat", "profiles.dat")

        with pytest.raises(CodesError):
            run.run()

    def test_run_raises_CodesError_given_run_process_returns_non_zero_exit_code(self):
        self.run_subprocess_mock.return_value = 1
        run = Run(self.default_pf, "input.dat", "output.dat", "profiles.dat")

        with pytest.raises(CodesError):
            run.run()


class TestPlasmodTeardown:
    plasmod_out_sample = (
        "     betan      0.14092930140E+0002\n"
        "      fbs       0.14366031154E+0002\n"
        "      rli       0.16682353334E+0002\n"
        " i_flag           1\n"
    )

    plasmod_prof_sample = (
        "ntrit           0.00000000000E+0000      0.25000000000E-0001     "
        " 0.50000000000E-0001\n"
    )

    def setup_method(self):
        self.default_pf = PlasmodSolverParams.from_json(PARAMS_FILE)

    def mock_file_open(self, filename, mode="r"):  # noqa: ARG002
        if "prof" in str(filename):
            content = self.plasmod_prof_sample
        elif "file" in str(filename):
            content = self.plasmod_out_sample

        file_object = mock.mock_open(read_data=content).return_value
        file_object.__iter__.return_value = content
        return file_object

    @pytest.mark.parametrize("run_mode_func", ["run", "read"])
    def test_run_mode_function_updates_plasmod_params_from_file(self, run_mode_func):
        teardown = Teardown(
            self.default_pf,
            "file.csv",
            "prof.csv",
            "/path/to/output/",
            "/path/to/output/",
        )

        with mock.patch("builtins.open", new=self.mock_file_open):
            getattr(teardown, run_mode_func)()

        assert teardown.params.beta_N.value == pytest.approx(0.14092930140e2)
        assert teardown.params.f_bs.value == pytest.approx(0.14366031154e2)
        assert teardown.params.l_i.value == pytest.approx(0.16682353334e2)

    def test_mock_leaves_plasmod_params_with_defaults(self):
        default_pf_copy = copy.deepcopy(self.default_pf)
        teardown = Teardown(self.default_pf, "file.csv", "file.csv", DATA_DIR, DATA_DIR)

        teardown.mock()

        # Read from data/mockPLASMOD.json
        assert teardown.params.beta_N.value == pytest.approx(3.109, rel=1e-3)
        assert teardown.params.f_bs.value == pytest.approx(0.369, rel=1e-3)
        assert teardown.params.l_i.value == pytest.approx(2.423, rel=1e-3)

    @pytest.mark.parametrize("run_mode_func", ["run", "read"])
    def test_CodesError_if_output_files_cannot_be_read(self, run_mode_func):
        teardown = Teardown(
            self.default_pf,
            "file.csv",
            "file.csv",
            "/path/to/output/",
            "/path/to/output/",
        )

        with mock.patch("builtins.open", side_effect=OSError), pytest.raises(CodesError):
            getattr(teardown, run_mode_func)()

    @pytest.mark.parametrize("run_mode_func", ["run", "read"])
    def test_run_mode_function_opens_both_output_files(self, run_mode_func):
        teardown = Teardown(
            self.default_pf,
            "out.csv",
            "prof.csv",
            "/path/to/output/",
            "/path/to/output/",
        )

        with mock.patch(
            "builtins.open",
            new_callable=mock.mock_open,
            read_data=self.plasmod_out_sample,
        ) as open_mock:
            getattr(teardown, run_mode_func)()

        assert open_mock.call_count == 2
        call_args = [call.args for call in open_mock.call_args_list]
        assert (Path("/path/to/output/out.csv"),) in call_args
        assert (Path("/path/to/output/prof.csv"),) in call_args

    @pytest.mark.parametrize("i_flag", [2, 0, -1, -2, 100, -100])
    def test_CodesError_if_plasmod_status_flag_ne_1(self, i_flag):
        output_sample = (
            "     betan      0.14092930140E+0002\n"
            "      fbs       0.14366031154E+0002\n"
            "      rli       0.16682353334E+0002\n"
            f" i_flag           {i_flag}\n"
        )
        open_mock = mock.mock_open(read_data=output_sample)
        teardown = Teardown(
            self.default_pf,
            "file.csv",
            "file.csv",
            "/path/to/output/",
            "/path/to/output/",
        )

        with mock.patch("builtins.open", new=open_mock), pytest.raises(CodesError):
            teardown.run()

    @mock.patch("bluemira.codes.interface.bluemira_warn")
    def test_warning_issued_if_output_param_is_missing(self, bm_warn_mock):
        open_mock = mock.mock_open(read_data=self.plasmod_out_sample)
        teardown = Teardown(
            self.default_pf,
            "file.csv",
            "file.csv",
            "/path/to/output/",
            "/path/to/output/",
        )

        with mock.patch("builtins.open", new=open_mock):
            teardown.run()

        assert bm_warn_mock.call_count > 0
        assert bm_warn_mock.call_count > 0
        assert "No value for output parameter" in bm_warn_mock.call_args[0][0]
        assert "PLASMOD" in bm_warn_mock.call_args[0][0]


class TestPlasmodSolver:
    def setup_method(self):
        self.default_pf = PlasmodSolverParams.from_json(PARAMS_FILE)
        self.build_config = {
            "input_file": tempfile.NamedTemporaryFile("w").name,
            "output_file": tempfile.NamedTemporaryFile("w").name,
            "profiles_file": tempfile.NamedTemporaryFile("w").name,
        }

    @classmethod
    def setup_class(cls):
        cls._run_subprocess_patch = mock.patch(
            RUN_SUBPROCESS_REF,
            wraps=cls._plasmod_run_subprocess_fake,
        )
        cls.run_subprocess_mock = cls._run_subprocess_patch.start()

    @classmethod
    def teardown_class(cls):
        cls._run_subprocess_patch.stop()

    @pytest.mark.parametrize(
        ("key", "default"),
        [
            ("binary", plasmod.BINARY),
            ("problem_settings", {}),
            ("input_file", plasmod.Solver.DEFAULT_INPUT_FILE),
            ("output_file", plasmod.Solver.DEFAULT_OUTPUT_FILE),
            ("profiles_file", plasmod.Solver.DEFAULT_PROFILES_FILE),
        ],
    )
    def test_init_sets_default_build_config_value(self, key, default):
        solver = plasmod.Solver(self.default_pf)

        assert getattr(solver, key) == default

    @pytest.mark.parametrize("param_type", ["kvdict", "dict", "frame"])
    def test_execute_in_run_mode_sets_expected_params(self, param_type):
        if param_type == "kvdict":
            param = {k: data["value"] for k, data in self.default_pf.to_dict().items()}
        elif param_type == "dict":
            param = self.default_pf.to_dict()
        else:
            param = self.default_pf

        solver = plasmod.Solver(param, self.build_config)

        pf = solver.execute(plasmod.RunMode.RUN)

        self.run_subprocess_mock.assert_called_once_with(
            [
                plasmod.BINARY,
                self.build_config["input_file"],
                self.build_config["output_file"],
                self.build_config["profiles_file"],
            ]
        )
        self.run_subprocess_mock.reset_mock()
        assert pf.beta_N.value == pytest.approx(3.0007884293)

    def test_get_raw_profile_returns_profile_array(self):
        solver = plasmod.Solver(self.default_pf, self.build_config)
        solver.execute(plasmod.RunMode.RUN)

        # Raw profile
        outputs = solver.plasmod_outputs()
        profile = outputs.Te
        # Expected values taken from 'data/sample_profiles.dat'
        expected_values = [43.9383, 44.7500, 45.3127, 45.6264, 45.6912, 45.5069, 45.0737]
        np.testing.assert_almost_equal(profile, np.array(expected_values), decimal=4)

    def test_get_profile_returns_profile_array(self):
        solver = plasmod.Solver(self.default_pf, self.build_config)
        solver.execute(plasmod.RunMode.RUN)

        # Scaled profile
        profile = solver.get_profile(plasmod.Profiles.Te)
        # Expected values taken from 'data/sample_profiles.dat' (in x_phi coordinates)
        expected_values = [43.9383, 44.7500, 45.3127, 45.6264, 45.6912, 45.5069, 45.0737]
        scaled_expected_values = interp1d(solver._x_psi, expected_values, kind="linear")(
            solver._x_phi
        )
        np.testing.assert_almost_equal(
            profile, np.array(scaled_expected_values), decimal=4
        )

    def test_scaled_profile_not_equal_raw_profile(self):
        solver = plasmod.Solver(self.default_pf, self.build_config)
        solver.execute(plasmod.RunMode.RUN)

        # Scaled profile
        profile = solver.get_profile(plasmod.Profiles.Te)
        # Raw profile
        outputs = solver.plasmod_outputs()
        raw_profile = outputs.Te
        assert np.any(np.not_equal(profile, raw_profile))

    def test_get_profiles_returns_dict_of_profiles(self):
        solver = plasmod.Solver(self.default_pf, self.build_config)
        solver.execute(plasmod.RunMode.RUN)
        profile_keys = [plasmod.Profiles.Te, plasmod.Profiles.g3]

        profiles = solver.get_profiles(profile_keys)

        assert all(profile in profiles for profile in profile_keys)
        assert all(isinstance(profile, np.ndarray) for profile in profiles.values())

    def test_plasmod_outputs_contains_unmapped_param(self):
        solver = plasmod.Solver(self.default_pf, self.build_config)
        solver.execute(plasmod.RunMode.RUN)

        outputs = solver.plasmod_outputs()

        # Expected value taken from 'data/sample_output.dat'
        assert outputs.cprotium == pytest.approx(0.77034698659)

    def test_execute_after_updating_settings_runs_with_new_config(self):
        # Run the solver once, edit the problem settings, then run it
        # again. Then verify it was run with the new settings.
        solver = plasmod.Solver(self.default_pf, self.build_config)
        solver.execute(plasmod.RunMode.RUN)

        solver.problem_settings["qdivt_sup"] = 10.0
        solver.execute(plasmod.RunMode.RUN)

        param_value = self._get_value_from_input_file(
            "qdivt_sup", self.build_config["input_file"]
        )
        assert param_value == 10.0

    def test_param_not_modified_in_plasmod_input_if_modify_mapping_send_is_False(self):
        solver = plasmod.Solver(self.default_pf, self.build_config)
        solver.params.q_95.set_value(5, source="Input")

        solver.modify_mappings({"q_95": {"send": False}})
        solver.execute(plasmod.RunMode.RUN)

        param_value = self._get_value_from_input_file(
            "q95", self.build_config["input_file"]
        )
        assert param_value != 5

    def test_output_param_not_modified_if_modify_mappings_recv_set_to_False(self):
        solver = plasmod.Solver(self.default_pf, self.build_config)
        original_beta_n = solver.params.beta_N.value

        solver.modify_mappings({"beta_N": {"recv": False}})
        solver.execute(plasmod.RunMode.RUN)

        assert solver.params.beta_N.value == original_beta_n

    def test_plasmod_solver_plotting(self):
        solver = plasmod.Solver(self.default_pf, self.build_config)
        solver.execute(plasmod.RunMode.RUN)

        f, ax = plasmod.plot_default_profiles(solver)
        assert len(f.get_axes()) == 6

    @staticmethod
    def read_data_file(file_name):
        data_dir = Path(Path(__file__).parent, "data")
        with open(Path(data_dir, file_name)) as f:
            return f.read()

    @staticmethod
    def _plasmod_run_subprocess_fake(command: List[str], **_):
        """
        Fake a run of plasmod, outputting some sample results files.

        This replaces the run_subprocess call for plasmod in this test
        class.
        """

        def write_file(file_path: str, content: str):
            with open(file_path, "w") as f:
                f.write(content)

        output_file = command[2]
        output_file_content = TestPlasmodSolver.read_data_file("sample_output.dat")
        profiles_file = command[3]
        profiles_file_content = TestPlasmodSolver.read_data_file("sample_profiles.dat")

        write_file(output_file, output_file_content)
        write_file(profiles_file, profiles_file_content)
        return 0

    def _get_value_from_input_file(self, param: str, file_path: str):
        """
        Find a parameter value from a plasmod input file via a regex.
        This should work with floats and ints.
        """
        with open(file_path) as f:
            input_file = f.read()
        param_regex = param + r" +([0-9]+(\.[0-9]+E[\+-][0-9]+)?)"
        re_match = re.search(param_regex, input_file, re.MULTILINE)
        if re_match:
            return float(re_match.group(1))
        return None
