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

from unittest import mock

import pytest

from bluemira.codes import process
from bluemira.codes.error import CodesError
from bluemira.codes.process._setup import Setup
from bluemira.codes.process.mapping import mappings as process_mappings
from bluemira.codes.process.params import ProcessSolverParams
from tests._helpers import file_exists
from tests.codes.process.utilities import PARAM_FILE

MODULE_REF = "bluemira.codes.process._setup"


class TestSetup:
    def setup_method(self):
        self.default_pf = ProcessSolverParams.from_json(PARAM_FILE)

        self._writer_patch = mock.patch(f"{MODULE_REF}._make_writer")
        self._indat_patch = mock.patch(f"{MODULE_REF}.InDat")

    def teardown_method(self):
        self._writer_patch.stop()
        self._indat_patch.stop()

    def test_run_adds_bluemira_params_to_InDat_writer(self):
        with self._writer_patch as writer_cls_mock:
            setup = Setup(self.default_pf, "")
            writer_cls_mock.return_value.data = {"x": 0}

            setup.run()

        writer = writer_cls_mock.return_value
        num_send_params = sum(1 for x in process_mappings.values() if x.send)
        assert writer.add_parameter.call_count == num_send_params
        # Expected value comes from default pf
        call_arg_list = writer.add_parameter.call_args_list
        assert mock.call("pnetelin", 500) in call_arg_list

    def test_run_adds_problem_setting_params_to_InDat_writer(self):
        problem_settings = {"input0": 0.0}
        with self._writer_patch as writer_cls_mock:
            setup = Setup(self.default_pf, "", problem_settings=problem_settings)
            writer_cls_mock.return_value.data = {"x": 0}

            setup.run()

        writer = writer_cls_mock.return_value
        assert writer.add_parameter.call_count > 0
        assert mock.call("input0", 0.0) in writer.add_parameter.call_args_list

    def test_run_inits_writer_with_template_file_if_file_exists(self):
        with self._indat_patch as indat_cls_mock:
            setup = Setup(self.default_pf, "", template_in_dat="template/path/in.dat")
            indat_cls_mock.return_value.data = {"input": 0.0}

            with file_exists("template/path/in.dat", f"{MODULE_REF}.Path.is_file"):
                setup.run()

        indat_cls_mock.assert_called_once_with(filename="template/path/in.dat")

    def test_run_inits_writer_without_template_returns_default_filled_data(self):
        with self._indat_patch as indat_cls_mock:
            setup = Setup(self.default_pf, "", template_in_dat=None)
            setup.run()

        assert indat_cls_mock.return_value.data == self.default_pf.template_defaults

    @pytest.mark.parametrize("run_func", ["run", "runinput"])
    def test_run_raises_CodesError_given_no_data_in_template_file(self, run_func):
        setup = Setup(self.default_pf, "", template_in_dat="template/path/in.dat")

        with pytest.raises(CodesError):
            getattr(setup, run_func)()

    def test_runinput_does_not_write_bluemira_outputs_to_in_dat(self):
        with self._writer_patch as writer_cls_mock:
            setup = Setup(self.default_pf, "")
            writer_cls_mock.return_value.data = {"x": 0}

            setup.runinput()

        writer_cls_mock.return_value.add_parameter.assert_not_called()

    @pytest.mark.parametrize(("model_name", "model_cls"), Setup.MODELS.items())
    def test_models_are_converted_to_Model_classes_given_str_value(
        self, model_name, model_cls
    ):
        with self._writer_patch as writer_cls_mock:
            setup = Setup(self.default_pf, "")
            # Just use the first member of the enum's name for testing
            field_str = next(iter(model_cls.__members__))
            writer_cls_mock.return_value.data = {
                model_name: mock.Mock(get_value=field_str)
            }

            setup.run()

        writer = writer_cls_mock.return_value
        call_args = writer.add_parameter.call_args_list
        assert mock.call(model_name, model_cls[field_str].value) in call_args


@pytest.mark.skipif(
    not process.ENABLED, reason="PROCESS is not installed on the system."
)
class TestSetupIntegration:
    @mock.patch(f"{MODULE_REF}.InDat")
    def test_obsolete_parameter_names_are_updated(self, writer_cls_mock):
        pf = ProcessSolverParams.from_json(PARAM_FILE)
        setup = Setup(pf, "")
        writer_cls_mock.return_value.data = {"x": 0}

        setup.run()

        writer = writer_cls_mock.return_value
        # 'dr_tf_case_out' is new name for 'casthi'
        assert mock.call("dr_tf_case_out", 0.04) in writer.add_parameter.call_args_list
