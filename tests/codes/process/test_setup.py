# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from unittest import mock

import pytest

from bluemira.codes import process
from bluemira.codes.params import ParameterMapping
from bluemira.codes.process._setup import Setup
from bluemira.codes.process.mapping import mappings as process_mappings
from bluemira.codes.process.params import ProcessSolverParams
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
    def test_obsolete_parameter_names_are_updated(self, writer_cls_mock, caplog):
        pf = ProcessSolverParams.from_json(PARAM_FILE)
        pf.mappings["tk_tf_front_ib"] = ParameterMapping(
            "dr_tf_case_out", send=True, recv=False, unit="m"
        )
        setup = Setup(pf, "")
        writer_cls_mock.return_value.data = {"x": 0}

        setup.run()
        # 'dr_tf_case_out' is old name for 'casthi'
        assert "Obsolete dr_tf_case_out" in caplog.records[0].message
        assert "name is casthi" in caplog.records[0].message

        writer = writer_cls_mock.return_value
        assert mock.call("casthi", 0.04) in writer.add_parameter.call_args_list
