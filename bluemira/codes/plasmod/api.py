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
API for the transport code PLASMOD and related functions
"""

from enum import auto
from typing import Any, Callable, Dict, Optional, Union

from bluemira.base.constants import raw_uc
from bluemira.base.look_and_feel import bluemira_debug, bluemira_warn
from bluemira.base.parameter import ParameterFrame
from bluemira.base.solver import RunMode as _RunMode
from bluemira.base.solver import SolverABC, Task
from bluemira.codes.error import CodesError
from bluemira.codes.plasmod.constants import BINARY as PLASMOD_BINARY
from bluemira.codes.plasmod.constants import NAME as PLASMOD_NAME
from bluemira.codes.plasmod.mapping import mappings as plasmod_mappings
from bluemira.codes.plasmod.params import PlasmodInputs, PlasmodOutputs
from bluemira.codes.utilities import (
    add_mapping,
    get_recv_mapping,
    get_send_mapping,
    run_subprocess,
)


class RunMode(_RunMode):
    """
    RunModes for plasmod
    """

    RUN = auto()
    READ = auto()
    MOCK = auto()


class PlasmodTask(Task):
    """
    A task related to plasmod.

    This adds plasmod parameter mappings to the input ParameterFrame.
    """

    def __init__(self, params: ParameterFrame) -> None:
        super().__init__(params)
        add_mapping(PLASMOD_NAME, self._params, plasmod_mappings)


class Setup(PlasmodTask):
    """
    Setup task for a plasmod solver.

    On run, this task writes a plasmod input file using the input values
    defined in this class.
    """

    def __init__(
        self,
        params: ParameterFrame,
        problem_settings: Dict[str, Any],
        # TODO(hsaunders1904): rename this; input_file is confusing,
        # because it's actually an output file of this task
        input_file: str,
    ) -> None:
        super().__init__(params)

        self.inputs = PlasmodInputs()
        self.input_file = input_file

        self.update_inputs(problem_settings)

    @property
    def params(self) -> ParameterFrame:
        """Return the parameters associated with this task."""
        return self._params

    def run(self):
        """
        Run plasmod setup.
        """
        self._write_input()

    def mock(self):
        """
        Run plasmod setup in mock mode.

        No need to generate an input file as results will be mocked.
        """
        pass

    def read(self):
        """
        Run plasmod setup in read mode.

        No need to generate an input file as results will be read from
        file.
        """
        pass

    def update_inputs(self, new_inputs: Dict[str, Any] = None):
        """
        Update plasmod inputs using the given values.
        """
        # Create a new PlasmodInputs objects so we still benefit from
        # the __post_init__ processing (converts models to enums)
        new_inputs = {} if new_inputs is None else new_inputs
        new = self.get_new_inputs()
        new.update(new_inputs)
        self.inputs = PlasmodInputs(**new)

    def _write_input(self):
        """
        Write inputs to file to be read by plasmod.
        """
        try:
            with open(self.input_file, "w") as io_stream:
                self.inputs.write(io_stream)
        except OSError as os_error:
            raise CodesError(
                f"Could not write plasmod input file: '{self.input_file}': {os_error}"
            ) from os_error

    def get_new_inputs(self, remapper: Optional[Union[Callable, Dict]] = None):
        """
        Get new key mappings from the ParameterFrame.

        Parameters
        ----------
        remapper: Optional[Union[callable, dict]]
            a function or dictionary for remapping variable names.
            Useful for renaming old variables

        Returns
        -------
        _inputs: dict
            key value pairs of external program variable names and values

        TODO unit conversion
        """
        # TODO(hsaunders): refactor out to base class, make private?
        _inputs = {}

        if not (callable(remapper) or isinstance(remapper, (type(None), Dict))):
            raise TypeError("remapper is not callable or a dictionary")
        elif isinstance(remapper, Dict):
            orig_remap = remapper.copy()

            def remapper(x):
                return orig_remap[x]

        elif remapper is None:

            def remapper(x):
                return x

        for prog_key, bm_key in self._send_mapping.items():
            prog_key = remapper(prog_key)
            if isinstance(prog_key, list):
                for key in prog_key:
                    _inputs[key] = self._convert_units(self.params.get_param(bm_key))
                continue

            _inputs[prog_key] = self._convert_units(self.params.get_param(bm_key))

        return _inputs

    def _convert_units(self, param):
        code_unit = param.mapping[PLASMOD_NAME].unit
        if code_unit is not None:
            return raw_uc(param.value, param.unit, code_unit)
        else:
            return param.value

    @property
    def _send_mapping(self) -> Dict[str, str]:
        self.__send_mapping = get_send_mapping(self.params, PLASMOD_NAME)
        return self.__send_mapping


class Run(PlasmodTask):
    """
    Run class for plasmod transport solver.
    """

    def __init__(
        self,
        params: ParameterFrame,
        input_file: str,
        output_file: str,
        profiles_file: str,
        binary=PLASMOD_BINARY,
    ):
        super().__init__(params)
        self.binary = binary
        self.input_file = input_file
        self.output_file = output_file
        self.profiles_file = profiles_file

    def run(self):
        """
        Run the plasmod shell task.

        Runs plasmod on the command line using the given input files and
        output path.

        Raises
        ------
        CodesError
            If the subprocess returns a non-zero exit code or raises an
            OSError (e.g., the plasmod binary does not exist).
        """
        command = [self.binary, self.input_file, self.output_file, self.profiles_file]
        bluemira_debug("Mode: run")
        try:
            self._run_subprocess(command)
        except OSError as os_error:
            raise CodesError(f"Failed to run plasmod: {os_error}") from os_error

    def _run_subprocess(self, command, **kwargs):
        """
        Run a subprocess command and raise CodesError if it returns a
        non-zero exit code.
        """
        return_code = run_subprocess(command, **kwargs)
        if return_code != 0:
            raise CodesError("plasmod 'Run' task exited with a non-zero error code.")


class Teardown(PlasmodTask):
    """
    Plasmod teardown task.

    In "RUN" and "READ" mode, this loads in plasmod results files and
    updates :code:`params` with the values.
    """

    def __init__(self, params: ParameterFrame, output_file: str, profiles_file: str):
        super().__init__(params)
        self.outputs = PlasmodOutputs()
        self.output_file = output_file
        self.profiles_file = profiles_file

    def run(self):
        """
        Load the plasmod results files and update this object's params
        with the read values.
        """
        self.read()

    def mock(self):
        """
        Update this object's plasmod params with default values.
        """
        self.outputs = PlasmodOutputs()
        self._update_params_from_outputs()

    def read(self):
        """
        Load the plasmod results files and update this object's params
        with the read values.

        Raises
        ------
        CodesError
            If any of the plasmod files cannot be opened.
        """
        try:
            with open(self.output_file, "r") as scalar_file:
                with open(self.profiles_file, "r") as profiles_file:
                    self.outputs = PlasmodOutputs.from_files(scalar_file, profiles_file)
        except OSError as os_error:
            raise CodesError(
                f"Could not read plasmod output file: {os_error}."
            ) from os_error
        self._update_params_from_outputs()

    @property
    def params(self) -> ParameterFrame:
        """Return the Bluemira parameters associated with this task."""
        return self._params

    def _update_params_from_outputs(self):
        """
        Update this object's ParameterFrame with plasmod outputs.
        """
        bm_outputs = self._map_outputs_to_bluemira()
        self._prepare_outputs(bm_outputs, source=PLASMOD_NAME)

    def _map_outputs_to_bluemira(self) -> Dict[str, Any]:
        """
        Iterate over the plasmod-bluemira parameter mappings and map the
        bluemira parameter names to plasmod output values.
        """
        bm_outputs: Dict[str, Any] = {}
        for plasmod_key, bm_key in self._recv_mapping.items():
            try:
                output_value = getattr(self.outputs, plasmod_key)
            except AttributeError as attr_error:
                raise CodesError(
                    f"No plasmod output '{plasmod_key}' in plasmod outputs list."
                ) from attr_error
            if output_value is None:
                # Catches cases where parameters may be missing from the
                # output file, in which case we get the default, which
                # can be None.
                bluemira_warn(
                    f"No value for plasmod parameter '{bm_key}' found in output."
                )
            else:
                bm_outputs[bm_key] = output_value
        return bm_outputs

    @property
    def _recv_mapping(self):
        """Return the plasmod-to-bluemira parameter mappings."""
        self.__recv_mapping = get_recv_mapping(self.params, PLASMOD_NAME)
        return self.__recv_mapping

    def _prepare_outputs(self, bm_outputs: Dict[str, Any], source: str):
        """
        Update this object's ParameterFrame with the given outputs.

        Implicitly converts to bluemira units if unit available.

        Parameters
        ----------
        outputs: Dict
            key value pair of code outputs
        source: Optional[str]
            Set the source of all outputs, by default is code name

        """
        for bm_key, value in bm_outputs.items():
            try:
                code_unit = self.params.get_param(bm_key).mapping[PLASMOD_NAME].unit
            except AttributeError as exc:
                raise CodesError(f"No mapping found for '{bm_key}'.") from exc
            if code_unit is not None:
                bm_outputs[bm_key] = {"value": value, "unit": code_unit}

        self.params.update_kw_parameters(bm_outputs, source=source)


class Solver(SolverABC):
    """
    Plasmod solver class.

    Parameters
    ----------
    params: ParameterFrame
        ParameterFrame for plasmod.
    build_config: Dict[str, Any]
        Build configuration dictionary.
        Expected keys include:
            - binary: str, path to the plasmod binary.
            - problem_settings: Dict[str, Any], any plasmod specific
              parameters (i.e., parameters that bluemira does not have
              direct mappings to through the ParameterFrame).
            - input_file: str, the path to write the plasmod input file
              to, this can be a relative path.
            - output_file: str, the path to write the plasmod scalar
              output file to.
            - profiles_file: str, the path to write the plasmod profiles
              output file to.
    """

    setup_cls = Setup
    run_cls = Run
    teardown_cls = Teardown

    DEFAULT_INPUT_FILE = "plasmod_input.dat"
    DEFAULT_OUTPUT_FILE = "plasmod_output.dat"
    DEFAULT_PROFILES_FILE = "plasmod_profiles.dat"

    def __init__(self, params: ParameterFrame, build_config: Dict[str, Any] = None):
        self.params = params
        self.build_config = {} if build_config is None else build_config

        self.binary = self.build_config.get("binary", PLASMOD_BINARY)
        self.problem_settings = self.build_config.get("problem_settings", {})
        self.input_file = self.build_config.get("input_file", self.DEFAULT_INPUT_FILE)
        self.output_file = self.build_config.get("output_file", self.DEFAULT_OUTPUT_FILE)
        self.profiles_file = self.build_config.get(
            "profiles_file", self.DEFAULT_PROFILES_FILE
        )

        # TODO(hsaunders): sanity check file paths are not equal?

        self._setup = Setup(self.params, self.problem_settings, self.input_file)
        self._run = Run(
            self.params,
            self.input_file,
            self.output_file,
            self.profiles_file,
            self.binary,
        )
        self._teardown = Teardown(self.params, self.output_file, self.profiles_file)

    def execute(self, run_mode: RunMode) -> ParameterFrame:
        """
        Execute this plasmod solver.

        This solver:
            1. writes a plasmod input file using the given bluemira and
               problem parameters.
            2. processes that file using a shell call to plasmod.
            3. reads the plasmod output files, and updates this object's
               ParameterFrame with the results.

        Parameters
        ----------
        run_mode: RunMode
            The mode to execute this solver in.
        """
        setup = self._get_execution_method(self._setup, run_mode)
        run = self._get_execution_method(self._run, run_mode)
        teardown = self._get_execution_method(self._teardown, run_mode)

        if setup:
            setup()
        if run:
            run()
        if teardown:
            teardown()

        return self.params
