# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
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
"""
External Codes Example
"""
# %%
from dataclasses import asdict, dataclass, fields
from enum import auto
from pathlib import Path
from pprint import pprint
from typing import ClassVar, Dict, List, Union

from ext_code_script import get_filename

from bluemira.base.file import get_bluemira_path
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.codes.interface import (
    BaseRunMode,
    CodesSetup,
    CodesSolver,
    CodesTask,
    CodesTeardown,
)
from bluemira.codes.params import MappedParameterFrame
from bluemira.codes.utilities import ParameterMapping

# %% [markdown]
# # External Code Wrapping
#
# This example goes though the minimal steps taken to wrap an external code and
# retrieve its outputs.
#
# Firstly we define the options available in the code and its name.
# In this example we're wrapping the python script
# [ext_code_script.py](ext_code_script.py).
# The script reads in a file and writes out to a different file with
# two possible modifications:
#   * Add a header
#   * Add line numbers
#
# Unusually the BINARY global variable is a list instead of a string because there are
# two base commands to run the script from the command line. We have used a little hack
# with the `get_filename` function so you don't have to find the external code script
# file location.
# Usually the program is in your PATH or provided by the config.
#
# The program has been named "External Code" for simplicity and is used to help the
# user trace where variables come from. It is a simple script that slightly modifies
# a file provided to it.
#
# There are 3 dataclasses containing the command line options for the code,
# the input parameters and the output parameters.

# %%
PRG_NAME = "External Code"
BINARY = ["python", get_filename()]


@dataclass
class ECOpts:
    """External Code Options"""

    add_header: bool = False
    number: bool = False

    def to_list(self) -> List:
        """Options list"""
        return [f"--{k.replace('_', '-')}" for k, v in asdict(self).items() if v]


@dataclass
class ECInputs:
    """External Code Inputs"""

    param1: float = None
    param2: float = 6


@dataclass
class ECOutputs:
    """External Code Outputs"""

    param1: float = None
    param2: float = None


# %% [markdown]
# ## Linking the code to bluemira
#
# To link an external code to bluemira you need a few bits of machinery
#
# * A `MappedParameterFrame` that links bluemira parameter names and units to the
#   external code
# * A `RunMode` class to specify the possibly running modes
# * A `Solver` that orchestrates the running of the code
# * Some task, or tasks, for the `Solver` to run. Typically there are three:
#     * A `Setup` task which writes the input file for the code
#     * A `Run` task which runs the code
#     * A `Teardown` task which reads the output file of the code
#
# The `MappedParameterFrame` here gets the defaults and sets the mappings.
# Notice that "param2" is sent to the code but "param1" is not.


# %%
@dataclass
class ECParameterFrame(MappedParameterFrame):
    """External Code ParameterFrame"""

    header: Parameter[bool]
    line_number: Parameter[bool]
    param1: Parameter[float]
    param2: Parameter[float]

    _defaults = (ECOpts(), ECInputs())

    _mappings: ClassVar = {
        "header": ParameterMapping("add_header", send=True, recv=False),
        "line_number": ParameterMapping("number", send=True, recv=False),
        "param1": ParameterMapping("param1", send=False, recv=True, unit="MW"),
        "param2": ParameterMapping("param2", send=True, recv=True, unit="GW"),
    }

    @property
    def mappings(self) -> Dict:
        """Code Mappings"""
        return self._mappings

    @classmethod
    def from_defaults(cls) -> MappedParameterFrame:
        """Setup from defaults"""
        dd = {}
        for _def in cls._defaults:
            dd = {**dd, **asdict(_def)}
        return super().from_defaults(dd)


# %%
class RunMode(BaseRunMode):
    """
    RunModes for external code
    """

    RUN = auto()
    READ = auto()
    MOCK = auto()


# %% [markdown]
# The `Setup` class pulls over the inputs from bluemira as described by the
# mapping and creates the input file. The output of the run method returns
# the command line options list.


# %%
class Setup(CodesSetup):
    """Setup task"""

    params: ECParameterFrame

    def __init__(self, params: ParameterFrame, problem_settings: Dict, infile: str):
        super().__init__(params, PRG_NAME)
        self.problem_settings = problem_settings
        self.infile = infile

    def update_inputs(self) -> Dict:
        """Update inputs from bluemira"""
        self.inputs = ECInputs()
        self.options = ECOpts()
        inp = self._get_new_inputs()

        # Get inputs
        for code_input in (self.problem_settings, inp):
            for k, v in code_input.items():
                if k in fields(self.inputs) and v is not None:
                    setattr(self.inputs, k, v)

        # Get options
        for code_input in (self.problem_settings, inp):
            for k, v in code_input.items():
                if k in fields(self.options) and v:
                    setattr(self.options, k, v)

        # Protects against writing default values if unset
        return {k: v for k, v in asdict(self.inputs).items() if v}

    def run(self) -> List:
        """Run mode"""
        inp = self.update_inputs()
        with open(self.infile, "w") as input_file:
            for k, v in inp.items():
                input_file.write(f"{k}  {v}\n")
        return self.options.to_list()


# %% [markdown]
# `Run` simply runs the code in a subprocess with the given options.


# %%
class Run(CodesTask):
    """Run task"""

    def __init__(
        self, params: ParameterFrame, infile: str, outfile: str, binary: List = BINARY
    ):
        super().__init__(params, PRG_NAME)
        self.binary = binary
        self.infile = infile
        self.outfile = outfile

    def run(self, options: List):
        """Run mode"""
        self._run_subprocess([*self.binary, *options, self.infile, self.outfile])


# %% [markdown]
# `Teardown` reads in a given output file or, in the case of mock, returns a known
# value, sending the new parameter values back to the `ParameterFrame`.


# %%
class Teardown(CodesTeardown):
    """Teardown task"""

    def __init__(self, params: ParameterFrame, outfile: str):
        super().__init__(params, PRG_NAME)
        self.outfile = outfile

    def _read_file(self):
        out_params = {}
        with open(self.outfile) as output_file:
            for line in output_file:
                if line.startswith("#"):
                    pass
                if line.startswith(" "):
                    k, v = line.split()
                    out_params[k] = float(v)
        self._update_params_with_outputs(out_params)

    def run(self) -> ParameterFrame:
        """Run mode"""
        self._read_file()
        return self.params

    def read(self) -> ParameterFrame:
        """Read mode"""
        self._read_file()
        return self.params

    def mock(self) -> ParameterFrame:
        """Mock mode"""
        self._update_params_with_outputs({"param1": 15})
        return self.params


# %% [markdown]
# `Solver` combines the three tasks into one object for execution.
# The execute method has been overridden here for our use-case and returns
# the `ParameterFrame`.


# %%
class Solver(CodesSolver):
    """The External Code Solver."""

    name = PRG_NAME
    params = ECParameterFrame
    setup_cls = Setup
    run_cls = Run
    teardown_cls = Teardown
    run_mode_cls = RunMode

    def __init__(self, params: Union[ParameterFrame, Dict], build_config: Dict):
        self.params = ECParameterFrame.from_defaults()
        self.params.update(params)

        self._setup = self.setup_cls(
            self.params, build_config.get("problem_settings", {}), build_config["infile"]
        )
        self._run = self.run_cls(
            self.params,
            build_config["infile"],
            build_config["outfile"],
            build_config.get("binary", BINARY),
        )
        self._teardown = self.teardown_cls(self.params, build_config["outfile"])

    def execute(self, run_mode: Union[str, RunMode]) -> ParameterFrame:
        """Execute the solver"""
        if isinstance(run_mode, str):
            run_mode = self.run_mode_cls.from_string(run_mode)
        result = None
        if setup := self._get_execution_method(self._setup, run_mode):
            result = setup()
        if run := self._get_execution_method(self._run, run_mode):
            run(result)
        if teardown := self._get_execution_method(self._teardown, run_mode):
            result = teardown()
        return result


# %% [markdown]
# ### Using the solver
#
# To run the solver you just need to provide the parameters and the configuration
# to initialise the object.
# Be aware `problem_settings` should be used sparingly for options that won't
# change within the rerunning of the solver,
# it has the same effect as modifying the default.
# Also note the units of the parameter values returned back have been updated.
#
# The files written and read by the external code are stored in the generated_data
# folder in the root of the bluemira repository.
#
# Some warnings will be shown because some of the situations here are usually
# undesirable.
# In this first block we will see 3 warnings.
# The first 2 are the same for the `run` and `read` modes:
#   * "No value for param1"
#       - param 1 has its send mapping set to `False` so no value is sent to
#         the code and therefore it is not read back in. As `read` mode is executed
#         with the output of the `run` mode the same error is repeated.
#
# The 3rd is for the `mock` mode:
#   * "No value for param2"
#       - The mock output doesn't have a param2 output
#
# Notice that `param2` does not take the value given in problems settings as we
# overwrite it because `solver.params.mappings['param2'].send == True`,
# set in the `ECParameterFrame` default mappings.

# %%
io_path = get_bluemira_path("", "generated_data")

params = {
    "header": {"value": False, "unit": "", "source": "here"},
    "param1": {"value": 5, "unit": "W"},
}

build_config = {
    "problem_settings": {"param2": 10},
    "infile": Path(io_path, "infile.txt"),
    "outfile": Path(io_path, "outfile.txt"),
}
solver = Solver(params, build_config)

# Running in all the different modes
for mode in ["run", "read", "mock"]:
    print(mode)
    out_params = solver.execute(mode)
    print(out_params)

# %% [markdown]
# 1 warning this time, we still haven't sent a value for `param1` in `run` mode.
# Notice how the default for `param2` from our `problem_settings` is used when we
# turn off the send mapping.

# %%
solver.modify_mappings({"param2": {"send": False}})
print(solver.execute("run"))

# %% [markdown]
# Again the same warning. This time we have modified the value of `param2` and turned
# the send mapping back on.

# %%
# problem_settings param2 overridden
solver.modify_mappings({"param2": {"send": True}})
solver.params.param2.value = 5
print(solver.execute("run"))

# %% [markdown]
# No warnings this time, we have now set a value for `param1` and sent it.

# %%
solver.modify_mappings({"param1": {"send": True}})
solver.params.param1.value = 5e3
print(solver.execute("run"))

# %% [markdown]
# Turning on the header option (that output wont change) but the source has changed
# because we haven't updated it.

# %%
solver.params.header.value = True
print(solver.execute("run"))

# %% [markdown]
# Now we set the `param2` source and only send and not receive the result.

# %%
solver.modify_mappings({"param2": {"recv": False}})
solver.params.param2.set_value(9, "param2 sent with new value")
print(solver.execute("run"))

# %% [markdown]
# Now we can show all the changes to the parameters during the solver runs.
# All Parameters have an associated history.

# %%
for param in solver.params:
    print(f"{param.name}\n{'-' * len(param.name)}")
    pprint(param.history())  # noqa: T203
    print()
