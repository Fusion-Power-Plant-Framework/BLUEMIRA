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
"""The API for the plasmod solver."""

from enum import auto
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
from scipy.interpolate import interp1d

from bluemira.base.parameter_frame import ParameterFrame
from bluemira.codes.error import CodesError
from bluemira.codes.interface import BaseRunMode, CodesSolver
from bluemira.codes.plasmod.api._outputs import PlasmodOutputs
from bluemira.codes.plasmod.api._run import Run
from bluemira.codes.plasmod.api._setup import Setup
from bluemira.codes.plasmod.api._teardown import Teardown
from bluemira.codes.plasmod.constants import BINARY as PLASMOD_BINARY
from bluemira.codes.plasmod.constants import NAME as PLASMOD_NAME
from bluemira.codes.plasmod.mapping import Profiles
from bluemira.codes.plasmod.params import PlasmodSolverParams


class RunMode(BaseRunMode):
    """
    RunModes for plasmod
    """

    RUN = auto()
    READ = auto()
    MOCK = auto()


class Solver(CodesSolver):
    """
    Plasmod solver class.

    Parameters
    ----------
    params:
        ParameterFrame for plasmod.
    build_config:
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

    name = PLASMOD_NAME
    setup_cls = Setup
    run_cls = Run
    teardown_cls = Teardown
    run_mode_cls = RunMode

    DEFAULT_INPUT_FILE = "plasmod_input.dat"
    DEFAULT_OUTPUT_FILE = "plasmod_output.dat"
    DEFAULT_PROFILES_FILE = "plasmod_profiles.dat"

    def __init__(
        self,
        params: Union[Dict, ParameterFrame],
        build_config: Optional[Dict[str, Any]] = None,
    ):
        # Init task objects on execution so parameters can be edited
        # between separate 'execute' calls.
        self._setup: Setup
        self._run: Run
        self._teardown: Teardown

        self.params = PlasmodSolverParams.from_defaults()

        if isinstance(params, ParameterFrame):
            self.params.update_from_frame(params)
        else:
            try:
                self.params.update_from_dict(params)
            except TypeError:
                self.params.update_values(params)

        self.build_config = {} if build_config is None else build_config
        self.binary = self.build_config.get("binary", PLASMOD_BINARY)
        self.problem_settings = self.build_config.get("problem_settings", {})
        self.input_file = self.build_config.get("input_file", self.DEFAULT_INPUT_FILE)
        self.output_file = self.build_config.get("output_file", self.DEFAULT_OUTPUT_FILE)
        self.profiles_file = self.build_config.get(
            "profiles_file", self.DEFAULT_PROFILES_FILE
        )
        self.run_directory = self.build_config.get(
            "run_directory", self.build_config.get("directory", "./")
        )
        self.read_directory = self.build_config.get(
            "read_directory", self.build_config.get("directory", "./")
        )

    def execute(self, run_mode: Union[str, RunMode]) -> ParameterFrame:
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
        run_mode:
            The mode to execute this solver in.
        """
        if isinstance(run_mode, str):
            run_mode = self.run_mode_cls.from_string(run_mode)

        self._setup = Setup(
            self.params, self.problem_settings, Path(self.run_directory, self.input_file)
        )
        self._run = Run(
            self.params,
            self.input_file,
            self.output_file,
            self.profiles_file,
            self.run_directory,
            self.binary,
        )
        self._teardown = Teardown(
            self.params,
            self.output_file,
            self.profiles_file,
            self.read_directory,
            self.run_directory,
        )
        if setup := self._get_execution_method(self._setup, run_mode):
            setup()
        if run := self._get_execution_method(self._run, run_mode):
            run()
        if teardown := self._get_execution_method(self._teardown, run_mode):
            teardown()

        self._scale_x_profile()

        return self.params

    def _scale_x_profile(self):
        self._x_phi = getattr(self.plasmod_outputs(), Profiles.x.name)
        self._x_phi /= np.max(self._x_phi)
        psi = getattr(self.plasmod_outputs(), Profiles.psi.name)
        self._x_psi = np.sqrt(psi / psi[-1])

    def _from_phi_to_psi(self, profile_data):
        """
        Convert the profile to the magnetic coordinate sqrt((psi - psi_ax)/(psi_b -
        psi_ax))
        """
        return interp1d(self._x_psi, profile_data, kind="linear")(self._x_phi)

    def get_profile(self, profile: Union[str, Profiles]) -> np.ndarray:
        """
        Get a single plasmod profile.

        Parameters
        ----------
        profile:
            A profile to get the data for.

        Returns
        -------
        A plasmod profile.

        Notes
        -----
        pprime and ffprime profiles from PLASMOD are currently inconsistent with the
        output jpar profile, even if isawt=FULLY_RELAXED. This is a known issue,
        and is under investigation. In the meantime, a crude rescaling of the flux
        functions is provided here.
        """
        if isinstance(profile, str):
            profile = Profiles(profile)

        if profile is Profiles.x:
            prof_data = self._x_phi
        else:
            prof_data = getattr(self.plasmod_outputs(), profile.name)
            prof_data = self._from_phi_to_psi(prof_data)

        return prof_data

    def get_profiles(
        self, profiles: Iterable[Union[str, Profiles]]
    ) -> Dict[Profiles, np.ndarray]:
        """
        Get a dictionary of plasmod profiles.

        Parameters
        ----------
        profiles:
            An iterable of Profiles enum values.

        Returns
        -------
        A dictionary mapping profile enum to values.
        """
        profiles_dict = {}
        for profile in profiles:
            profiles_dict[profile] = self.get_profile(profile)
        return profiles_dict

    def plasmod_outputs(self) -> PlasmodOutputs:
        """
        Return a structure of unmapped plasmod outputs.

        Use :code:`params` attribute for mapped outputs.

        Returns
        -------
        The scalar plasmod outputs.
        """
        try:
            return self._teardown.outputs
        except AttributeError as attr_error:
            raise CodesError(
                "Cannot get outputs before the solver has been run."
            ) from attr_error
