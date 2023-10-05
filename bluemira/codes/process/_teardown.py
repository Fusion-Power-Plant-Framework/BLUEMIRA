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
PROCESS teardown functions
"""

from pathlib import Path
from typing import Dict, Iterable, List, Union

import numpy as np

from bluemira.base.look_and_feel import bluemira_print, bluemira_warn
from bluemira.codes.error import CodesError
from bluemira.codes.interface import CodesTeardown
from bluemira.codes.process.api import MFile, update_obsolete_vars
from bluemira.codes.process.constants import NAME as PROCESS_NAME
from bluemira.codes.process.params import ProcessSolverParams
from bluemira.codes.utilities import read_mock_json_or_raise


class Teardown(CodesTeardown):
    """
    Teardown task for PROCESS solver.

    Parameters
    ----------
    params:
        The parameters for this task.
    run_directory:
        The directory in which to run PROCESS. Used in run, and runinput
        functions.
    read_directory:
        The directory to read PROCESS output files from. Used in read,
        readall, and mock functions.
    """

    params: ProcessSolverParams

    MOCK_JSON_NAME = "mockPROCESS.json"

    def __init__(
        self, params: ProcessSolverParams, run_directory: str, read_directory: str
    ):
        super().__init__(params, PROCESS_NAME)
        self.run_directory = run_directory
        self.read_directory = read_directory
        self._mfile_wrapper: _MFileWrapper = None

    def run(self):
        """
        Teardown the PROCESS solver.

        This loads the MFile in the run directory and maps its outputs
        to bluemira parameters.
        """
        self._load_mfile(Path(self.run_directory, "MFILE.DAT"), recv_all=False)

    def runinput(self):
        """
        Teardown the PROCESS solver.

        This loads the MFile in the run directory and maps its outputs
        to bluemira parameters.
        """
        self._load_mfile(Path(self.run_directory, "MFILE.DAT"), recv_all=True)

    def read(self):
        """
        Teardown the PROCESS solver.

        This loads the MFile in the run directory and maps its outputs
        to bluemira parameters.
        """
        self._load_mfile(Path(self.read_directory, "MFILE.DAT"), recv_all=False)

    def readall(self):
        """
        Teardown the PROCESS solver.

        This loads the MFile in the run directory and maps its outputs
        to bluemira parameters.
        """
        self._load_mfile(Path(self.read_directory, "MFILE.DAT"), recv_all=True)

    def mock(self):
        """
        Mock teardown the PROCESS solver.

        This loads a mockProcess.json file from the run directory and
        loads the values into this task's params.
        """
        bluemira_print("Mocking PROCESS systems code run")
        mock_file_path = Path(self.read_directory, self.MOCK_JSON_NAME)
        outputs = read_mock_json_or_raise(mock_file_path, self._name)
        self.params.update_values(outputs, source=self._name)

    def get_raw_outputs(self, params: Union[Iterable, str]) -> List[float]:
        """
        Get raw variables from an MFILE.

        Mapped bluemira parameters will have bluemira names.

        Parameters
        ----------
        params:
            Names of parameters to access.

        Returns
        -------
        The parameter values.
        """
        if not self._mfile_wrapper:
            raise CodesError(
                f"Cannot retrieve output from {self._name} MFile. "
                "The solver has not been run, so no MFile is available to read."
            )
        if isinstance(params, str):
            params = [params]
        outputs = []
        data = self._mfile_wrapper.data
        for param_name in params:
            if mapping := self.params.mappings.get(param_name, None):
                process_name = mapping.name
            else:
                process_name = param_name
            try:
                value = data[process_name]
            except KeyError:
                raise CodesError(
                    f"No {self._name} output, or bluemira parameter mapped to a"
                    f" {self._name} output, with name '{param_name}'."
                ) from None
            outputs.append(value)
        return outputs

    def _load_mfile(self, path: str, recv_all: bool):
        """
        Load the MFile at the given path, and update this object's
        params with the MFile's values.

        If recv_all, then ignore existing mappings and update all the
        params that correspond to a PROCESS output. If recv_all is
        False, then only update a parameter if its mapping has
        ``recv == True``.
        """
        mfile = self._read_mfile(path)
        self._update_params_with_outputs(mfile.data, recv_all)

    def _read_mfile(self, path: str):
        """
        Read an MFile, applying the given mappings, and performing unit
        conversions.
        """
        self._mfile_wrapper = _MFileWrapper(path, self._name)
        self._mfile_wrapper.read()
        return self._mfile_wrapper


class _MFileWrapper:
    """
    Utility class to wrap a PROCESS MFile, and map its data to bluemira
    parameters.

    Parameters
    ----------
    file_path:
        Path to an MFile.
    """

    def __init__(self, file_path: str, name: str = "PROCESS"):
        if not Path(file_path).is_file():
            raise CodesError(f"Path '{file_path}' is not a file.")
        self._name = name
        self.file_path = file_path
        self.mfile = MFile(file_path)
        _raise_on_infeasible_solution(self)
        self.data = {}

    def read(self) -> Dict:
        """
        Read the data from the PROCESS MFile.

        Store the result in ``data`` attribute.
        """
        self.data = {}
        for process_param_name, value in self.mfile.data.items():
            param_name = update_obsolete_vars(process_param_name)
            if param_name is None:
                bluemira_warn(
                    f"{self._name} parameter '{process_param_name}' is obsolete and has"
                    " no alternative. Setting value to NaN"
                )
                self.data[process_param_name] = np.nan
            elif isinstance(param_name, list):
                for name in param_name:
                    self.data[name] = value["scan01"]
            else:
                self.data[param_name] = value["scan01"]

        self.data.update(self._derive_radial_build_params(self.data))

    def _derive_radial_build_params(self, data: Dict) -> Dict[str, float]:
        """
        Derive radial build parameters that PROCESS does not directly calculate.

        Notes
        -----
        The PROCESS radial build is taken along the diagonal (maximum
        length) of the TF coil, so this must be taken into consideration
        when translating the geometry into the mid-plane.
        """
        try:
            shield_th = data["thshield"]
        except KeyError:
            # PROCESS updated their parameter names in v2.4.0, splitting
            # 'thshield' into 'thshield_ib', 'thshield_ob', and 'thshield_vb'
            shield_th = data["thshield_ib"] + data["thshield_ib"]

        try:
            rtfin = data["bore"] + data["ohcth"] + data["precomp"] + data["gapoh"]
            r_ts_ib_in = rtfin + data["tfcth"] + data["tftsgap"] + shield_th
            r_vv_ib_in = r_ts_ib_in + data["gapds"] + data["d_vv_in"] + data["shldith"]
            r_fw_ib_in = r_vv_ib_in + data["vvblgap"] + data["blnkith"] + data["fwith"]
            r_fw_ob_in = (
                r_fw_ib_in + data["scrapli"] + 2 * data["rminor"] + data["scraplo"]
            )
            r_vv_ob_in = r_fw_ob_in + data["fwoth"] + data["blnkoth"] + data["vvblgap"]
        except KeyError as key_error:
            raise CodesError(
                f"Missing PROCESS parameter in '{self.file_path}': {key_error}\n"
                "Cannot derive required bluemira parameters."
            ) from None
        return {
            "rtfin": rtfin,
            "r_ts_ib_in": r_ts_ib_in,
            "r_vv_ib_in": r_vv_ib_in,
            "r_fw_ib_in": r_fw_ib_in,
            "r_fw_ob_in": r_fw_ob_in,
            "r_vv_ob_in": r_vv_ob_in,
        }


def _raise_on_infeasible_solution(m_file: _MFileWrapper):
    """
    Check that PROCESS found a feasible solution.

    Parameters
    ----------
    m_file:
        The PROCESS MFILE to check for a feasible solution

    Raises
    ------
    CodesError
        If a feasible solution was not found.
    """
    error_code = int(m_file.mfile.data["ifail"]["scan01"])
    if error_code != 1:
        message = (
            f"{m_file._name} did not find a feasible solution. ifail = {error_code}."
            " Check PROCESS logs."
        )
        raise CodesError(message)
