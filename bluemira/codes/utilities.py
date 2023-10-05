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
Utility functions for interacting with external codes
"""

import json
import os
import subprocess
import threading
from enum import Enum
from types import ModuleType
from typing import Any, Dict, List

from bluemira.base.look_and_feel import (
    _bluemira_clean_flush,
    bluemira_error_clean,
    bluemira_print,
    bluemira_print_clean,
)
from bluemira.codes.error import CodesError
from bluemira.codes.params import ParameterMapping
from bluemira.utilities.tools import get_module


class Model(Enum):
    """
    Base Model Enum
    """

    @classmethod
    def info(cls):
        """
        Show Model options
        """
        infostr = f"{cls.__doc__}\n" + "\n".join(repr(l_) for l_ in list(cls))
        bluemira_print(infostr)


def read_mock_json_or_raise(file_path: str, name: str) -> Dict[str, float]:
    """
    Read json file or raise CodesError
    """
    try:
        with open(file_path) as f:
            return json.load(f)
    except OSError as os_error:
        raise CodesError(
            f"Cannot open mock {name} results file '{file_path}'."
        ) from os_error


def get_code_interface(module: str) -> ModuleType:
    """
    Dynamically import code interface

    Parameters
    ----------
    module:
        module to import

    Returns
    -------
    code module
    """
    try:
        return get_module(f"bluemira.codes.{module.lower()}")
    except ImportError:
        return get_module(module)


def create_mapping(
    in_mappings=None, out_mappings=None, io_mappings=None, none_mappings=None
) -> Dict[str, Any]:
    """
    Creates mappings for external codes

    Returns
    -------
    A mapping from bluemira names to an external code ParameterMapping

    """
    mappings = {}
    ins = {"send": True, "recv": False}
    outs = {"send": False, "recv": True}
    inouts = {"send": True, "recv": True}
    nones = {"send": False, "recv": False}

    for puts, sr in [
        [in_mappings, ins],
        [out_mappings, outs],
        [io_mappings, inouts],
        [none_mappings, nones],
    ]:
        if puts is not None:
            for bm_key, (ec_key, unit) in puts.items():
                mappings[bm_key] = ParameterMapping(
                    ec_key, send=sr["send"], recv=sr["recv"], unit=unit
                )

    return mappings


class LogPipe(threading.Thread):
    """
    Capture logs for subprocesses

    https://codereview.stackexchange.com/questions/6567/redirecting-subprocesses-output-stdout-and-stderr-to-the-logging-module

    Parameters
    ----------
    loglevel:
        print or error flush printing

    """

    def __init__(self, loglevel: str):
        super().__init__(daemon=True)

        self.logfunc = {"print": bluemira_print_clean, "error": bluemira_error_clean}[
            loglevel
        ]
        self.logfunc_flush = _bluemira_clean_flush
        self.fd_read, self.fd_write = os.pipe()
        self.pipe = os.fdopen(self.fd_read, encoding="utf-8", errors="ignore")
        self.start()

    def fileno(self):
        """
        Return the write file descriptor of the pipe
        """
        return self.fd_write

    def run(self):
        """
        Run the thread and pipe it all into the logger.
        """
        for line in iter(self.pipe.readline, ""):
            if line.startswith("==>"):
                self.logfunc_flush(line.strip("\n"))
            else:
                self.logfunc(line)

        self.pipe.close()

    def close(self):
        """
        Close the write end of the pipe.
        """
        os.close(self.fd_write)


def run_subprocess(command: List[str], run_directory: str = ".", **kwargs) -> int:
    """
    Run a subprocess terminal command piping the output into bluemira's
    logs.

    Parameters
    ----------
    command:
        The arguments of the command to run.
    run_directory:
        The directory to run the command in. Default is current working
        directory.
    **kwargs:
        Arguments passed directly to subprocess.Popen.

    Returns
    -------
    return_code: int
        The return code of the subprocess.
    """
    stdout = LogPipe("print")
    stderr = LogPipe("error")

    kwargs["cwd"] = run_directory
    kwargs.pop("shell", None)  # Protect against user input

    with subprocess.Popen(
        command, stdout=stdout, stderr=stderr, shell=False, **kwargs  # noqa: S603
    ) as s:
        stdout.close()
        stderr.close()

    return s.returncode
