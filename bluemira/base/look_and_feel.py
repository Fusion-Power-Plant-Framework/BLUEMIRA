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
Aesthetic and ambiance functions.
"""

import logging
import os
import platform
import shutil
import subprocess
from getpass import getuser
from pathlib import Path
from textwrap import dedent, wrap
from typing import Callable, Dict, List, Optional

from bluemira import __version__
from bluemira.base.constants import ANSI_COLOR, EXIT_COLOR
from bluemira.base.file import get_bluemira_path, get_bluemira_root
from bluemira.base.logs import logger_setup

LOGGER = logger_setup()

# Calculate the number of lines in this file
try:
    with open(
        Path(get_bluemira_path("base"), "look_and_feel.py"),
        encoding="utf-8",
    ) as f:
        LOCAL_LINES = len(f.read().splitlines())
except FileNotFoundError:
    # Approximately
    LOCAL_LINES = 550

# =============================================================================
# Getters for miscellaneous information
# =============================================================================


def get_git_version(directory: str) -> str:
    """
    Get the version string of the current git branch, e.g.: '0.0.3-74-g70d48be'.

    Parameters
    ----------
    directory:
        The full path directory of the folder to get git information from

    Returns
    -------
    The git version bytestring
    """
    return subprocess.check_output(
        ["git", "describe", "--tags", "--always"], cwd=directory  # noqa: S603, S607
    ).strip()


def get_git_branch(directory: str) -> str:
    """
    Get the name of the current git branch, e.g. 'develop'.

    Parameters
    ----------
    directory:
        The full path directory of the folder to get git information from

    Returns
    -------
    The git branch string
    """
    return (
        subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],  # noqa: S603, S607
            cwd=directory,
        )
        .strip()
        .decode("utf-8")
    )


def get_git_files(directory: str, branch: str) -> List[str]:
    """
    Get the names of the files in the directory of the specified branch name.

    Parameters
    ----------
    directory:
        The full path directory of the folder to get git information from
    branch:
        The name of the git branch to retrieve the filenames from

    Returns
    -------
    The list of git-controlled path strings
    """
    return (
        subprocess.check_output(
            ["git", "ls-tree", "-r", branch, "--name-only"],  # noqa: S603, S607
            cwd=directory,
        )
        .decode("utf-8")
        .splitlines()
    )


def get_platform() -> str:
    """
    Get the OS platform.

    Returns
    -------
    The generic name of the platform (e.g. Linux, Windows)
    """
    return platform.uname()[0]


def count_slocs(
    directory: str,
    branch: str,
    exts: Optional[List[str]] = None,
    ignore: Optional[List[str]] = None,
) -> Dict:
    """
    Counts lines of code within a given directory for a given git branch

    Parameters
    ----------
    directory:
        The full path directory of the folder to get git information from
    branch:
        The git branch string
    exts:
        The list of file extensions to search the directory for
    ignore:
        The list of extensions and filenames to ignore

    Returns
    -------
    The dictionary of number of lines of code per file extension, and the
    total linecount
    """
    if ignore is None:
        ignore = [".git", ".txt", "look_and_feel.py"]

    if exts is None:
        exts = [".py"]

    lines = {}
    for k in exts:
        lines[k] = 0
    files = get_git_files(directory, branch)
    for name in files:
        if Path(name).parts[-1] not in ignore and name not in ignore:
            for e in exts:
                if name.endswith(e):
                    path = Path(directory, name)
                    try:
                        with open(path, encoding="utf-8") as file:
                            lines[e] += len(file.read().splitlines())

                    except FileNotFoundError:
                        bluemira_warn(
                            "count_slocs: Probably not on the right git branch"
                        )
                        continue

    lines[".py"] += LOCAL_LINES
    lines["total"] = sum([lines[k] for k in lines])
    return lines


# =============================================================================
# Printing functions
# =============================================================================


def _print_color(string: str, color: str) -> str:
    """
    Create text to print. NOTE: Does not call print command

    Parameters
    ----------
    string:
        The text to colour
    color:
        The color to make the color-string for

    Returns
    -------
    The string with ANSI color decoration
    """
    return f"{ANSI_COLOR[color]}{string}{EXIT_COLOR}"


def _bm_print(string: str, width: int = 73):
    """
    Create the text string for boxed text to print to the console.

    Parameters
    ----------
    string:
        The string of text to colour and box
    width:
        The width of the box, default = 73 (leave this alone for best results)

    Returns
    -------
    The text string of the boxed text
    """
    strings = [
        " " if s == "\n" and i != 0 else s[:-1] if s.endswith("\n") else s
        for i, s in enumerate(string.splitlines(keepends=True))
    ]
    bw = width - 4
    t = [
        wrap(s, width=bw, replace_whitespace=False, drop_whitespace=False)
        for s in strings
    ]

    s = [dedent(item) for sublist in t for item in sublist]
    lines = ["".join(["| "] + [i] + [" "] * (width - 2 - len(i)) + [" |"]) for i in s]
    h = "".join(["+", "-" * width, "+"])
    return h + "\n" + "\n".join(lines) + "\n" + h


def colourise(string: str, width: int = 73, color: str = "blue") -> str:
    """
    Print coloured, boxed text to the console. Default template for bluemira
    information.

    Parameters
    ----------
    string:
        The string of text to colour and box
    width:
        The width of the box, default = 73 (leave this alone for best results)
    color:
        The color to print the text in from `bluemira.base.constants.ANSI_COLOR`
    """
    text = _bm_print(string, width=width)
    return _print_color(text, color)


def bluemira_critical(string: str):
    """
    Standard template for bluemira critical errors.
    """
    return LOGGER.critical(colourise(f"CRITICAL: {string}", color="darkred"))


def bluemira_error(string: str):
    """
    Standard template for bluemira errors.
    """
    return LOGGER.error(colourise(f"ERROR: {string}", color="red"))


def bluemira_warn(string: str):
    """
    Standard template for bluemira warnings.
    """
    return LOGGER.warning(colourise(f"WARNING: {string}", color="orange"))


def bluemira_print(string: str):
    """
    Standard template for bluemira information messages.
    """
    return LOGGER.info(colourise(string, color="blue"))


def bluemira_debug(string: str):
    """
    Standard template for bluemira debugging.
    """
    return LOGGER.debug(colourise(string, color="green"))


def _bm_print_singleflush(string: str, width: int = 73, color: str = "blue"):
    """
    Create the text string for coloured, boxed text to flush print to the
    console.

    Parameters
    ----------
    string:
        The string of text to colour and box
    width:
        The width of the box, default = 73 (leave this alone for best results)
    color:
        The color to print the text in, one of ['blue', 'red', 'green']

    Returns
    -------
        The text string of the boxed coloured text to flush print
    """
    a = width - len(string) - 2
    text = "| " + string + a * " " + " |"
    return _print_color(text, color)


def _bluemira_clean_flush(string, func: Callable[[str], None] = LOGGER.info):
    """
    Print and flush string. Useful for updating information.

    Parameters
    ----------
    string:
        The string to colour flush print
    func:
        The function to use for logging, by default LOGGER.info
    """
    _terminator_handler(func, "\r" + string, fhterm=logging.StreamHandler.terminator)


def _terminator_handler(func: Callable[[str], None], string: str, *, fhterm: str = ""):
    """
    Log string allowing modification to handler terminator

    Parameters
    ----------
    func:
        The function to use for logging (e.g LOGGER.info)
    string:
        The string to colour flush print
    fhterm:
        FileHandler Terminator
    """
    original_terminator = logging.StreamHandler.terminator
    logging.StreamHandler.terminator = ""
    logging.FileHandler.terminator = fhterm
    try:
        func(string)
    finally:
        logging.StreamHandler.terminator = original_terminator
        logging.FileHandler.terminator = original_terminator


def bluemira_print_flush(string: str):
    """
    Print a coloured, boxed line to the console and flushes it. Useful for
    updating information.

    Parameters
    ----------
    string:
        The string to colour flush print
    """
    _bluemira_clean_flush(_bm_print_singleflush(string), func=LOGGER.info)


def bluemira_debug_flush(string: str):
    """
    Print a coloured, boxed line to the console and flushes it. Useful for
    updating information when running at the debug logging level.

    Parameters
    ----------
    string:
        The string to colour flush print for debug messages.
    """
    _bluemira_clean_flush(
        _bm_print_singleflush(string, color="green"), func=LOGGER.debug
    )


def bluemira_print_clean(string: str):
    """
    Print to the logging info console with no modification.
    Useful for external programs

    Parameters
    ----------
    string:
        The string to print
    """
    _terminator_handler(LOGGER.info, string)


def bluemira_error_clean(string: str):
    """
    Print to the logging error console, colouring the output red.
    No other modification is made. Useful for external programs

    Parameters
    ----------
    string:
        The string to colour print
    """
    _terminator_handler(LOGGER.error, _print_color(string, "red"))


# =============================================================================
# Banner printing
# =============================================================================


BLUEMIRA_ASCII = r"""+-------------------------------------------------------------------------+
|  _     _                      _                                         |
| | |   | |                    (_)                                        |
| | |__ | |_   _  ___ _ __ ___  _ _ __ __ _ __                            |
| | '_ \| | | | |/ _ \ '_ ` _ \| | '__/ _| |_ \                           |
| | |_) | | |_| |  __/ | | | | | | | | (_| |_) |                          |
| |_.__/|_|\__,_|\___|_| |_| |_|_|_|  \__|_|__/                           |
+-------------------------------------------------------------------------+"""  # noqa: E501


def print_banner():
    """
    Print the initial banner to the console upon running the bluemira code.
    """
    LOGGER.info(_print_color(BLUEMIRA_ASCII, color="blue"))
    v = version_banner()
    v.extend(user_banner())
    bluemira_print("\n".join(v))


def version_banner() -> List[str]:
    """
    Get the string for the version banner.

    Returns
    -------
    The list of strings of text describing the version and code information
    """
    mapping = {
        "SLOC": "total",
    }
    root = get_bluemira_root()
    if not Path(f"{root}/.git").is_dir() or shutil.which("git") is None:
        return [
            f"Version    : {__version__}",
            "git branch : docker",
            "SLOC      : N/A",
        ]
    branch = get_git_branch(root)
    sloc = count_slocs(get_bluemira_path().rstrip(os.sep), branch)
    v = str(get_git_version(root))

    output = [f"Version    : {v[2:-1]}", f"git branch : {branch}"]

    for k, v in mapping.items():
        if sloc[v] > 0:
            line = k + " " * (11 - len(k)) + f": {int(sloc[v])}"
            output.append(line)
    return output


def user_banner() -> List[str]:
    """
    Get user and platform info and create text to print to banner.

    Returns
    -------
    The text for the banner containing user and platform information
    """
    return [
        f"User       : {getuser()}",
        f"Platform   : {get_platform()}",
    ]
