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

import datetime
import logging
import os
import platform
import shutil
import subprocess  # noqa :S404
import time
from getpass import getuser
from textwrap import dedent, wrap
from typing import Tuple

from rich import traceback

from bluemira import __version__
from bluemira.base.constants import ANSI_COLOR, EXIT_COLOR
from bluemira.base.file import get_bluemira_path, get_bluemira_root
from bluemira.base.logs import logger_setup

traceback.install()

# =============================================================================
# Printing functions
# =============================================================================


def _print_color(string, color):
    """
    Create text to print. NOTE: Does not call print command

    Parameters
    ----------
    string: str
        The text to colour
    color: str
        The color to make the color-string for

    Returns
    -------
    color_text: str
        The string with ANSI color decoration
    """
    return f"{ANSI_COLOR[color]}{string}{EXIT_COLOR}"


def _bm_print(string, width=73):
    """
    Create the text string for boxed text to print to the console.

    Parameters
    ----------
    string: str
        The string of text to colour and box
    width: int (default = 73)
        The width of the box (leave this alone for best results)

    Returns
    -------
    ss: str
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
    h = "".join(["+"] + ["-" * width] + ["+"])
    return h + "\n" + "\n".join(lines) + "\n" + h


def colourise(string, width=73, color="blue"):
    """
    Print coloured, boxed text to the console. Default template for bluemira
    information.

    Parameters
    ----------
    string: str
        The string of text to colour and box
    width: int (default = 73)
        The width of the box (leave this alone for best results)
    color: str from bluemira.base.constants.ANSI_COLOR
        The color to print the text in
    """
    text = _bm_print(string, width=width)
    color_text = _print_color(text, color)
    return color_text


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
+-------------------------------------------------------------------------+"""  # noqa


def get_banner() -> Tuple[str, ...]:
    """
    Get the initial banner to the console upon running the bluemira code.
    """
    header = _print_color(BLUEMIRA_ASCII, color="blue")
    v = version_banner()
    v.extend(user_banner())
    info = colourise("\n".join(v), color="blue")
    return header, info


def version_banner():
    """
    Get the string for the version banner.

    Returns
    -------
    output: List[str]
        The list of strings of text describing the version and code information
    """
    mapping = {
        "SLOC": "total",
    }
    root = get_bluemira_root()
    if not os.path.isdir(f"{root}/.git") or shutil.which("git") is None:
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


def user_banner():
    """
    Get user and platform info and create text to print to banner.

    Returns
    -------
    s: str
        The text for the banner containing user and platform information
    """
    return [
        f"User       : {getuser()}",
        f"Platform   : {get_platform()}",
    ]


# Calculate the number of lines in this file
try:
    with open(
        os.sep.join([get_bluemira_path("base"), "look_and_feel.py"]),
        "r",
        encoding="utf-8",
    ) as f:
        LOCAL_LINES = len(f.read().splitlines())
except FileNotFoundError:
    # Approximately
    LOCAL_LINES = 550

# =============================================================================
# Getters for miscellaneous information
# =============================================================================


def get_git_version(directory):
    """
    Get the version string of the current git branch, e.g.: '0.0.3-74-g70d48be'.

    Parameters
    ----------
    directory: str
        The full path directory of the folder to get git information from

    Returns
    -------
    vinfo: bytes
        The git version bytestring
    """
    return subprocess.check_output(  # noqa :S603, S607
        ["git", "describe", "--tags", "--always"], cwd=directory
    ).strip()


def get_git_branch(directory):
    """
    Get the name of the current git branch, e.g. 'develop'.

    Parameters
    ----------
    directory: str
        The full path directory of the folder to get git information from

    Returns
    -------
    branch: str
        The git branch string
    """
    out = subprocess.check_output(  # noqa :S603, S607
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=directory
    )
    return out.strip().decode("utf-8")


def get_git_files(directory, branch):
    """
    Get the names of the files in the directory of the specified branch name.

    Parameters
    ----------
    directory: str
        The full path directory of the folder to get git information from
    branch: str
        The name of the git branch to retrieve the filenames from

    Returns
    -------
    files: List[str]
        The list of git-controlled path strings
    """
    return (
        subprocess.check_output(  # noqa :S603, S607
            ["git", "ls-tree", "-r", branch, "--name-only"], cwd=directory
        )
        .decode("utf-8")
        .splitlines()
    )


def get_platform():
    """
    Get the OS platform.

    Returns
    -------
    platform: str
        The generic name of the platform (e.g. Linux, Windows)
    """
    return platform.uname()[0]


def count_slocs(
    directory,
    branch,
    exts=None,
    ignore=None,
):
    """
    Counts lines of code within a given directory for a given git branch

    Parameters
    ----------
    directory: str
        The full path directory of the folder to get git information from
    branch: str
        The git branch string
    exts: list
        The list of file extensions to search the directory for
    ignore: list
        The list of extensions and filenames to ignore

    Returns
    -------
    lines: dict
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
        if name.split(os.sep)[-1] not in ignore and name not in ignore:
            for e in exts:
                if name.endswith(e):
                    path = os.sep.join([directory, name])
                    try:
                        with open(path, "r", encoding="utf-8") as file:
                            lines[e] += len(file.read().splitlines())

                    except FileNotFoundError:
                        bluemira_warn(
                            "count_slocs: Probably not on the right git branch"
                        )
                        continue

    lines[".py"] += LOCAL_LINES
    lines["total"] = sum([lines[k] for k in lines.keys()])
    return lines


LOGGER = logger_setup(banner=get_banner)


def bluemira_critical(string):
    """
    Standard template for bluemira critical errors.
    """
    return LOGGER.critical(colourise(f"CRITICAL: {string}", color="darkred"))


def bluemira_error(string):
    """
    Standard template for bluemira errors.
    """
    return LOGGER.error(colourise(f"ERROR: {string}", color="red"))


def bluemira_warn(string):
    """
    Standard template for bluemira warnings.
    """
    return LOGGER.warning(colourise(f"WARNING: {string}", color="orange"))


def bluemira_print(string):
    """
    Standard template for bluemira information messages.
    """
    return LOGGER.info(colourise(string, color="blue"))


def bluemira_debug(string):
    """
    Standard template for bluemira debugging.
    """
    return LOGGER.debug(colourise(string, color="green"))


def _bm_print_singleflush(string, width=73, color="blue"):
    """
    Create the text string for coloured, boxed text to flush print to the
    console.

    Parameters
    ----------
    string: str
        The string of text to colour and box
    width: int (default = 73)
        The width of the box (leave this alone for best results)
    color: str from ['blue', 'red', 'green']
        The color to print the text in

    Returns
    -------
    ss: str
        The text string of the boxed coloured text to flush print
    """
    a = width - len(string) - 2
    text = "| " + string + a * " " + " |"
    return _print_color(text, color)


def _bluemira_clean_flush(string, func=LOGGER.info):
    """
    Print and flush string. Useful for updating information.

    Parameters
    ----------
    func: Callable[[str], None]
        The function to use for logging, by default LOGGER.info
    string: str
        The string to colour flush print
    """
    _terminator_handler(func, "\r" + string, fhterm=logging.StreamHandler.terminator)


def _terminator_handler(func, string, *, fhterm=""):
    """
    Log string allowing modification to handler terminator

    Parameters
    ----------
    func: Callable[[str], None]
        The function to use for logging (e.g LOGGER.info)
    string: str
        The string to colour flush print
    fhterm: str
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


def bluemira_print_flush(string):
    """
    Print a coloured, boxed line to the console and flushes it. Useful for
    updating information.

    Parameters
    ----------
    string: str
        The string to colour flush print
    """
    _bluemira_clean_flush(_bm_print_singleflush(string), func=LOGGER.info)


def bluemira_debug_flush(string):
    """
    Print a coloured, boxed line to the console and flushes it. Useful for
    updating information when running at the debug logging level.

    Parameters
    ----------
    string: str
        The string to colour flush print for debug messages.
    """
    _bluemira_clean_flush(
        _bm_print_singleflush(string, color="green"), func=LOGGER.debug
    )


def bluemira_print_clean(string):
    """
    Print to the logging info console with no modification.
    Useful for external programs

    Parameters
    ----------
    string: str
        The string to print
    """
    _terminator_handler(LOGGER.info, string)


def bluemira_error_clean(string):
    """
    Print to the logging error console, colouring the output red.
    No other modification is made. Useful for external programs

    Parameters
    ----------
    string: str
        The string to colour print
    """
    _terminator_handler(LOGGER.error, _print_color(string, "red"))


class BluemiraClock:
    """
    A printed progress bar.

    Parameters
    ----------
    n_iter: int
        The number of iterations
    print_rate: int
        The update rate
    width: int
        The width of the progress bar
    """

    def __init__(self, n_iter, print_rate=1, width=73):
        self.rate = print_rate
        self.elapsed = " elapsed"
        self.left = " left"

        self.width = (
            width - len(self.elapsed) - len(self.left) - 2 * (9 + len(str(n_iter))) - 17
        )

        # Constructors
        self.i = 0
        self.t_start = None
        self.n_iter = None

        self.start(n_iter)

    def start(self, n_iter):
        """
        Start the clock.
        """
        self.i = 0
        self.t_start = time.time()
        self.n_iter = n_iter

    def stop(self):
        """
        Stop the clock.
        """
        return time.time() - self.t_start

    def tock(self):
        """
        Tick the iterations of the clock over.
        """
        self.i += 1
        if self.i % self.rate == 0 and self.i > 0:
            elapsed = time.time() - self.t_start
            remain = elapsed * (self.n_iter - self.i) / self.i
            prog_str = f"{self.i:1d}/{self.n_iter:1d}"

            str_elapsed = str(datetime.timedelta(seconds=int(elapsed)))
            str_left = str(datetime.timedelta(seconds=int(remain)))
            prog_str += f" {str_elapsed:0>8}s{self.elapsed}"
            prog_str += f" {str_left:0>8}s{self.left}"
            prog_str += f" {1e2 * self.i / self.n_iter:1.1f}%"
            nh = int((self.i / self.n_iter) * self.width)
            prog_str += " |" + nh * "#" + (self.width - nh) * "-" + "|"

            bluemira_print_flush(prog_str)

        if self.i == self.n_iter:
            print("\n")
