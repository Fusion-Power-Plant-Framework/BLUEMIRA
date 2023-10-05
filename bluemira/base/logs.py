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


"""Logging system setup and control."""

import logging
import sys
from enum import Enum
from typing import Iterable, Union

from bluemira.base.error import LogsError


class LogLevel(Enum):
    """Linking level names and corresponding numbers."""

    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0


def logger_setup(
    logfilename: str = "bluemira.log", *, level: Union[str, int] = "INFO"
) -> logging.Logger:
    """
    Create logger with two handlers.

    Parameters
    ----------
    logfilename:
        Name of file to write logs to, default = bluemira.log
    level:
        The initial logging level to be printed to the console, default = INFO.

    Returns
    -------
    The logger to be used

    Notes
    -----
    set to debug initially
    """
    root_logger = logging.getLogger("")
    bm_logger = logging.getLogger("bluemira")

    py_level = _convert_log_level(level).value

    # what will be shown on screen

    on_screen_handler = logging.StreamHandler(stream=sys.stderr)
    on_screen_handler.setLevel(py_level)

    # what will be written to a file

    recorded_handler = logging.FileHandler(logfilename)
    recorded_formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    recorded_handler.setLevel(logging.DEBUG)
    recorded_handler.setFormatter(recorded_formatter)

    bm_logger.setLevel(logging.DEBUG)

    root_logger.addHandler(on_screen_handler)
    root_logger.addHandler(recorded_handler)

    return bm_logger


def set_log_level(
    verbose: Union[int, str] = 1,
    increase: bool = False,
    logger_names: Iterable[str] = ("bluemira",),
):
    """
    Get new log level and check if it is possible.

    Parameters
    ----------
    verbose:
        Amount the severity level of the logger should be changed by or to
    increase:
        Whether level should be increased by specified amount or changed to it
    logger_names:
        The loggers for which to set the level, default = ("bluemira")
    """
    # change loggers level
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)

        current_level = logger.getEffectiveLevel() if increase else 0
        new_level = _convert_log_level(verbose, current_level)
        _modify_handler(new_level, logger)


def get_log_level(logger_name: str = "bluemira", as_str: bool = True) -> Union[str, int]:
    """
    Return the current logging level.

    Parameters
    ----------
    logger_name
        The named logger to get the level for.
    as_str
        If True then return the logging level as a string, else as an int.
    """
    logger = logging.getLogger(logger_name)

    max_level = 0
    for handler in logger.handlers or logger.parent.handlers:
        if not isinstance(handler, logging.FileHandler) and handler.level > max_level:
            max_level = handler.level
    if as_str:
        return LogLevel(max_level).name
    return max_level // 10


def _convert_log_level(level: Union[str, int], current_level: int = 0) -> LogLevel:
    """
    Convert the provided logging level to a LogLevel objects.

    Parameters
    ----------
    level:
        The bluemira logging level.
    current_level:
        The current bluemira logging level to increment from.

    Returns
    -------
    The LogLevel corresponding to the requested level.
    """
    try:
        if isinstance(level, str):
            new_level = LogLevel[level]
        else:
            value = int(current_level + (level * 10))
            new_level = LogLevel(value)
    except ValueError:
        raise LogsError(f"Unknown severity level - {value}") from None
    except KeyError:
        raise LogsError(f"Unknown severity level - {level}") from None
    return new_level


def _modify_handler(new_level: LogLevel, logger: logging.Logger):
    """
    Change level of the logger from user's input.

    Parameters
    ----------
    new_level:
        Severity level for handler to be changed to, from set_log_level
    logger:
        Logger to be used
    """
    for handler in logger.handlers or logger.parent.handlers:
        if not isinstance(handler, logging.FileHandler):
            handler.setLevel(new_level.value)


class LoggingContext:
    """
    A context manager for temporarily adjusting the logging level

    Parameters
    ----------
    level:
        The bluemira logging level to set within the context.
    """

    def __init__(self, level: Union[str, int]):
        self.level = level
        self.original_level = get_log_level()

    def __enter__(self):
        """
        Set the logging level to the new level when we enter the context.
        """
        set_log_level(self.level)

    def __exit__(self, type, value, traceback):  # noqa: A002
        """
        Set the logging level to the original level when we exit the context.
        """
        set_log_level(self.original_level)
