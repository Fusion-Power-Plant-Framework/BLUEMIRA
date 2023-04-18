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
from typing import Callable, Iterable, Optional

from bluemira.base.error import LogsError


class LogLevel(Enum):
    """Linking level names and corresponding numbers."""

    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0


class LevelFilter(logging.Filter):
    """
    Filter out some logging levels
    """

    def __init__(self, *filtered: LogLevel):
        self.filter_list = filtered

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter out logs with a specific level"""
        return LogLevel[record.levelname] not in self.filter_list


def logger_setup(
    logfilename="bluemira.log", *, level="INFO", banner: Optional[Callable] = None
):
    """
    Create logger with two handlers.

    Parameters
    ----------
    logfilename: str (default = bluemira.log)
        Name of file to write logs to
    level: str or int (default = INFO)
        The initial logging level to be printed to the console.

    Returns
    -------
    logger: logging.RootLogger
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
    on_screen_handler.addFilter(
        LevelFilter(LogLevel.NOTSET, LogLevel.DEBUG, LogLevel.INFO)
    )

    on_screen_handler_stdout = logging.StreamHandler(stream=sys.stdout)
    on_screen_handler_stdout.setLevel(py_level)
    on_screen_handler_stdout.addFilter(
        LevelFilter(LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL)
    )

    # what will be written to a file

    recorded_handler = logging.FileHandler(logfilename)
    recorded_formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    recorded_handler.setLevel(logging.DEBUG)
    recorded_handler.setFormatter(recorded_formatter)

    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(recorded_handler)

    bm_logger.setLevel(logging.DEBUG)
    bm_logger.addHandler(on_screen_handler)
    bm_logger.addHandler(on_screen_handler_stdout)

    if banner:
        header, info = banner()
        bm_logger.info(header)
        bm_logger.info(info)

    return bm_logger


def set_log_level(verbose=1, increase=False, logger_names: Optional[Iterable] = None):
    """
    Get new log level and check if it is possible.

    Parameters
    ----------
    verbose: str or int (default = 1)
        Amount the severity level of the logger should be changed by or to
    increase: bool (default = False)
        Whether level should be increased by specified amount or changed to it
    logger_names: List[str] (default = ["bluemira"])
        The loggers for which to set the level
    """
    if logger_names is None:
        logger_names = [name for name in logging.root.manager.loggerDict]

    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)

        current_level = logger.getEffectiveLevel() if increase else 0
        new_level = _convert_log_level(verbose, current_level)
        _modify_handler(new_level, logger)


def get_log_level(logger_name="bluemira", as_str=True):
    """
    Return the current logging level.

    Parameters
    ----------
    logger_name: str (default = "")
        The named logger to get the level for.
    as_str: bool (default = True)
        If True then return the logging level as a string, else as an int.
    """
    logger = logging.getLogger(logger_name)

    max_level = 0
    for handler in logger.handlers or logger.parent.handlers:
        if not isinstance(handler, logging.FileHandler):
            if handler.level > max_level:
                max_level = handler.level
    if as_str:
        return LogLevel(max_level).name
    else:
        return max_level // 10


def _convert_log_level(level, current_level=0):
    """
    Convert the provided logging level to a LogLevel objects.

    Parameters
    ----------
    level: str or int
        The bluemira logging level.
    current_level: int
        The current bluemira logging level to increment from.

    Returns
    -------
    new_level: LogLevel
        The LogLevel corresponding to the requested level.
    """
    try:
        if isinstance(level, str):
            new_level = LogLevel[level]
        else:
            value = int(current_level + (level * 10))
            new_level = LogLevel(value)
    except ValueError:
        raise LogsError(f"Unknown severity level - {value}")
    except KeyError:
        raise LogsError(f"Unknown severity level - {level}")
    return new_level


def _modify_handler(new_level, logger):
    """
    Change level of the logger from user's input.

    Parameters
    ----------
    new_level: LogLevel
        Severity level for handler to be changed to, from set_log_level
    logger: logging.RootLogger
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
    level: str or int
        The bluemira logging level to set within the context.
    """

    def __init__(self, level):
        self.level = level
        self.original_level = get_log_level()

    def __enter__(self):
        """
        Set the logging level to the new level when we enter the context.
        """
        set_log_level(self.level)

    def __exit__(self, type, value, traceback):
        """
        Set the logging level to the original level when we exit the context.
        """
        set_log_level(self.original_level)
