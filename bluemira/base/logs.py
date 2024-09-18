# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later


"""Logging system setup and control."""

from __future__ import annotations

import logging
import sys
from enum import Enum
from textwrap import dedent, wrap
from types import DynamicClassAttribute
from typing import TYPE_CHECKING

from bluemira.base.constants import ANSI_COLOR, EXIT_COLOR
from bluemira.base.error import LogsError

if TYPE_CHECKING:
    from collections.abc import Iterable


class LogLevel(Enum):
    """Linking level names and corresponding numbers."""

    CRITICAL = 5
    ERROR = 4
    WARNING = 3
    INFO = 2
    DEBUG = 1
    NOTSET = 0

    @classmethod
    def _missing_(cls, value: int | str) -> LogLevel:
        if isinstance(value, int):
            if cls.CRITICAL.value < value < 10:  # noqa: PLR2004
                return cls.CRITICAL
            value = max(value // 10 + value % 10, 0)
            if value <= cls.CRITICAL.value:
                return cls(value)
            return cls.CRITICAL
        try:
            return cls[value.upper()]
        except (KeyError, AttributeError):
            raise LogsError(
                f"Unknown severity level: {value}. Choose from: {(*cls._member_names_,)}"
            ) from None

    @DynamicClassAttribute
    def _value_for_logging(self) -> int:
        """Return builtin logging level value"""
        return int(self.value * 10)


class Formatter(logging.Formatter):
    """Custom formatter for our logging"""

    flush_previous = False

    def format(self, record) -> str:
        """Format logging strings"""
        if record.msg.startswith("\r"):
            self.flush_previous = True
        elif self.flush_previous:
            record.msg = f"\n{record.msg}"
            self.flush_previous = False

        return super().format(record)


class LoggerAdapter(logging.Logger):
    """Adapt the base logging class for our uses"""

    def debug(
        self,
        msg,
        *args,
        flush: bool = False,
        fmt: bool = True,
        clean: bool = False,
        **kwargs,
    ):
        """Debug"""
        return super().debug(
            msg if clean else colourise(msg, color="green", flush=flush, fmt=fmt),
            *args,
            **kwargs,
        )

    def info(
        self,
        msg,
        *args,
        flush: bool = False,
        fmt: bool = True,
        clean: bool = False,
        **kwargs,
    ):
        """Info"""
        return super().info(
            msg if clean else colourise(msg, color="blue", flush=flush, fmt=fmt),
            *args,
            **kwargs,
        )

    def warning(self, msg, *args, flush: bool = False, fmt: bool = True, **kwargs):
        """Warning"""
        return super().warning(
            colourise(f"WARNING: {msg}", color="orange", flush=flush, fmt=fmt),
            *args,
            **kwargs,
        )

    def error(self, msg, *args, flush: bool = False, fmt: bool = True, **kwargs):
        """Error"""
        return super().error(
            colourise(f"ERROR: {msg}", color="red", flush=flush, fmt=fmt),
            *args,
            **kwargs,
        )

    def critical(self, msg, *args, flush: bool = False, fmt: bool = True, **kwargs):
        """Critical"""
        return super().critical(
            colourise(f"CRITICAL: {msg}", color="darkred", flush=flush, fmt=fmt),
            *args,
            **kwargs,
        )


def _bm_print(string: str, width: int = 73) -> str:
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
    :
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


def _bm_print_singleflush(string: str, width: int = 73) -> str:
    """
    Create the text string for coloured, boxed text to flush print to the
    console.

    Parameters
    ----------
    string:
        The string of text to colour and box
    width:
        The width of the box, default = 73 (leave this alone for best results)

    Returns
    -------
    :
        The text string of the boxed coloured text to flush print
    """
    a = width - len(string) - 2
    return "| " + string + a * " " + " |"


def colourise(
    string: str,
    width: int = 73,
    color: str = "blue",
    *,
    flush: bool = False,
    fmt: bool = True,
) -> str:
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
    if fmt:
        text = _bm_print_singleflush(string) if flush else _bm_print(string, width=width)
    else:
        text = string

    return ("\r" if flush else "") + _print_color(text, color)


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
    :
        The string with ANSI color decoration
    """
    return f"{ANSI_COLOR[color]}{string}{EXIT_COLOR}"


def logger_setup(
    logfilename: str = "bluemira.log", *, level: str | int = "INFO"
) -> logging.Logger:
    """
    Create logger with two handlers.

    Parameters
    ----------
    logfilename:
        Name of file to write logs to, default = bluemira.log
    level:
        The initial logging level to be printed to the console, default = INFO.

    Notes
    -----
    set to debug initially
    """
    logging.setLoggerClass(LoggerAdapter)
    root_logger = logging.getLogger("")
    bm_logger = logging.getLogger("bluemira")

    # what will be shown on screen
    formatter = Formatter()
    on_screen_handler_out = logging.StreamHandler(stream=sys.stdout)
    on_screen_handler_out.setLevel(LogLevel(level)._value_for_logging)
    on_screen_handler_out.setFormatter(formatter)
    on_screen_handler_out.addFilter(lambda record: record.levelno < logging.WARNING)

    on_screen_handler_err = logging.StreamHandler(stream=sys.stderr)
    on_screen_handler_err.setLevel(LogLevel(level)._value_for_logging)
    on_screen_handler_err.setFormatter(formatter)
    on_screen_handler_err.addFilter(lambda record: record.levelno >= logging.WARNING)

    # what will be written to a file
    recorded_handler = logging.FileHandler(logfilename)
    recorded_formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    recorded_handler.setLevel(logging.DEBUG)
    recorded_handler.setFormatter(recorded_formatter)

    bm_logger.setLevel(logging.DEBUG)

    root_logger.addHandler(on_screen_handler_out)
    root_logger.addHandler(on_screen_handler_err)
    root_logger.addHandler(recorded_handler)

    return bm_logger


def set_log_level(
    verbose: int | str = 1,
    *,
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
        _modify_handler(
            LogLevel(verbose)
            if isinstance(verbose, str)
            else LogLevel(int(current_level + verbose)),
            logger,
        )


def get_log_level(logger_name: str = "bluemira", *, as_str: bool = True) -> str | int:
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
            max_level = LogLevel(handler.level).value
    if as_str:
        return LogLevel(max_level).name
    return max_level


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
            handler.setLevel(new_level._value_for_logging)


class LoggingContext:
    """
    A context manager for temporarily adjusting the logging level

    Parameters
    ----------
    level:
        The bluemira logging level to set within the context.
    """

    def __init__(self, level: str | int):
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
