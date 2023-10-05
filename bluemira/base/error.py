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
Bluemira base error class
"""

from textwrap import dedent, fill


class BluemiraError(Exception):
    """
    Base exception class. Sub-class from this for module level Errors.
    """

    def __str__(self) -> str:
        """
        Prettier handling of the Exception strings
        """
        return fill(dedent(self.args[0]))


class BuilderError(BluemiraError):
    """
    Exception class for Builders.
    """


class ComponentError(BluemiraError):
    """
    Exception class for Components.
    """


class LogsError(BluemiraError):
    """
    Exception class for Components.
    """


class ParameterError(BluemiraError):
    """
    Exception class for Parameters.
    """


class DesignError(BluemiraError):
    """
    Exception class for Designs.
    """


class ReactorError(BluemiraError):
    """Exceptions related to :class:`bluemira.base.reactor.Reactor` objects."""


class ReactorConfigError(BluemiraError):
    """
    Exceptions related to
    :class:`bluemira.base.reactor_config.ReactorConfig` objects.
    """
