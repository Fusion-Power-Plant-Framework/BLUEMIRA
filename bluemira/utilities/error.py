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
Error classes for the utilities module.
"""

from bluemira.base.error import BluemiraError


class UtilitiesError(BluemiraError):
    """
    The base class for utilities errors.
    """


class PositionerError(UtilitiesError):
    """
    Error for positioner utilities.
    """


class OptUtilitiesError(UtilitiesError):
    """
    Error for optimisation utilities.
    """


class OptVariablesError(OptUtilitiesError):
    """
    Error for optimisation utilities.
    """


class InternalOptError(OptUtilitiesError):
    """
    Error class for errors inside the optimisation algorithms.
    """


class ExternalOptError(OptUtilitiesError):
    """
    Error class for errors relating to the optimisation, but not originating
    inside the optimisers.
    """
