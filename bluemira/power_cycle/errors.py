# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Exception classes for the power cycle model.
"""

from bluemira.base.error import BluemiraError


class PowerCycleError(BluemiraError):
    """PowerCycle base error."""


class PowerLoadError(PowerCycleError):
    """PowerCycleLoad error class."""


class ScenarioLoadError(PowerCycleError):
    """
    Exception class for 'ScenarioLoad' class of the Power Cycle module.
    """


class CoilSupplySystemError(PowerCycleError):
    """
    Exception class for 'CoilSupplySystem' class of the Power Cycle module.
    """
