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
"""Coilset optimisation problem classes and tools."""

from bluemira.equilibria.optimisation.problem._breakdown import (
    BreakdownCOP,
    BreakdownZoneStrategy,
    CircularZoneStrategy,
    InboardBreakdownZoneStrategy,
    InputBreakdownZoneStrategy,
    OutboardBreakdownZoneStrategy,
)
from bluemira.equilibria.optimisation.problem._minimal_current import MinimalCurrentCOP
from bluemira.equilibria.optimisation.problem._nested_position import (
    NestedCoilsetPositionCOP,
    PulsedNestedPositionCOP,
)
from bluemira.equilibria.optimisation.problem._position import CoilsetPositionCOP
from bluemira.equilibria.optimisation.problem._tikhonov import (
    TikhonovCurrentCOP,
    UnconstrainedTikhonovCurrentGradientCOP,
)
from bluemira.equilibria.optimisation.problem.base import CoilsetOptimisationProblem

__all__ = [
    "BreakdownCOP",
    "BreakdownZoneStrategy",
    "CircularZoneStrategy",
    "CoilsetOptimisationProblem",
    "CoilsetPositionCOP",
    "InboardBreakdownZoneStrategy",
    "InputBreakdownZoneStrategy",
    "MinimalCurrentCOP",
    "NestedCoilsetPositionCOP",
    "OutboardBreakdownZoneStrategy",
    "PulsedNestedPositionCOP",
    "TikhonovCurrentCOP",
    "UnconstrainedTikhonovCurrentGradientCOP",
]
