# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Equilibria and Equilibria optimisation analysis tools"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bluemira.equilibria.equilibrium import MHDState
    from bluemira.equilibria.optimisation.problem.base import CoilsetOptimisationProblem


class EqAnalysis:
    """Equilibria analysis toolbox"""

    def __init__(self, eq: MHDState):
        self._eq = eq

    def plot(self):
        """Plot equilibria"""
        return self.eq.plot()


class COPAnalysis(EqAnalysis):
    """Coilset Optimisation Problem analysis toolbox"""

    def __init__(self, cop: CoilsetOptimisationProblem):
        super().__init__(cop.eq)
        self._cop = cop
