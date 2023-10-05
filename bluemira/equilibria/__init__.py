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
The bluemira equilibria module
"""

from bluemira.equilibria.coils import Coil, CoilSet, SymmetricCircuit
from bluemira.equilibria.equilibrium import Breakdown, Equilibrium
from bluemira.equilibria.find import (
    find_LCFS_separatrix,
    find_OX_points,
    find_flux_surfs,
)
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.limiter import Limiter
from bluemira.equilibria.opt_constraints import (
    AutoConstraints,
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
    PsiBoundaryConstraint,
    PsiConstraint,
)
from bluemira.equilibria.opt_problems import (
    CoilsetPositionCOP,
    NestedCoilsetPositionCOP,
    TikhonovCurrentCOP,
    UnconstrainedTikhonovCurrentGradientCOP,
)
from bluemira.equilibria.profiles import BetaIpProfile, CustomProfile
from bluemira.equilibria.shapes import (
    flux_surface_cunningham,
    flux_surface_johner,
    flux_surface_manickam,
)
from bluemira.equilibria.solve import PicardIterator

__all__ = [
    "Coil",
    "CoilSet",
    "SymmetricCircuit",
    "Breakdown",
    "Equilibrium",
    "find_flux_surfs",
    "find_LCFS_separatrix",
    "find_OX_points",
    "Grid",
    "Limiter",
    "AutoConstraints",
    "FieldNullConstraint",
    "IsofluxConstraint",
    "MagneticConstraintSet",
    "PsiBoundaryConstraint",
    "PsiConstraint",
    "CoilsetPositionCOP",
    "NestedCoilsetPositionCOP",
    "TikhonovCurrentCOP",
    "UnconstrainedTikhonovCurrentGradientCOP",
    "BetaIpProfile",
    "CustomProfile",
    "PicardIterator",
    "flux_surface_cunningham",
    "flux_surface_johner",
    "flux_surface_manickam",
]
