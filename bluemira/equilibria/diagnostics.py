# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Diagnostic options for use in the equilibria module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, Flag, auto
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike

    from bluemira.equilibria.equilibrium import Equilibrium


class PsiPlotType(Flag):
    """FIXME"""

    PSI = auto()
    PSI_DIFF = auto()
    PSI_ABS_DIFF = auto()
    PSI_REL_DIFF = auto()
    DIFF = PSI_DIFF | PSI_ABS_DIFF | PSI_REL_DIFF


class LCFSMask(Enum):
    """
    For LCFS masking in plots.
    Block the area within or outside of the refernce LCFS.
    """

    IN = auto()
    OUT = auto()


class CSData(Enum):
    """
    For the coilset comparision tables.
    Value to be comapred: current, x-position, or z-position.
    """

    CURRENT = auto()
    XLOC = auto()
    ZLOC = auto()


class FixedOrFree(Enum):
    """
    For use in select_eq - to create appropriate
    Equilibrium or FixedPlasmaEquilibrium object.
    Fixed or free boundary equilibrium.
    """

    FIXED = auto()
    FREE = auto()


@dataclass
class EqDiagnosticOptions:
    """Diagnostic plotting options for Equilibrium."""

    reference_eq: Equilibrium | None = None
    psi_diff: PsiPlotType = PsiPlotType.PSI
    split_psi_plots: bool = False
    lcfs_mask: LCFSMask | None = None
    plot_name: str = "default_0"
    folder: str | PathLike | None = None
    save: bool = False

    def __post_init__(self):
        """Post init folder definition"""
        self.folder = Path.cwd() if self.folder is None else Path(self.folder)
