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
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike

    from bluemira.equilibria.equilibrium import Equilibrium


@dataclass
class EqDiagnosticOptions:
    """Diagnostic plotting options for Equilibrium."""

    reference_eq: Equilibrium
    psi_diff: bool = False
    split_psi_plots: bool = False
    plot_name: str = "default_0"
    folder: str | PathLike | None = None
    save: bool = False

    def __post_init__(self):
        """Post init folder definition"""
        self.folder = Path.cwd() if self.folder is None else Path(self.folder)
