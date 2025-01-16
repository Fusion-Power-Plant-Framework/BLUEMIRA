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

from bluemira.base.file import try_get_bluemira_path


class EqSubplots(Enum):
    """
    Type of plot axes.
    Determines number of subplots, axes label, etc.
    """

    XZ = auto()
    """Plot x vs z"""
    XZ_COMPONENT_PSI = auto()
    """PLot x vs z for differnt psi components."""
    VS_PSI_NORM = auto()
    """Plot parmeters (numbers of which can vary) against the normailised psi."""
    VS_X = auto()
    """Plot parmeters (numbers of which can vary) against x."""


class PsiPlotType(Flag):
    """
    For use with psi comparison plotter.
    """

    PSI = auto()
    """Plot equilibrium psi."""
    PSI_DIFF = auto()
    """
    Plot the difference between a reference equilibrium psi
    and the equilibrium psi.
    """
    PSI_ABS_DIFF = auto()
    """
    Plot the absolute difference between a reference equilibrium psi
    and the equilibrium psi.
    """
    PSI_REL_DIFF = auto()
    """
    Plot the realstive difference between a reference equilibrium psi
    and the equilibrium psi.
    """
    DIFF = PSI_DIFF | PSI_ABS_DIFF | PSI_REL_DIFF


class DivLegsToPlot(Flag):
    """Which divertor legs to create plots for."""

    UP = auto()
    """Upper pair of legs."""
    LW = auto()
    """Lower pair of legs."""
    ALL = auto()
    """All availble legs."""
    PAIR = UP | LW


class LCFSMask(Enum):
    """
    For LCFS masking in plots.
    Block the area within or outside of the refernce LCFS.
    """

    IN = auto()
    """Mask out values inside reference LCFS."""
    OUT = auto()
    """Mask out values outside reference LCFS."""


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


class EqBPlotParam(Flag):
    """
    The paramater to plot for an equilibria xz plot.
    """

    PSI = auto()
    """Poloidal Magnetic Flux"""
    BP = auto()
    """Poloidal Field"""
    BT = auto()
    """Toroidal Field"""
    FIELD = BP | BT


class FluxSurfaceType(Enum):
    """
    For flux surface comparision plotting.
    Compare LCFSs, separaticesor flux surfeaces with a given normailised psi.
    """

    LCFS = auto()
    SEPARATRIX = auto()
    PSI_NORM = auto()


@dataclass
class EqDiagnosticOptions:
    """Diagnostic plotting options for Equilibrium."""

    reference_eq: Equilibrium | None = None
    psi_diff: PsiPlotType = PsiPlotType.PSI
    split_psi_plots: EqSubplots = EqSubplots.XZ
    lcfs_mask: LCFSMask | None = None
    plot_name: str = "default_0"
    folder: str | PathLike | None = None
    save: bool = False

    def __post_init__(self):
        """Post init folder definition"""
        self.folder = Path.cwd() if self.folder is None else Path(self.folder)


class PicardDiagnostic(Enum):
    """Type of plot to view during optimisation."""

    EQ = auto()
    """Plot the equilibrium"""
    CONVERGENCE = auto()
    """Plot the convergence"""
    NO_PLOT = auto()


@dataclass
class PicardDiagnosticOptions:
    """
    Diagnostic plotting options for the Picard Iterator

    plot:
        What type of plot to produce. None for no plotting.
    gif:
        Whether or not to make a GIF
    plot_name:
        GIF plot file base-name
    figure_folder:
        The path where figures will be saved. If the input value is None (e.g. default)
        then this will be reinterpreted as the path data/plots/equilibria under the
        bluemira root folder, if that path is available.
    """

    plot: PicardDiagnostic = PicardDiagnostic.NO_PLOT
    gif: bool = False
    plot_name: str = "default_0"
    figure_folder: str | PathLike | None = None

    def __post_init__(self):
        """Post init folder definition"""
        if self.figure_folder is None:
            figure_folder = try_get_bluemira_path(
                "", subfolder="generated_data", allow_missing=not self.gif
            )
        self.figure_folder = figure_folder
