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
Parameter definitions for Plasmod.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from typing import ClassVar, Dict, Union

from bluemira.base.parameter_frame import Parameter
from bluemira.codes.params import MappedParameterFrame
from bluemira.codes.plasmod.api._inputs import PlasmodInputs
from bluemira.codes.plasmod.mapping import mappings
from bluemira.codes.utilities import ParameterMapping


@dataclass
class PlasmodSolverParams(MappedParameterFrame):
    """Parameters required in :class:`bluemira.codes.plasmod.Solver`."""

    # Input parameters
    A: Parameter[float]
    """Plasma aspect ratio [dimensionless]."""
    B_0: Parameter[float]
    """Toroidal field at plasma center [T]."""
    delta_95: Parameter[float]
    """Plasma triangularity at 95% flux [dimensionless]."""
    kappa_95: Parameter[float]
    """Plasma elongation at 95% flux [dimensionless]."""
    R_0: Parameter[float]
    """Plasma major radius [m]."""
    V_p: Parameter[float]
    """
    Constrained plasma volume (set negative value to disable volume constraining) [m3].
    """
    e_nbi: Parameter[float]
    """NBI energy [keV]."""
    f_ni: Parameter[float]
    """Required fraction of non inductive current, if 0, dont use CD [dimensionless]."""
    q_control: Parameter[float]
    """Fixed auxiliary heating power required for control [MW]."""
    PsepB_qAR_max: Parameter[float]
    """Divertor challenging criterion Psep * Bt / (q95 * A * R_0) [MW.T/m]"""

    # In-out parameters
    delta: Parameter[float]
    """
    Plasma edge triangularity (used only for first iteration, then
    iterated to constrain delta95) [dimensionless].
    """
    kappa: Parameter[float]
    """
    Plasma edge elongation (used only for first iteration, then
    iterated to constrain kappa95) [dimensionless].
    """
    I_p: Parameter[float]
    """
    Plasma current (used if i_equiltype == 2. Otherwise Ip is
    calculated and q95 is used as input) [MA].
    """
    q_95: Parameter[float]
    """
    Safety factor at 95% flux surface (used if i_equiltype == 1.
    Otherwise q95 is calculated and Ip is used as input) [dimensionless].
    """
    T_e_ped: Parameter[float]
    """Electrons/ions temperature at pedestal (ignored if i_pedestal = 2) [keV]."""

    # Output parameters
    beta_p: Parameter[float]
    """Poloidal beta [dimensionless]."""
    beta_N: Parameter[float]  # noqa: N815
    """Normalized beta [dimensionless]."""
    f_bs: Parameter[float]
    """Plasma bootstrap current fraction [dimensionless]."""
    l_i: Parameter[float]
    """Normalised plasma internal inductance [dimensionless]."""
    H_star: Parameter[float]
    """Radiation-corrected H-factor [dimensionless]."""
    tau_e: Parameter[float]
    """Global energy confinement time [s]."""
    res_plasma: Parameter[float]
    """Plasma resistance [Ohm]."""
    P_fus_DD: Parameter[float]
    """DD fusion power [W]."""
    P_fus_DT: Parameter[float]
    """DT fusion power [W]."""
    P_fus: Parameter[float]
    """Fusion power [W]."""
    P_rad: Parameter[float]
    """Total radiation power [W]."""
    P_sep: Parameter[float]
    """Total power across plasma separatrix [W]."""
    P_sync: Parameter[float]
    """Synchrotron radiation power [W]."""
    P_brehms: Parameter[float]
    """Bremsstrahlung radiation power [W]."""
    P_line: Parameter[float]
    """Line radiation power [W]."""
    P_LH: Parameter[float]
    """LH transition power [W]."""
    P_ohm: Parameter[float]
    """Ohmic heating power [W]."""
    Z_eff: Parameter[float]
    """Plasma effective charge [dimensionless]."""
    v_burn: Parameter[float]
    """Target loop voltage (if lower than -1e-3, ignored)-> plasma loop voltage [V]."""

    _mappings: ClassVar = deepcopy(mappings)
    _defaults = PlasmodInputs()

    @property
    def mappings(self) -> Dict[str, ParameterMapping]:
        """Define mappings between these parameters and Plasmod's."""
        return self._mappings

    @property
    def defaults(self) -> Dict[str, Union[float, Enum]]:
        """Defaults for Plasmod"""
        return self._defaults.to_dict()

    @classmethod
    def from_defaults(cls) -> MappedParameterFrame:
        """
        Initialise from defaults
        """
        return super().from_defaults(asdict(cls._defaults))
