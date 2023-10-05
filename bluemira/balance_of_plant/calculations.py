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
Simple relations for power.
"""

from typing import Tuple

import numpy as np

from bluemira.base.constants import raw_uc
from bluemira.base.look_and_feel import bluemira_warn


def cryo_power(
    s_tf: float,
    m_cold: float,
    nucl_heating: float,
    e_pf_max: float,
    t_pulse: float,
    tf_current: float,
    n_TF: int,
) -> float:
    """
    Calculates cryogenic loads (taken from PROCESS)

    Parameters
    ----------
    s_tf:
        TF coil total surface area [m^2]
    m_cold:
        Total cold mass [kg]
    nucl_heating:
        Total coil nuclear heating [W]
    e_pf_max:
        Maximum stored energy in the PF coils [J]
    t_pulse:
        Pulse length [s]
    tf_current:
        TF coil current per turn [A]
    n_TF:
        Number of TF coils

    Returns
    -------
    Total power required to cool cryogenic components

    Note
    ----
    Author: P J Knight, CCFE, Culham Science Centre
    D. Slack memo SCMDG 88-5-1-059, LLNL ITER-88-054, Aug. 1988
    """
    # TODO: Temperature!
    # Steady-state loads
    qss = 4.3e-4 * m_cold + 2 * s_tf
    # AC losses
    qac = raw_uc(e_pf_max, "J", "kJ") / t_pulse
    # Current leads
    qcl = 13.6e-3 * n_TF * tf_current
    # Misc. loads (piping and reserves)
    fmisc = 0.45
    return (1 + fmisc) * (nucl_heating + qcl + qac + qss)


def He_pumping(  # noqa: N802
    pressure_in: float,
    pressure_out: float,
    t_in: float,
    t_out: float,
    blanket_power: float,
    eta_isen: float,
    eta_el: float,
) -> Tuple[float, float]:
    """
    Calculate the pumping power for helium-cooled blankets.

    Parameters
    ----------
    pressure_in:
        Inlet pressure [Pa]
    pressure_out:
        Pressure drop [Pa]
    t_in:
        Inlet temperature [K]
    t_out:
        Outlet temperature [K]
    blanket_power:
        Total blanket power excluding pumping power [W]
    eta_isen:
        Isentropic efficiency of the He compressors
    eta_el:
        Electrical efficiency of the He compressors

    Returns
    -------
    P_pump_is:
        The isentropic pumping power (added to the working fluid) [W]
    P_pump_el:
        The electrical pumping power (parasitic load) [W]

    \t:math:`T_{in_{comp}} = \\dfrac{T_{in_{BB}}}{\\dfrac{P}{P-dP}^{\\dfrac{\\gamma-1}{\\gamma}}}`\n
    \t:math:`f_{p} = \\dfrac{T_{in_{comp}}}{\\eta_{is}dT}\\Bigg(\\dfrac{P}{P-dP}^{\\dfrac{\\gamma-1}{\\gamma}}-1\\Bigg)`

    Notes
    -----
    \t:math:`f_{p} = \\dfrac{T_{in_{BB}}}{\\eta_{is}dT}\\Bigg(1-\\dfrac{P-dP}{P}^{\\dfrac{\\gamma-1}{\\gamma}}\\Bigg)`
    **Outputs:**\n
    \t:math:`P_{pump} = \\dfrac{f_{p}P_{plasma}}{1-f_p}` [W]\n
    \t:math:`P_{pump,el} = \\dfrac{P_{pump}}{\\eta_{el}}` [W]\n
    **No longer in use:**
    \t:math:`f_{pump}=\\dfrac{dP}{dTc_P\\rho_{av}}`
    """  # noqa: W505, E501
    d_temp = t_out - t_in
    t_bb_inlet = t_in
    # Ideal monoatomic gas - small compression ratios
    t_comp_inlet = t_bb_inlet / ((pressure_in / pressure_out) ** (2 / 5))
    # Ivo not sure why can't refind it - probably right but very little
    # difference ~ 1 K
    # T_comp_inlet = eta_isen*T_bb_inlet/((P/(P-dP))**(6/15)+eta_isen-1)
    f_pump = (t_comp_inlet / (eta_isen * d_temp)) * (
        (pressure_in / pressure_out) ** (2 / 5) - 1
    )  # kJ/kg
    p_pump_is = f_pump * blanket_power / (1 - f_pump)
    p_pump_el = p_pump_is / eta_el
    return p_pump_is, p_pump_el


def H2O_pumping(  # noqa: N802
    p_blanket: float, f_pump: float, eta_isen: float, eta_el: float
) -> Tuple[float, float]:
    """
    H20-cooling pumping power calculation strategy

    Parameters
    ----------
    f_pump:
        Fraction of thermal power required to pump
    eta_isen:
        Isentropic efficiency of the water pumps
    eta_el:t
        Electrical efficiency of the water pumps

    Returns
    -------
    P_pump_is:
        The isentropic pumping power (added to the working fluid)
    P_pump_el:
        The eletrical pumping power (parasitic load)
    """
    # TODO: Add proper pump model
    f_pump /= eta_isen

    p_pump_is = f_pump * p_blanket / (1 - f_pump)
    p_pump_el = p_pump_is / eta_el
    return p_pump_is, p_pump_el


def superheated_rankine(
    blanket_power: float, div_power: float, bb_outlet_temp: float, delta_t_turbine: float
) -> float:
    """
    PROCESS C. Harrington correlation. Accounts for low-grade heat penalty.
    Used for He-cooled blankets. Not applicable to H2O temperatures.

    Parameters
    ----------
    blanket_power:
        Blanket thermal power [W]
    div_power:
        Divertor thermal power [W]
    bb_outlet_temp:
        Blanket outlet temperature [K]
    delta_t_turbine:
        Turbine inlet temperature drop [K]

    Returns
    -------
    Efficiency of a superheated Rankine cycle
    """
    t_turb = bb_outlet_temp - delta_t_turbine
    if t_turb < 657 or t_turb > 915:  # noqa: PLR2004
        bluemira_warn("BoP turbine inlet temperature outside range of validity.")
    f_lgh = div_power / (blanket_power + div_power)
    delta_eta = 0.339 * f_lgh
    return 0.1802 * np.log(t_turb) - 0.7823 - delta_eta
