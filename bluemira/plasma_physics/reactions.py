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
Fusion reactions
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Union

import numpy as np
import numpy.typing as npt

from bluemira.base.constants import (
    AMU_TO_KG,
    C_LIGHT,
    D_MOLAR_MASS,
    ELECTRON_MOLAR_MASS,
    EV_TO_J,
    HE3_MOLAR_MASS,
    HE_MOLAR_MASS,
    J_TO_EV,
    NEUTRON_MOLAR_MASS,
    N_AVOGADRO,
    PROTON_MOLAR_MASS,
    T_MOLAR_MASS,
    raw_uc,
)
from bluemira.base.look_and_feel import bluemira_warn

__all__ = [
    "E_DT_fusion",
    "E_DD_fusion",
    "n_DT_reactions",
    "n_DD_reactions",
    "reactivity",
    "r_T_burn",
    "r_D_burn_DT",
]


def E_DT_fusion() -> float:
    """
    Calculates the total energy released from the D-T fusion reaction

    Returns
    -------
    The energy released from the D-T fusion reaction [eV]

    Notes
    -----
    .. math::
        {^{2}_{1}H}+{^{3}_{1}H}~\\rightarrow~{^{4}_{2}He}~
        (3.5~\\text{MeV})+\\text{n}^{0} (14.1 ~\\text{MeV})\n
        \\Delta E = \\Delta m c^2
    """
    delta_m = (D_MOLAR_MASS + T_MOLAR_MASS) - (HE_MOLAR_MASS + NEUTRON_MOLAR_MASS)
    return delta_m * C_LIGHT**2 * AMU_TO_KG * J_TO_EV


def E_DD_fusion() -> float:
    """
    Calculates the total energy released from the D-D fusion reaction

    Returns
    -------
    The energy released from the D-D fusion reaction [eV]

    Notes
    -----
    .. math::
        {^{2}_{1}H}+{^{2}_{1}H}~\\rightarrow~{^{3}_{1}H}
        (1.01 ~\\text{MeV})+\\text{p} (3.02~\\text{MeV})~~[50 \\textrm{\\%}]
        ~~~~~~~~~~\\rightarrow~{^{3}_{2}He} (0.82~\\text{MeV})+\\text{n}^{0} (2.45~\\text{MeV})~~[50 \\text{\\%}]\n
        \\Delta E = \\Delta m c^2
    """  # noqa: W505, E501
    # NOTE: Electron mass must be included with proton mass
    delta_m = np.array(
        [
            D_MOLAR_MASS
            + D_MOLAR_MASS
            - (T_MOLAR_MASS + PROTON_MOLAR_MASS + ELECTRON_MOLAR_MASS),
            (D_MOLAR_MASS + D_MOLAR_MASS) - (HE3_MOLAR_MASS + NEUTRON_MOLAR_MASS),
        ]
    )
    delta_m = np.average(delta_m)
    return delta_m * C_LIGHT**2 * AMU_TO_KG * J_TO_EV


def n_DT_reactions(p_fus: float) -> float:
    """
    Calculates the number of D-T fusion reactions per s for a given D-T fusion
    power

    :math:`n_{reactions} = \\frac{P_{fus}[MW]}{17.58 [MeV]eV[J]} [1/s]`

    Parameters
    ----------
    p_fus:
        D-T fusion power [MW]

    Returns
    -------
    Number of D-T reactions per second [1/s]
    """
    e_dt = E_DT_fusion()
    return raw_uc(p_fus, "MW", "W") / (e_dt * EV_TO_J)


def n_DD_reactions(p_fus: float) -> float:
    """
    Calculates the number of D-D fusion reactions per s for a given D-D fusion
    power

    :math:`n_{reactions} = \\frac{P_{fus}[MW]}{E_{DD} [MeV] eV[J]} [1/s]`

    Parameters
    ----------
    p_fus:
        D-D fusion power [W]

    Returns
    -------
    Number of D-D reactions per second [1/s]
    """
    e_dd = E_DD_fusion()
    return p_fus / (e_dd * EV_TO_J)


def r_T_burn(p_fus: float) -> float:  # noqa: N802
    """
    Calculates the tritium burn rate for a given fusion power

    :math:`\\dot{m_{b}} = \\frac{P_{fus}[MW]M_{T}[g/mol]}{17.58 [MeV]eV[J]N_{A}[1/mol]} [g/s]`

    Parameters
    ----------
    p_fus:
        D-T fusion power [MW]

    Returns
    -------
    T burn rate in the plasma [g/s]
    """  # noqa: W505, E501
    return n_DT_reactions(p_fus) * T_MOLAR_MASS / N_AVOGADRO


def r_D_burn_DT(p_fus: float) -> float:
    """
    Calculates the deuterium burn rate for a given fusion power in D-T

    Parameters
    ----------
    p_fus:
        D-T fusion power [MW]

    Returns
    -------
    D burn rate in the plasma [g/s]

    Notes
    -----
    .. math::
        \\dot{m_{b}} = \\frac{P_{fus}[MW]M_{D}[g/mol]}
        {17.58 [MeV]eV[J]N_{A}[1/mol]} [g/s]
    """
    return n_DT_reactions(p_fus) * D_MOLAR_MASS / N_AVOGADRO


class Reactions(Enum):
    """
    Reactions with support for reactivity.
    """

    D_T = auto()  # D + T --> 4He + n reaction
    D_D = auto()  # D + D --> 0.5 D-D1 + 0.5 D-D2
    D_D1 = auto()  # D + D --> 3He + n reaction [50 %]
    D_D2 = auto()  # D + D --> T + p reaction [50 %]
    D_He3 = auto()  # D + 3He --> 4He + p reaction


class ReactivityMethod(Enum):
    BOSCH_HALE = auto()
    PLASMOD = auto()
    JOHNER = auto()


def reactivity(
    temp_k: Union[float, np.ndarray],
    reaction: Union[str, Reactions] = Reactions.D_T,
    method: Union[str, ReactivityMethod] = ReactivityMethod.BOSCH_HALE,
) -> Union[float, np.ndarray]:
    """
    Calculate the thermal reactivity of a fusion reaction in Maxwellian plasmas,
    \\t:math:`<\\sigma v>`

    Parameters
    ----------
    temp_k:
        Temperature [K]
    reaction:
        The fusion reaction
    method:
        The parameterisation to use when calculating the reactivity

    Returns
    -------
    Reactivity of the reaction at the specified temperature(s) [m^3/s]
    """
    if not isinstance(reaction, Reactions):
        reaction = Reactions[reaction.replace("-", "_")]
    if not isinstance(method, ReactivityMethod):
        method = ReactivityMethod[method.replace("-", "_").upper()]
    mapping = {
        ReactivityMethod.BOSCH_HALE: _reactivity_bosch_hale,
        ReactivityMethod.PLASMOD: _reactivity_plasmod,
        ReactivityMethod.JOHNER: _reactivity_johner,
    }

    return mapping[method](raw_uc(temp_k, "K", "keV"), reaction)


@dataclass
class BoschHale_DT_4Hen:
    """
    Bosch-Hale parameterisation data for the reaction:

    D + T --> 4He + n

    H.-S. Bosch and G.M. Hale 1992 Nucl. Fusion 32 611
    DOI 10.1088/0029-5515/32/4/I07
    """

    t_min: float = 0.2  # [keV]
    t_max: float = 100  # [keV]
    bg: float = 34.3827  # [keV**0.5]
    mrc2: float = 1.124656e6  # [keV]
    c: npt.NDArray = field(
        default_factory=lambda: np.array(
            [
                1.17302e-9,
                1.51361e-2,
                7.51886e-2,
                4.60643e-3,
                1.35000e-2,
                -1.06750e-4,
                1.36600e-5,
            ]
        )
    )


@dataclass
class BoschHale_DD_3Hen:
    """
    Bosch-Hale parameterisation data for the reaction:

    D + D --> 3He + n

    H.-S. Bosch and G.M. Hale 1992 Nucl. Fusion 32 611
    DOI 10.1088/0029-5515/32/4/I07
    """

    t_min: float = 0.2  # [keV]
    t_max: float = 100  # [keV]
    bg: float = 31.3970  # [keV**0.5]
    mrc2: float = 0.937814e6  # [keV]
    c: npt.NDArray = field(
        default_factory=lambda: np.array(
            [
                5.43360e-12,
                5.85778e-3,
                7.68222e-3,
                0.0,
                -2.96400e-6,
                0.0,
                0.0,
            ]
        )
    )


@dataclass
class BoschHale_DD_Tp:
    """
    Bosch-Hale parameterisation data for the reaction:

    D + D --> T + p

    H.-S. Bosch and G.M. Hale 1992 Nucl. Fusion 32 611
    DOI 10.1088/0029-5515/32/4/I07
    """

    t_min: float = 0.2  # [keV]
    t_max: float = 100  # [keV]
    bg: float = 31.3970  # [keV**0.5]
    mrc2: float = 0.937814e6  # [keV]
    c: npt.NDArray = field(
        default_factory=lambda: np.array(
            [
                5.65718e-12,
                3.41267e-3,
                1.99167e-3,
                0.0,
                1.05060e-5,
                0.0,
                0.0,
            ]
        )
    )


@dataclass
class BoschHale_DHe3_4Hep:
    """
    Bosch-Hale parameterisation data for the reaction:

    D + 3He --> 4He + p

    H.-S. Bosch and G.M. Hale 1992 Nucl. Fusion 32 611
    DOI 10.1088/0029-5515/32/4/I07
    """

    t_min: float = 0.5  # [keV]
    t_max: float = 190  # [keV]
    bg: float = 68.7508  # [keV**0.5]
    mrc2: float = 1.124572e6  # [keV]
    c: npt.NDArray = field(
        default_factory=lambda: np.array(
            [
                5.51036e-10,
                6.41918e-3,
                -2.02896e-3,
                -1.91080e-5,
                1.35776e-4,
                0.0,
                0.0,
            ]
        )
    )


def _reactivity_bosch_hale(
    temp_kev: Union[float, np.ndarray], reaction: Reactions
) -> Union[float, np.ndarray]:
    """
    Bosch-Hale reactivity parameterisation for Maxwellian plasmas

    Parameters
    ----------
    temp_kev:
        Temperature [keV]
    reaction:
        The fusion reaction

    Returns
    -------
    Reactivity of the reaction at the specified temperature(s) [m^3/s]

    Notes
    -----
    H.-S. Bosch and G.M. Hale 1992 Nucl. Fusion 32 611
    DOI 10.1088/0029-5515/32/4/I07
    """
    if reaction == Reactions.D_D:
        return 0.5 * (
            _reactivity_bosch_hale(temp_kev, Reactions.D_D1)
            + _reactivity_bosch_hale(temp_kev, Reactions.D_D2)
        )
    mapping = {
        Reactions.D_T: BoschHale_DT_4Hen(),
        Reactions.D_D1: BoschHale_DD_3Hen(),
        Reactions.D_D2: BoschHale_DD_Tp(),
        Reactions.D_He3: BoschHale_DHe3_4Hep(),
    }
    data = mapping[reaction]

    if np.min(temp_kev) < data.t_min:
        bluemira_warn(
            f"The Bosch-Hale parameterisation for reaction {reaction} is only valid "
            f"between {data.t_min} and {data.t_max} keV, not {np.min(temp_kev)} keV."
        )
    if np.max(temp_kev) > data.t_max:
        bluemira_warn(
            f"The Bosch-Hale parameterisation for reaction {reaction} is only valid "
            f"between {data.t_min} and {data.t_max} keV, not {np.max(temp_kev)} keV."
        )

    frac = (
        temp_kev
        * (data.c[1] + temp_kev * (data.c[3] + temp_kev * data.c[5]))
        / (1 + temp_kev * (data.c[2] + temp_kev * (data.c[4] + temp_kev * data.c[6])))
    )
    theta = temp_kev / (1 - frac)
    chi = (data.bg**2 / (4 * theta)) ** (1 / 3)
    return (
        1e-6
        * data.c[0]
        * theta
        * np.sqrt(chi / (data.mrc2 * temp_kev**3))
        * np.exp(-3 * chi)
    )


def _reactivity_plasmod(
    temp_kev: Union[float, np.ndarray], reaction: Reactions
) -> Union[float, np.ndarray]:
    """
    Reactivity equations used in PLASMOD (original source unknown)

    Parameters
    ----------
    temp_kev:
        Temperature [keV]
    reaction:
        The fusion reaction

    Returns
    -------
    Reactivity of the reaction at the specified temperature(s) [m^3/s]
    """
    if reaction == Reactions.D_T:
        t3 = temp_kev ** (-1 / 3)

        term_1 = 8.972 * np.exp(-19.9826 * t3) * t3**2
        term_2 = (temp_kev + 1.0134) / (1 + 6.386e-3 * (temp_kev + 1.0134) ** 2)
        term_3 = 1.877 * np.exp(-0.16176 * temp_kev * np.sqrt(temp_kev))
        return 1e-19 * term_1 * (term_2 + term_3)

    if reaction == Reactions.D_D:
        term_1 = (
            0.16247 + 0.001741 * temp_kev - 0.029 * np.exp(-0.3843 * np.sqrt(temp_kev))
        )
        term_2 = np.exp(-18.8085 / (temp_kev ** (1 / 3))) / (temp_kev ** (1 / 3)) ** 2
        return 1e-19 * term_1 * term_2

    raise ValueError(
        f"This function only supports D-D and D-T, not {reaction.name.replace('_','-')}"
    )


def _reactivity_johner(
    temp_kev: Union[float, np.ndarray], reaction: Reactions
) -> Union[float, np.ndarray]:
    """
    Johner's monomial fit for analytical calculations

    Parameters
    ----------
    temp_kev:
        Temperature [keV]
    reaction:
        The fusion reaction

    Returns
    -------
    Reactivity of the reaction at the specified temperature(s) [m^3/s]

    Notes
    -----
    Johner, Jean (2011). HELIOS: a zero-dimensional tool for next step and reactor
    studies. Fusion Science and Technology, 59(2), 308-313. Appendix E.II
    """
    if reaction != Reactions.D_T:
        raise ValueError(
            f"This function only supports D-T, not {reaction.name.replace('_','-')}"
        )

    if np.max(temp_kev) > 100:  # noqa: PLR2004
        bluemira_warn("The Johner parameterisation is not valid for T > 100 keV")
    if np.min(temp_kev) < 5.3:  # noqa: PLR2004
        bluemira_warn("The Johner parameterisation is not valid for T < 5.3 keV")

    sigma_v = np.zeros_like(temp_kev)
    idx_1 = np.nonzero((temp_kev >= 5.3) & (temp_kev <= 10.3))[0]  # noqa: PLR2004
    idx_2 = np.nonzero((temp_kev >= 10.3) & (temp_kev <= 18.5))[0]  # noqa: PLR2004
    idx_3 = np.nonzero((temp_kev >= 18.5) & (temp_kev <= 39.9))[0]  # noqa: PLR2004
    idx_4 = np.nonzero((temp_kev >= 39.9) & (temp_kev <= 100.0))[0]  # noqa: PLR2004
    t1 = temp_kev[idx_1]
    t2 = temp_kev[idx_2]
    t3 = temp_kev[idx_3]

    sigma_v[idx_1] = 1.15e-25 * t1**3
    sigma_v[idx_2] = 1.18e-24 * t2**2
    sigma_v[idx_3] = 2.18e-23 * t3
    sigma_v[idx_4] = 8.69e-22
    if isinstance(temp_kev, (float, int)):
        return float(sigma_v)
    return sigma_v
