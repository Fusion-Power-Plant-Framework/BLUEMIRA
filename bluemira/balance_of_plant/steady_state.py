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
Crude 0-D steady-state balance of plant model. Mostly for visualisation purposes.
"""

import abc
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from bluemira.balance_of_plant.calculations import (
    H2O_pumping,
    He_pumping,
    superheated_rankine,
)
from bluemira.balance_of_plant.error import BalanceOfPlantError
from bluemira.balance_of_plant.plotting import BalanceOfPlantPlotter
from bluemira.base.constants import HE3_MOLAR_MASS, HE_MOLAR_MASS, NEUTRON_MOLAR_MASS
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter_frame import Parameter, ParameterFrame, make_parameter_frame


class CoolantPumping(abc.ABC):
    """
    Pumping power strategy abstract base class
    """

    @abc.abstractmethod
    def pump(self, power: float) -> Tuple[float, float]:
        """
        Calculate the pump work and electrical pumping power required for a given power.
        """


class HePumping(CoolantPumping):
    """
    He-cooling pumping power calculation strategy

    Parameters
    ----------
    pressure_in:
        Inlet pressure [Pa]
    pressure_out:
        Pressure drop [Pa]
    temp_in:
        Inlet temperature [K]
    temp_out:
        Outlet temperature [K]
    eta_isen:
        Isentropic efficiency of the He compressors
    eta_el:
        Electrical efficiency of the He compressors
    """

    def __init__(
        self,
        pressure_in: float,
        pressure_out: float,
        temp_in: float,
        temp_out: float,
        eta_isentropic: float,
        eta_electric: float,
    ):
        self.p_in = pressure_in
        self.p_out = pressure_out
        self.t_in = temp_in
        self.t_out = temp_out
        self.eta_isen = eta_isentropic
        self.eta_el = eta_electric

    def pump(self, power: float) -> Tuple[float, float]:
        """
        Calculate the pump work and electrical pumping power required for a given power.

        Parameters
        ----------
        power:
            Total blanket power excluding pumping power [W]

        Returns
        -------
        P_pump_is:
            The isentropic pumping power (added to the working fluid)
        P_pump_el:
            The electrical pumping power (parasitic load)
        """
        p_pump, p_electric = He_pumping(
            self.p_in,
            self.p_out,
            self.t_in,
            self.t_out,
            power,
            self.eta_isen,
            self.eta_el,
        )
        return p_pump, p_electric


class H2OPumping(CoolantPumping):
    """
    H20-cooling pumping power calculation strategy

    Parameters
    ----------
    f_pump:
        Fraction of thermal power required to pump
    eta_isen:
        Isentropic efficiency of the water pumps
    eta_el:
        Electrical efficiency of the water pumps
    """

    def __init__(self, f_pump: float, eta_isentropic: float, eta_electric: float):
        self.f_pump = f_pump
        self.eta_isen = eta_isentropic
        self.eta_el = eta_electric

    def pump(self, power: float) -> Tuple[float, float]:
        """
        Calculate the pump work and electrical pumping power required for a given power.

        Parameters
        ----------
        power:
            Total blanket power excluding pumping power [W]

        Returns
        -------
        P_pump_is:
            The isentropic pumping power (added to the working fluid)
        P_pump_el:
            The eletrical pumping power (parasitic load)
        """
        p_pump, p_electric = H2O_pumping(power, self.f_pump, self.eta_isen, self.eta_el)
        return p_pump, p_electric


class PowerCycleEfficiencyCalc(abc.ABC):
    """
    Power cycle efficiency calculation abstract base class
    """

    @abc.abstractmethod
    def calculate(self, *args) -> float:
        """
        Calculate the efficiency of the power cycle
        """


class PredeterminedEfficiency(PowerCycleEfficiencyCalc):
    """
    Predetermined efficiency 'calculation'

    Parameters
    ----------
    efficiency:
        The efficiency to 'calculate'
    """

    def __init__(self, efficiency: float):
        self.efficiency = efficiency

    def calculate(self, p_blanket: float, p_divertor: float) -> float:  # noqa: ARG002
        """
        Calculate the efficiency of the power cycle

        Parameters
        ----------
        p_blanket:
            Blanket thermal power [MW]
        p_divertor:
            Divertor thermal power [MW]

        Returns
        -------
        The efficiency
        """
        return self.efficiency


class SuperheatedRankine(PowerCycleEfficiencyCalc):
    """
    Superheated Rankine power cycle for use with gas coolants

    Parameters
    ----------
    bb_t_out:
        Breeding blanket outlet temperature [K]
    delta_t_turbine:
        Turbine inlet delta T [K]
    """

    def __init__(self, bb_t_out: float, delta_t_turbine: float):
        self.bb_t_out = bb_t_out
        self.delta_t_turbine = delta_t_turbine

    def calculate(self, p_blanket: float, p_divertor: float) -> float:
        """
        Calculate the efficiency of the power cycle

        Parameters
        ----------
        p_blanket:
            Blanket thermal power [MW]
        p_divertor:
            Divertor thermal power [MW]

        Returns
        -------
        The efficiency
        """
        return superheated_rankine(
            p_blanket, p_divertor, self.bb_t_out, self.delta_t_turbine
        )


class FractionSplitStrategy(abc.ABC):
    """
    Strategy ABC for splitting flows according to fractions.
    """

    @abc.abstractmethod
    def split(self, *args):
        """
        Split flows somehow.
        """

    def check_fractions(self, fractions: List[float]) -> bool:
        """
        Check that fractions sum to 1.0

        Raises
        ------
        BalanceOfPlantError
            If they do not
        """
        frac_sum = sum(fractions)
        if not np.isclose(frac_sum, 1.0, atol=1e-6, rtol=0):
            raise BalanceOfPlantError(
                f"{self.__class__.__name__} fractions sum to {frac_sum:.7f}"
            )


class NeutronPowerStrategy(FractionSplitStrategy):
    """
    Strategy for distributing neutron power among components

    Parameters
    ----------
    f_blanket:
        Fraction of neutron power going to the blankets
    f_divertor:
        Fraction of neutron power going to the divertors
    f_vessel:
        Fraction of neutron power going to the vacuum vessel
    f_other:
        Fraction of neutron power going to other systems
    energy_multiplication:
        Energy multiplication factor applied to blanket neutron power
    decay_multiplication:
        Decay energy multiplication applied to the blanket neutron power
    """

    def __init__(
        self,
        f_blanket: float,
        f_divertor: float,
        f_vessel: float,
        f_other: float,
        energy_multiplication: float,
        decay_multiplication: float,
    ):
        self.check_fractions([f_blanket, f_divertor, f_vessel, f_other])
        self.f_blanket = f_blanket
        self.f_divertor = f_divertor
        self.f_vessel = f_vessel
        self.f_other = f_other
        if energy_multiplication < 1.0:  # noqa: PLR2004
            raise BalanceOfPlantError(
                "Energy multiplication factor cannot be less than 1.0"
            )
        if decay_multiplication < 1.0:  # noqa: PLR2004
            raise BalanceOfPlantError(
                "Decay multiplication factor cannot be less than 1.0"
            )
        self.nrg_mult = energy_multiplication
        self.dec_mult = decay_multiplication

    def split(
        self, neutron_power: float
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Split neutron power into several flows

        Parameters
        ----------
        neutron_power:
            Total neutron power

        Returns
        -------
        blk_power:
            Neutron power to blankets
        div_power:
            Neutron power to divertors
        vv_power:
            Neutron power to vessel
        aux_power:
            Neutron power to auxiliary systems
        mult_power:
            Energy multiplication power which is assumed to come solely from the blanket
        decay_power:
            Decay power which is assumed to come solely from the blanket
        """
        blk_power = self.f_blanket * self.nrg_mult * neutron_power
        div_power = self.f_divertor * neutron_power
        vv_power = self.f_vessel * neutron_power
        aux_power = self.f_other * neutron_power
        mult_power = blk_power - self.f_blanket * neutron_power
        decay_power = (self.dec_mult - 1.0) * neutron_power
        return blk_power, div_power, vv_power, aux_power, mult_power, decay_power


class RadChargedPowerStrategy(FractionSplitStrategy):
    """
    Strategy for distributing radiation and charged particle power from the plasma
    core and scrape-off layer

    Parameters
    ----------
    f_core_rad_fw:
        Fraction of core radiation power distributed to the first wall
    f_sol_rad:
        Fraction of SOL power that is radiated
    f_sol_rad_fw:
        Fraction of radiated SOL power that is distributed to the first wall
    f_sol_ch_fw:
        Fraction of SOL charged particle power that is distributed to the first wall
    f_fw_aux:
        Fraction of first power that actually goes into auxiliary systems
    """

    def __init__(
        self,
        f_core_rad_fw: float,
        f_sol_rad: float,
        f_sol_rad_fw: float,
        f_sol_ch_fw: float,
        f_fw_aux: float,
    ):
        self.f_core_rad_fw = f_core_rad_fw
        self.f_sol_rad = f_sol_rad
        self.f_sol_rad_fw = f_sol_rad_fw
        self.f_sol_ch_fw = f_sol_ch_fw
        self.f_fw_aux = f_fw_aux

    def split(
        self, p_radiation: float, p_separatrix: float
    ) -> Tuple[float, float, float]:
        """
        Split the radiation and charged particle power

        Parameters
        ----------
        p_radiation:
            Plasma core radiation power
        p_separatrix:
            Charged particle power crossing the separatrix

        Returns
        -------
        p_rad_sep_blk:
            Radiation power from separatrix to blanket
        p_rad_sep_div:
            Radiation power from separatrix to divertor
        p_rad_sep_aux:
            Radiation power from separatrix to auxiliaries
        """
        # Core radiation
        p_core_rad_fw = p_radiation * self.f_core_rad_fw
        p_core_rad_div = p_radiation - p_core_rad_fw

        # Scrape-off layer radiation
        p_sol_rad = p_separatrix * self.f_sol_rad
        p_sol_rad_fw = p_sol_rad * self.f_sol_rad_fw
        p_sol_rad_div = p_sol_rad - p_sol_rad_fw

        # Scrape-off layer charged particles
        p_sol_charged = p_separatrix - p_sol_rad
        p_sol_charged_fw = p_sol_charged * self.f_sol_ch_fw
        p_sol_charged_div = p_sol_charged - p_sol_charged_fw

        # Split first wall into blanket and auxiliary
        p_rad_sep_fw = p_core_rad_fw + p_sol_rad_fw + p_sol_charged_fw
        p_rad_sep_blk = p_rad_sep_fw * (1 - self.f_fw_aux)
        p_rad_sep_aux = p_rad_sep_fw - p_rad_sep_blk
        p_rad_sep_div = p_core_rad_div + p_sol_rad_div + p_sol_charged_div
        return p_rad_sep_blk, p_rad_sep_div, p_rad_sep_aux


class ParasiticLoadStrategy(abc.ABC):
    """
    Strategy for calculating the parasitic loads
    """

    @abc.abstractmethod
    def calculate(*args, **kwargs) -> Tuple[float, float, float, float]:
        """
        Calculate the parasitic loads somehow

        Returns
        -------
        p_mag:
            Parasitic loads to power the magnets
        p_cryo:
            Parasitic loads to power the cryoplant
        p_t_plant:
            Parasitic loads to power the tritium plant
        p_other:
            Parasitic loads to power other miscellaneous things
        """


@dataclass
class BoPModelParams(ParameterFrame):
    """Parameters required to run :class:`BalanceOfPlantModel`."""

    P_fus_DT: Parameter[float]
    P_fus_DD: Parameter[float]
    P_rad: Parameter[float]
    P_hcd_ss: Parameter[float]
    P_hcd_ss_el: Parameter[float]


class BalanceOfPlantModel:
    """
    Balance of plant calculator for a fusion power reactor.

    Parameters
    ----------
    params:
        Structure containing input parameters.
        If this is a dictionary, required keys are:

            * P_fus_DT: float
            * P_fus_DD: float
            * P_rad: float
            * P_hcd_ss: float
            * P_hcd_ss_el: float

        See :class:`BoPModelParams` for parameter details.
    rad_sep_strat:
        Strategy to calculate the where the radiation and charged particle power
        in the scrape-off-layer is carried to
    neutron_strat:
        Strategy to calculate where the neutron power is carried to
    blanket_pump_strat:
        Strategy to calculate the coolant pumping power for the blanket
    divertor_pump_strat:
        Strategy to calculate the coolant pumping power for the divertor
    bop_cycle_strat:
        Strategy to calculate the balance of plant thermal efficiency
    parasitic_load_strat:
        Strategy to calculate the parasitic loads

    Notes
    -----
    .. math::
        P_{el}={\\eta}_{BOP}\\Bigg[\\Bigg(\\frac{4}{5}P_{fus}f_{nrgm}-\\
        P_{n_{aux}}-P_{n_{DIV}}+f_{SOL_{rad}}f_{SOL_{ch}}\\Big(\\frac{P_{fus}}{5}+P_{HCD}\\Big)\\Bigg)\\
        \\Big(1+\\frac{f_{p_{BB}}}{1-f_{p_{BB}}}\\Big)
        +\\Bigg(P_{n_{DIV}}+f_{SOL_{rad}}f_{SOL_{ch}}f_{fw}\\Big(\\frac{P_{fus}}{5}+P_{HCD}\\Big)\\Bigg)\\
        \\Big(1+\\frac{f_{p_{DIV}}}{1-f_{p_{DIV}}}\\Big)\\Bigg]

    """

    _plotter = BalanceOfPlantPlotter

    def __init__(
        self,
        params: Union[Dict[str, float], BoPModelParams],
        rad_sep_strat: FractionSplitStrategy,
        neutron_strat: FractionSplitStrategy,
        blanket_pump_strat: CoolantPumping,
        divertor_pump_strat: CoolantPumping,
        bop_cycle_strat: PowerCycleEfficiencyCalc,
        parasitic_load_strat: ParasiticLoadStrategy,
    ):
        self.params = make_parameter_frame(params, BoPModelParams)
        self.rad_sep_strat = rad_sep_strat
        self.neutron_strat = neutron_strat
        self.blanket_pump_strat = blanket_pump_strat
        self.divertor_pump_strat = divertor_pump_strat
        self.bop_strat = bop_cycle_strat
        self.parasitic_strat = parasitic_load_strat
        self.flow_dict = None

    def build(self):
        """
        Carry out the balance of plant calculation
        """
        p_fusion = self.params.P_fus_DT.value + self.params.P_fus_DD.value
        f_neutron_DT = HE_MOLAR_MASS / (HE_MOLAR_MASS + NEUTRON_MOLAR_MASS)
        f_neutron_DD = 0.5 * HE3_MOLAR_MASS / (NEUTRON_MOLAR_MASS + HE3_MOLAR_MASS)
        p_neutron = (
            f_neutron_DT * self.params.P_fus_DT.value
            + f_neutron_DD * self.params.P_fus_DD.value
        )
        p_charged = self.params.P_fus_DT.value + self.params.P_fus_DD.value - p_neutron

        p_radiation = self.params.P_rad.value
        p_hcd = self.params.P_hcd_ss.value
        p_hcd_el = self.params.P_hcd_ss_el.value
        p_separatrix = p_charged - p_radiation + p_hcd
        (
            p_n_blk,
            p_n_div,
            p_n_vv,
            p_n_aux,
            p_nrgm,
            p_blk_decay,
        ) = self.neutron_strat.split(p_neutron)
        p_rad_sep_blk, p_rad_sep_div, p_rad_sep_aux = self.rad_sep_strat.split(
            p_radiation, p_separatrix
        )
        p_rad_sep_fw = p_rad_sep_blk + p_rad_sep_aux

        p_blanket = p_n_blk + p_blk_decay + p_rad_sep_blk
        p_blk_pump, p_blk_pump_el = self.blanket_pump_strat.pump(p_blanket)
        p_blanket += p_blk_pump

        p_div = p_n_div + p_rad_sep_div
        p_div_pump, p_div_pump_el = self.divertor_pump_strat.pump(p_div)
        p_div += p_div_pump

        eta_bop = self.bop_strat.calculate(p_blanket, p_div)
        total_bop_power = p_blanket + p_div
        p_bop = eta_bop * total_bop_power
        p_bop_loss = total_bop_power - p_bop

        p_mag, p_cryo, p_t_plant, p_other = self.parasitic_strat.calculate(p_fusion)

        p_el_net = (
            p_bop
            - p_hcd_el
            - p_blk_pump_el
            - p_cryo
            - p_mag
            - p_other
            - p_t_plant
            - p_div_pump_el
        )

        self.flow_dict = {
            "Plasma": [p_fusion, p_hcd, -p_neutron, -p_separatrix - p_radiation],
            "H&CD": [p_hcd_el, -p_hcd, -(p_hcd_el - p_hcd)],
            "Neutrons": [p_neutron, p_nrgm, -p_n_blk, -p_n_div, -p_n_vv - p_n_aux],
            "Radiation and \nseparatrix": [
                p_radiation + p_separatrix,
                -p_rad_sep_fw,
                -p_rad_sep_div,
            ],
            "Blanket": [p_n_blk, p_rad_sep_blk, p_blk_pump, p_blk_decay, -p_blanket],
            "Divertor": [p_n_div, p_rad_sep_div, -p_n_div - p_rad_sep_div],
            "First wall": [p_rad_sep_fw, -p_rad_sep_aux, -p_rad_sep_blk],
            "BoP": [p_blanket, p_div, -p_bop_loss, -p_bop],
            "Electricity": [
                p_bop,
                -p_t_plant,
                -p_other,
                -p_cryo,
                -p_mag,
                -p_hcd_el,
                -p_div_pump_el,
                -p_blk_pump_el,
                -p_el_net,
            ],
            "_H&CD loop": [p_hcd_el, -p_hcd_el],
            "_Divertor 2": [p_rad_sep_div, -p_rad_sep_div],
            "_H&CD loop 2": [p_hcd_el, -p_hcd_el],
            "_DIV to BOP": [p_div - p_div_pump, p_div_pump, -p_div],
            "_BB coolant loop turn": [
                p_blk_pump_el,
                -p_blk_pump_el + p_blk_pump,
                -p_blk_pump,
            ],
            "_BB coolant loop blanket": [p_blk_pump, -p_blk_pump],
            "_DIV coolant loop turn": [
                p_div_pump_el,
                -p_div_pump_el + p_div_pump,
                -p_div_pump,
            ],
            "_DIV coolant loop divertor": [p_div_pump, -p_div_pump],
        }
        self.sanity()

    def sanity(self):
        """
        Perform a series of sanity checks.
        """
        delta_truth = 0
        # Per block check
        for label, flow in self.flow_dict.items():
            delta = sum(flow)
            if round(delta) != 0:
                bluemira_warn(
                    f"Power block {label} is not self-consistent.. {delta:.2f} MW are"
                    " missing"
                )
            delta_truth += delta

        # Global check
        if round(delta_truth) != 0:
            bluemira_warn(
                f"The balance of plant model is inconsistent: {delta_truth:.2f} MW are"
                " lost somewhere."
            )

    def plot(self, title: Optional[str] = None, **kwargs):
        """
        Plot the BalanceOfPlant object.

        Parameters
        ----------
        title:
            Title to print on the plot

        Other Parameters
        ----------------
        see BALANCE_PLOT_DEFAULTS for details
        """
        plotter = self._plotter(**kwargs)
        return plotter.plot(self.flow_dict, title=title)
