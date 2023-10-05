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
Fusion power reactor lifecycle object.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from bluemira.base.constants import S_TO_YR, YR_TO_S, raw_uc
from bluemira.base.look_and_feel import bluemira_print, bluemira_warn
from bluemira.fuel_cycle.timeline import Timeline
from bluemira.utilities.tools import abs_rel_difference, is_num, json_writer

if TYPE_CHECKING:
    from bluemira.fuel_cycle.timeline_tools import (
        LearningStrategy,
        OperationalAvailabilityStrategy,
    )

__all__ = ["LifeCycle"]


class LifeCycle:
    """
    A life cycle object for a fusion reactor.

    Parameters
    ----------
    config:
        Parameters for the reactor life cycle. See
        :class:`bluemira.fuel_cycle.lifecycle.LifeCycleParams` for
        parameters details.
        Set to an empty ``dict`` to use default values.
    learning_strategy:
        A concrete instance of a ``LearningStrategy`` for distributing
        the total operational availability over different operational
        phases.
    availability_strategy:
        Concrete instance of an OperationalAvailabilityStrategy
        implementing a strategy to generate distributions of unplanned
        outages.
    inputs:
        Currently unused.
    """

    def __init__(
        self,
        config: Union[LifeCycleParams, Dict[str, float]],
        learning_strategy: LearningStrategy,
        availability_strategy: OperationalAvailabilityStrategy,
        inputs: dict,
    ):
        self.learning_strategy = learning_strategy
        self.availability_strategy = availability_strategy
        self.inputs = inputs

        if isinstance(config, LifeCycleParams):
            self.params = config
        elif isinstance(config, dict):
            self.params = LifeCycleParams(**config)
        else:
            raise TypeError(
                "Invalid type for 'params'. Must be one of 'dict', or "
                f"'LifeCycleParams'; found '{type(config).__name__}'."
            )

        # Constructors
        self.total_planned_maintenance = None
        self.total_ramptime = None
        self.t_unplanned_m = None
        self.t_on_total = None
        self.t_interdown = None
        self.cs_down = None
        self.unplanned = None
        self.min_downtime = None
        self.T = None
        self.a_ops = None
        self.phase_names = None
        self.phase_durations = None
        self.n_blk_replace = None
        self.n_div_replace = None
        self.fpy = None
        self.tf_lifeend = None
        self.vv_lifeend = None
        self.A_global = None
        self.n_cycles = None  # Total number of D-T pulses

        # Derive/convert inputs
        self.maintenance_l = self.params.bmd  # [s]
        self.maintenance_s = self.params.dmd  # [s]
        self.t_rampup = self.params.I_p / self.params.s_ramp_up  # [s]
        self.t_rampdown = self.params.I_p / self.params.s_ramp_down  # [s]
        self.t_flattop = self.params.t_pulse - self.t_rampup - self.t_rampdown  # [s]
        self.t_min_down = max(self.params.t_cs_recharge, self.params.t_pumpdown)

        # Build timeline
        self.life_neutronics()
        self.set_availabilities(self.params.A_global)

    def life_neutronics(self):
        """
        Calculate the lifetime of various components based on their damage limits
        and fluences.
        """
        tf_ins_nflux = self.params.tf_ins_nflux
        divl = self.params.div_dpa / self.params.div_dmg  # [fpy] Divertor life
        blk1l = self.params.blk_1_dpa / self.params.blk_dmg  # [fpy] 1st Blanket life
        blk2l = self.params.blk_2_dpa / self.params.blk_dmg  # [fpy] 2nd Blanket life

        # Number of divertor changes in 1st blanket life
        ndivch_in1blk = int(blk1l / divl)
        # Number of divertor changes in 2nd blanket life (1 for div reset)
        ndivch_in2blk = int(blk2l / divl)

        self.n_blk_replace = 1  # HLR
        self.n_div_replace = ndivch_in1blk + ndivch_in2blk
        m_short = self.maintenance_s * S_TO_YR
        m_long = self.maintenance_l * S_TO_YR
        phases = []
        for i in range(ndivch_in1blk):
            p_str = "Phase P1." + str(i + 1)
            m_str = "Phase M1." + str(i + 1)
            phases.extend(([divl, p_str], [m_short, m_str]))
            count = i
        if ndivch_in1blk == 0:
            count = 0
        phases.extend(
            (
                [blk1l % divl, "Phase P1." + str(count + 2)],
                [m_long, "Phase M1." + str(count + 2)],
            )
        )
        for i in range(ndivch_in2blk):
            p_str = "Phase P2." + str(i + 1)
            m_str = "Phase M2." + str(i + 1)
            phases.extend(([divl, p_str], [m_short, m_str]))
            count2 = i
        phases.append([blk2l % divl, "Phase P2." + str(count2 + 1)])
        self.phase_durations = [p[0] for p in phases]
        self.phase_names = [p[1] for p in phases]
        self.calc_n_pulses(phases)
        fpy = 0
        for i in range(len(phases)):
            if phases[i][1].startswith("Phase P"):
                fpy += phases[i][0]
        self.fpy = fpy
        # Irreplaceable components life checks
        self.t_on_total = self.fpy * YR_TO_S  # [s] total fusion time
        tf_ins_life_dose = tf_ins_nflux * self.t_on_total / self.params.tf_fluence
        if tf_ins_life_dose > 1:
            self.tf_lifeend = round(self.params.tf_fluence / tf_ins_nflux / YR_TO_S, 2)
            tflifeperc = round(100 * self.tf_lifeend / self.fpy, 1)
            bluemira_warn(
                f"TF coil insulation fried after {self.tf_lifeend:.2f} full-power years"
                f", or {tflifeperc:.2f} % of neutron budget."
            )
        vv_life_dmg = self.params.vv_dmg * self.fpy / self.params.vv_dpa
        if vv_life_dmg > 1:
            self.vv_lifeend = round(self.params.vv_dpa / self.params.vv_dmg, 2)
            vvlifeperc = round(100 * self.vv_lifeend / self.fpy, 1)
            bluemira_warn(
                f"VV fried after {self.vv_lifeend:.2f} full-power"
                f" years, or {vvlifeperc:.2f} % of neutron budget."
            )
            # TODO: treat output parameter
        self.n_cycles = self.fpy * YR_TO_S / self.t_flattop

    def set_availabilities(self, load_factor: float):
        """
        Sets availability and distributes it between the two phases of planned operation.
        The planned maintenance windows are subtracted from the availability which
        needs to be achieved during the phase of operation. The target overall plant
        lifetime availability as specified in input parameter A remains the same.

        Notes
        -----
        \t:math:`A_{overall}=\\dfrac{t_{on}}{t_{on}+t_{off}}`
        \t:math:`A_{operations}=\\dfrac{t_{on}}{t_{on}+t_{ramp}+t_{CS_{recharge}}+t_{m_{unplanned}}}`
        """
        self.total_planned_maintenance = self.maintenance_l * self.n_blk_replace + (
            self.maintenance_s * self.n_div_replace
        )
        self.t_interdown = sum(self.n_pulse_p) * self.t_min_down
        self.total_ramptime = sum(self.n_pulse_p) * (self.t_rampup + self.t_rampdown)
        self.min_downtime = (
            self.total_planned_maintenance + self.t_interdown + self.total_ramptime
        )
        self.unplanned = self.t_on_total / load_factor - (
            self.total_planned_maintenance
            + self.t_interdown
            + self.total_ramptime
            + self.t_on_total
        )

        # TODO: Treat global load factor vs lifetime operational availability properly..!
        op_durations = self.get_op_phases()
        self.a_ops = self.learning_strategy.generate_phase_availabilities(
            self.params.A_global, op_durations
        )

    def calc_n_pulses(self, phases: List[List[float]]):
        """
        Calculate the number of pulses per phase.
        """
        self.n_pulse_p = [
            int(YR_TO_S * phases[i][0] // self.t_flattop)
            for i in range(len(phases))
            if phases[i][1].startswith("Phase P")
        ]

    def get_op_phases(self) -> List[float]:
        """
        Get the operational phases for the LifeCycle.
        """
        return [
            d for n, d in zip(self.phase_names, self.phase_durations) if "Phase P" in n
        ]

    def make_timeline(self) -> Timeline:
        """
        Builds a Timeline instance
        """
        n = len(self.n_pulse_p)

        for k in ["t_rampup", "t_flattop", "t_rampdown", "t_min_down"]:
            v = getattr(self, k)
            if is_num(v):
                setattr(self, k, v * np.ones(n))

        n_DT_reactions = self.params.n_DT_reactions * np.ones(n)
        n_DD_reactions = self.params.n_DD_reactions * np.ones(n)
        Ip = self.params.I_p * np.ones(n)

        timeline = Timeline(
            self.phase_names,
            self.phase_durations,
            self.a_ops,
            self.n_pulse_p,
            self.t_rampup,
            self.t_flattop,
            self.t_rampdown,
            self.t_min_down,
            n_DT_reactions,
            n_DD_reactions,
            Ip,
            self.params.A_global,
            self.params.blk_dmg,
            self.params.blk_1_dpa,
            self.params.blk_2_dpa,
            self.params.div_dmg,
            self.params.div_dpa,
            self.params.tf_ins_nflux,
            self.params.tf_fluence,
            self.params.vv_dmg,
            self.params.vv_dpa,
            self.availability_strategy,
        )
        self.T = timeline
        self.t_unplanned_m = self.T.t_unplanned_m
        return self.T

    def sanity(self):
        """
        Perform a sanity check. Will raise warnings if the LifeCycle generates
        results that violate the tolerances.
        """
        life = self.fpy / self.params.A_global
        actual_life = S_TO_YR * (
            self.t_on_total
            + self.total_ramptime
            + self.t_interdown
            + self.total_planned_maintenance
            + self.t_unplanned_m
        )
        actual_lf = self.fpy / actual_life
        delt = abs_rel_difference(actual_life, life)
        delta2 = abs_rel_difference(actual_lf, self.params.A_global)
        if delt > 0.015:  # noqa: PLR2004
            bluemira_warn(
                "FuelCycle::Lifecyle: discrepancy between actual and planned\n"
                "reactor lifetime\n"
                f"Actual: {actual_life:.2f}\n"
                f"Planned: {life:.2f}\n"
                f"% diff: {100*delt:.4f}\n"
                "the problem is probably related to unplanned maintenance."
            )
            self.__init__(
                self.params,
                self.learning_strategy,
                self.availability_strategy,
                self.inputs,
            )  # Phoenix

        if delta2 > 0.015:  # noqa: PLR2004
            bluemira_warn(
                "FuelCycle::Lifecyle: availability discrepancy greater than\n"
                "specified tolerance\n"
                f"Actual: {actual_lf:.4f}\n"
                f"Planned: {self.params.A_global:.4f}\n"
                f"% diff: {100*delta2:.4f}\n"
                "the problem is probably related to unplanned maintenance."
            )
            self.__init__(
                self.params,
                self.learning_strategy,
                self.availability_strategy,
                self.inputs,
            )  # Phoenix

        if self.params.A_global > self.fpy / (self.fpy + S_TO_YR * self.min_downtime):
            bluemira_warn("FuelCycle::Lifecyle: Input availability is unachievable.")
        # Re-assign A
        self.params.A_global = actual_lf

    def summary(self):
        """
        Plot the load factor breakdown and learning curve
        """
        f, ax = plt.subplots(1, 2)
        self.plot_learning(ax=ax[0])
        self.plot_load_factor(ax=ax[1])

    def plot_life(self):
        """
        Plot the different maintenance events in the Lifecycle, and the slope
        of each operational phase
        """
        f, ax = plt.subplots(1, 1)
        # Plotting crutches
        fs, s, h = 0, 0, 0.95 * self.fpy
        ft, rt = [0], [0]
        j = 0
        for p_n, p_d in zip(self.phase_names, self.phase_durations):
            if p_n.startswith("Phase P"):
                c = "b"
                ft.append(fs + p_d)
                fs += p_d
                length = p_d / self.a_ops[j]
                rt.append(s + length)
                j += 1
            elif p_n.startswith("Phase M"):
                ft.append(ft[-1])
                c = "r"
                length = p_d
                rt.append(rt[-1] + p_d)
                m = "s" if p_n.startswith("Phase M1.2") else "o"
                ax.plot(rt[-1] - p_d / 2, 0.3, marker=m, color="r", ms=25)
                ax.axvspan(s, s + length, color=c, alpha=0.2)
            h -= 0.08 * self.fpy
            s += length

        legend = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Divertor replacement",
                markerfacecolor="r",
                markersize=25,
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label="Blanket and divertor \nreplacement",
                markerfacecolor="r",
                markersize=25,
            ),
        ]
        ax.legend(handles=legend)
        ax.plot(rt, ft, color="#0072bd")
        ax.set_xlabel("Elapsed plant lifetime [years]")
        ax.set_ylabel("Full power years [fpy]")
        ax.set_xlim([0, self.fpy / self.params.A_global])
        ax.set_ylim(bottom=0)

    def plot_load_factor(self, typ: str = "pie", ax: Optional[plt.Axes] = None):
        """
        Plots a pie or bar chart of the breakdown of the reactor lifetime

        Parameters
        ----------
        typ:
            Whether to plot a pie or bar chart ['pie', 'bar']
        """
        if ax is None:
            ax = plt.gca()
        c = ["#0072bd", "#d95319", "#edb120", "#7e2f8e", "#77ac30"]
        labels = [
            "Fusion time",
            "Ramp-up and\nramp-down",
            "CS recharge",
            "Planned\nmaintenance",
            "Unallocated\ndowntime",
        ]
        sizes = [
            self.t_on_total,
            self.total_ramptime,
            self.t_interdown,
            self.total_planned_maintenance,
            self.t_unplanned_m,
        ]
        if typ == "pie":
            plt.pie(
                sizes,
                labels=labels,
                colors=c,
                startangle=90,
                autopct="%.2f",
                counterclock=False,
            )
            plt.axis("equal")
        elif typ == "bar":
            bottom = 0
            for i, s in enumerate(sizes):
                ax.bar(1, s / sum(sizes) * 100, bottom=bottom, label=labels[i])
                bottom += s / sum(sizes) * 100
            ax.set_xticklabels([""])
            ax.set_xlim([0, 6])
            ax.legend()
        plt.title(
            f"Breakdown of DEMO reactor lifetime\n A = {self.params.A_global:.2f},"
            f"{self.fpy:.2f} fpy, {self.T.plant_life:.2f} years"
        )

    def write(self, filename: str, **kwargs):
        """
        Save a Timeline to a JSON file.
        """
        bluemira_print(f"Writing {filename}")
        data = self.T.to_dict()
        return json_writer(data, filename, **kwargs)

    @staticmethod
    def read(filename: str):
        """
        Load a Timeline from a JSON file.
        """
        bluemira_print(f"Reading {filename}")
        with open(filename) as f_h:
            return json.load(f_h)


@dataclass
class LifeCycleParams:
    """Parameters for running :class:`bluemira.fuel_cycle.lifecycle.LifeCycle`"""

    A_global: float = 0.3
    """Global load factor [dimensionless]. Not always used."""

    I_p: float = 19e6
    """Plasma current [A]. None."""

    bmd: float = field(default_factory=lambda: raw_uc(150, "days", "s"))
    """Blanket maintenance duration [days]. Full replacement intervention duration."""

    dmd: float = field(default_factory=lambda: raw_uc(90, "days", "s"))
    """Divertor maintenance duration [s]. Full replacement intervention duration."""

    t_pulse: float = 7200
    """Pulse length [s]. Includes ramp-up and ramp-down time."""

    t_cs_recharge: float = 600
    """CS recharge time [s]. Presently assumed to dictate minimum dwell period."""

    t_pumpdown: float = 599
    """
    Pump down duration of the vessel in between pulses [s]. Presently assumed to
    take less time than the CS recharge.
    """

    s_ramp_up: float = 1e5
    """Plasma current ramp-up rate [A/s]. None."""

    s_ramp_down: float = 1e5
    """Plasma current ramp-down rate [A/s]. None."""

    n_DT_reactions: float = 7.078779946428698e20
    """D-T fusion reaction rate [1/s]. At full power."""

    n_DD_reactions: float = 8.548069652616976e18
    """D-D fusion reaction rate [1/s]. At full power."""

    blk_1_dpa: float = 20
    """
    Starter blanket life limit (EUROfer) [dpa].
    https://iopscience.iop.org/article/10.1088/1741-4326/57/9/092002/pdf.
    """

    blk_2_dpa: float = 50
    """
    Second blanket life limit (EUROfer) [dpa].
    https://iopscience.iop.org/article/10.1088/1741-4326/57/9/092002/pdf.
    """

    div_dpa: float = 5
    """
    Divertor life limit (CuCrZr) [dpa].
    https://iopscience.iop.org/article/10.1088/1741-4326/57/9/092002/pdf.
    """

    vv_dpa: float = 3.25
    """Vacuum vessel life limit (SS316-LN-IG) [dpa]. RCC-Mx or whatever it is called."""

    tf_fluence: float = 3.2e21
    """
    Insulation fluence limit for ITER equivalent to 10 MGy [1/m^2].
    https://ieeexplore.ieee.org/document/6374236/.
    """

    tf_ins_nflux: float = 1.4e13
    """
    TF insulation peak neutron flux [1/m^2/s]. Pavel Pereslavtsev sent me an email
    20/02/2017.
    """

    blk_dmg: float = 10.2
    """Blanket neutron damage rate [dpa/fpy]. Pavel Pereslavtsev 2M7HN3 fig. 20."""

    div_dmg: float = 3
    """
    Divertor neutron damage rate [dpa/fpy].
    https://iopscience.iop.org/article/10.1088/1741-4326/57/9/092002/pdf.
    """

    vv_dmg: float = 0.3
    """Vacuum vessel neutron damage rate [dpa/fpy]. Pavel Pereslavtsev 2M7HN3 fig. 18."""
