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
Partially randomised fusion reactor load signal object and tools
"""
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.constants import S_TO_YR, YR_TO_S
from bluemira.fuel_cycle.timeline_tools import (
    LogNormalAvailabilityStrategy,
    OperationalAvailabilityStrategy,
)

__all__ = ["Timeline"]


class Phase:
    """
    Abstract object parent mixin class for Phases

    Attributes
    ----------
    t: calendar time [years]
    ft: fusion time [years]
    inventory: plasma current [A]
    DD_rate: D-D fusion rate [1/s]
    DT_rate: D-T fusion rate [1/s]
    """

    def __init__(self):
        self.t = None
        self.ft = None
        self.inventory = None
        self.DD_rate = None
        self.DT_rate = None

    def plot(self):
        """
        Plot the Phase.
        """
        _, ax = plt.subplots()
        ax.plot(self.t, self.inventory)
        ax2 = ax.twinx()
        ax2.plot(self.t, self.DT_rate, label="D-T")
        ax2.plot(self.t, self.DD_rate, label="D-D")
        ax2.set_xlabel("$t$ [s]")
        ax2.legend()


class OperationPhase(Phase):
    """
    Object describing an operational phase in a reactor LifeCycle

    Parameters
    ----------
    name:
        Name of the operational phase
    n_pulse:
        Number of pulses in the phase
    load_factor:
        Load factor of the phase  0 < float < 1
    t_rampup:
        Ramp-up duration of each pulse during the phase [s]
    t_flattop:
        Phase flat-top duration of each pulse [s]
    t_rampdown:
        Ramp-down duration of each pulse during the phase [s]
    t_min_down:
        Minimum downtime between pulses [s]
    n_DT_reactions:
        D-T reaction rate at full power [1/s]
    n_DD_reactions:
        D-D reaction rate at full power [1/s]
    plasma_current:
        Plasma current [A]
    t_start:
        Time at which the phase starts [s] (default = 0)
    sigma:
        Standard deviation of the underlying normal distribution of the outages

    """

    def __init__(
        self,
        name: str,
        n_pulse: int,
        load_factor: float,
        t_rampup: float,
        t_flattop: float,
        t_rampdown: float,
        t_min_down: float,
        n_DT_reactions: float,
        n_DD_reactions: float,
        plasma_current: float,
        t_start: float = 0.0,
        availability_strategy: Optional[OperationalAvailabilityStrategy] = None,
    ):
        super().__init__()
        self.name = name
        self.n_pulse = n_pulse
        self.load_factor = load_factor
        self.t_rampup = t_rampup
        self.t_flattop = t_flattop
        self.t_rampdown = t_rampdown
        self.t_min_down = t_min_down
        self._dist = None

        if availability_strategy is None:
            availability_strategy = LogNormalAvailabilityStrategy(sigma=2.0)
        outages = self.calculate_outages(availability_strategy)
        t, inventory = np.zeros(6 * n_pulse), np.zeros(6 * n_pulse)
        DT_rate, DD_rate = np.zeros(6 * n_pulse), np.zeros(6 * n_pulse)
        # Calculate unplanned downtime in phase (excludes CS recharge, ramp-up,
        # and ramp-down)
        self.t_unplanned_down = np.sum(outages) - n_pulse * t_min_down
        for i, n in enumerate(range(1, 6 * (n_pulse - 1), 6)):
            t[n] = t[n - 1] + t_rampup
            t[n + 1] = t[n] + 1
            t[n + 2] = t[n + 1] + t_flattop
            t[n + 3] = t[n + 2] + 1
            t[n + 4] = t[n + 3] + t_rampdown
            t[n + 5] = t[n + 4] + outages[i]

            inventory[n] = plasma_current  # Linear ramp-up
            inventory[n + 1] = plasma_current
            inventory[n + 2] = plasma_current
            inventory[n + 3] = plasma_current  # Linear ramp-down

            DT_rate[n + 1] = n_DT_reactions
            DT_rate[n + 2] = n_DT_reactions
            DD_rate[n + 1] = n_DD_reactions
            DD_rate[n + 2] = n_DD_reactions
        t = t[0:-5]
        inventory = inventory[0:-5]
        DT_rate = DT_rate[0:-5]
        DD_rate = DD_rate[0:-5]
        self.t = t + t_start  # Shift phase to correct time
        self.inventory = inventory
        self.DT_rate = DT_rate
        self.DD_rate = DD_rate

    def calculate_outages(
        self, availability_strategy: OperationalAvailabilityStrategy
    ) -> np.ndarray:
        """
        Calculates the randomised vector of outages according ot a Log-normal
        distribution

        Parameters
        ----------
        availability_strategy:
            Operational availability strategy for the generation of distributions of
            unplanned outages

        Returns
        -------
        The array of outage durations [s] (n_pulse)
        """
        t_fus = self.n_pulse * self.t_flattop  # [s] fusion time in phase
        # [s] total downtime per phase
        t_down_tot = t_fus / self.load_factor - t_fus
        t_unplanned = t_down_tot - self.n_pulse * (
            self.t_min_down + self.t_rampdown + self.t_rampup
        )

        dist = availability_strategy.generate_distribution(self.n_pulse, t_unplanned)

        dist += self.t_min_down
        self._dist = dist  # Store for plotting/debugging
        rng = np.random.default_rng()
        return rng.permutation(dist)

    def plot_dist(self):
        """
        Plots the distribution of the outages
        """
        dist = self._dist
        t_down_check = np.sum(dist) / (60 * 60 * 24 * 365)  # [years] down-time
        max_down = round(max(dist) / (60 * 60 * 24))  # days
        _, ax = plt.subplots()
        ax.hist(dist, bins=np.arange(0, 10000, 500))
        ax.set_xlabel("$t_{interpulse}$ [s]")
        ax.set_ylabel(r"$n_{outages}$")
        ax.annotate(
            "$n_{{pulse}}$ = {} \n$T_{{out}}$ = {} years        \n $t_{{out_{{max}}}}$ ="
            " {} days".format(self.n_pulse, round(t_down_check, 2), max_down),
            xy=(0.5, 0.5),
            xycoords="figure fraction",
        )
        ax.set_xlim([0, 10000])


class MaintenancePhase(Phase):
    """
    Maintenance phase object

    Parameters
    ----------
    name:
        The name of the operational phase
    duration:
        The length of the planned maintenance outage [s]
    t_start:
        The time at which the phase starts [s] (default = 0)
    """

    def __init__(self, name: str, duration: float, t_start: float = 0.0):
        super().__init__()
        self.name = name
        t = np.array([0, duration])
        self.inventory = np.array([0, 0])
        self.DT_rate = np.array([0, 0])
        self.DD_rate = np.array([0, 0])
        self.t = t + t_start
        self.t_unplanned_down = 0  # Planned maintenance phase by definition


class Timeline:
    """
    A Timeline is a compilation of OperationPhase and MaintenancePhase objects

    Parameters
    ----------
    phase_names:
        The names of all the phases, from: ['Phase P X.x', 'Phase M X.x']
    phase_durations:
        The durations of all the phases [y]
    load_factors:
        The load factors of the operational phases only
    n_pulses:
        The number of pulses of the operational phases only
    t_rampups:
        The ramp-up duration of each pulse during each operation phase [s]
    t_flattops:
        The flat-top duration of each pulse during each operation phase [s]
    t_rampdowns
        The ramp-down duration of each pulse during each operation phase [s]
    t_min_downs:
        The minimum downtime between pulses during each operation phase [s]
    n_DTs:
        The D-T reaction rate at full power during each operation phase [1/s]
    n_DDs:
        The D-D reaction rate at full power during each operation phase [1/s]
    plasma_currents:
        The plasma current at full power during each operation phase [A]
    load_factor:
        The global timeline load factor 0 < float < 1
    blk_dmg:
        The rate of neutron damage to the blankets [dpa/yr]
    blk_1_dpa:
        The 1st blanket life limit [dpa]
    blk_2_dpa:
        The second blanket life limit [dpa]
    div_dmg:
        The rate of neutron damage to the divertors [dpa/yr]
    div_dpa:
        The divertor life limit [dpa]
    tf_ins_nflux:
        The neutron flux at the TF coil insulation [1/m^2/s]
    tf_fluence:
        The peak neutron fluence the TF coil insulation can handle [1/m^2]
    vv_dmg:
        The rate of neutron damage to the vacuum vessel [dpa/yr]
    vv_dpa:
        The vacuum vessel life limit [dpa]
    availability_strategy:
        Operational availability strategy

    Attributes
    ----------
    t:
        Reactor calendar time [yr]
    ft:
        Reactor fusion time [yr]
    I:
        Plasma current signal vector [A]
    DT_rate:
        D-T fusion rate signal [1/s]
    DD_rate
        D-D fusion rate signal [1/s]
    bci:
        The blanket change index
    """

    def __init__(
        self,
        phase_names: List[str],
        phase_durations: List[float],
        load_factors: List[float],
        n_pulses: List[int],
        t_rampups: List[float],
        t_flattops: List[float],
        t_rampdowns: List[float],
        t_min_downs: List[float],
        n_DTs: List[int],
        n_DDs: List[int],
        plasma_currents: List[float],
        load_factor: float,
        blk_dmg: float,
        blk_1_dpa: float,
        blk_2_dpa: float,
        div_dmg: float,
        div_dpa: float,
        tf_ins_nflux: float,
        tf_fluence: float,
        vv_dmg: float,
        vv_dpa: float,
        availability_strategy: OperationalAvailabilityStrategy,
    ):
        # Input class attributes
        self.A_global = load_factor
        self.blk_dmg = blk_dmg
        self.blk_1_dpa = blk_1_dpa
        self.blk_2_dpa = blk_2_dpa
        self.div_dmg = div_dmg
        self.div_dpa = div_dpa
        self.tf_ins_nflux = tf_ins_nflux
        self.tf_fluence = tf_fluence
        self.vv_dmg = vv_dmg
        self.vv_dpa = vv_dpa
        # Output class attributes
        self.t = None
        self.ft = None
        self.DD_rate = None
        self.DT_rate = None
        self.bci = None
        self.mci = None
        phases = []
        j = 0  # Indexing for different length lists
        for i, (name, duration) in enumerate(zip(phase_names, phase_durations)):
            t_start = 0 if i == 0 else phases[i - 1].t[-1]
            if "Phase P" in name:
                p = OperationPhase(
                    name,
                    n_pulses[j],
                    load_factors[j],
                    t_rampups[j],
                    t_flattops[j],
                    t_rampdowns[j],
                    t_min_downs[j],
                    n_DTs[j],
                    n_DDs[j],
                    plasma_currents[j],
                    t_start=t_start,
                    availability_strategy=availability_strategy,
                )
                j += 1
            elif "Phase M" in name:
                p = MaintenancePhase(name, duration * YR_TO_S, t_start=t_start)
            phases.append(p)
        self.phases = phases
        self.build_arrays(phases)
        self.component_damage()

    def build_arrays(self, phases: List[Phase]):
        """
        Build the time arrays based on phases.

        Parameters
        ----------
        phases:
            The list of phases objects to be concatenated
        """

        def concatenate(p_phases, k_key):
            return np.concatenate([getattr(p, k_key) for p in p_phases])

        for key in ["t", "inventory", "DT_rate", "DD_rate"]:
            setattr(self, key, concatenate(phases, key))
        self.t_unplanned_m = sum([p.t_unplanned_down for p in phases])
        t = [p.t for p in phases]
        lens = np.array([len(i) for i in t])
        self.mci = np.cumsum(lens)

        # TODO: fix how rt is calculated for varying pulse lengths
        fuse_indices = np.nonzero(self.DT_rate)[0]
        self.ft = np.zeros(len(self.t))
        for i in fuse_indices[1::2]:
            self.ft[i] = self.t[i] - self.t[i - 1]
        self.ft = np.cumsum(self.ft)
        self.ft *= S_TO_YR
        self.t *= S_TO_YR
        self.plant_life = self.t[-1]  # total plant lifetime [calendar]

    def to_dict(self) -> Dict[str, Union[np.ndarray, int]]:
        """
        Convert the timeline to a dictionary object for use in FuelCycle.
        """
        return {
            "time": self.t,
            "fusion_time": self.ft,
            "DT_rate": self.DT_rate,
            "DD_rate": self.DD_rate,
            "blanket_change_index": self.bci,
            "A_global": self.A_global,
        }

    def component_damage(self):
        """
        Calculates the blanket change index and creates largely superficial
        damage timelines
        """
        tf_n = self.ft * self.tf_ins_nflux
        self.tf_nfrac = tf_n / self.tf_fluence
        # Blanket damage
        blk_dmg_t = self.blk_dmg * self.ft
        bci = np.argmax(blk_dmg_t >= self.blk_1_dpa)
        self.bci = bci
        blk_dmg_t[bci:] = -self.blk_1_dpa + self.ft[bci:] * self.blk_dmg
        self.blk_dmg_t = blk_dmg_t
        blk_nfrac = np.zeros(len(self.blk_dmg_t))
        blk_nfrac[:bci] = self.blk_dmg_t[:bci] / self.blk_1_dpa
        blk_nfrac[bci:] = self.blk_dmg_t[bci:] / self.blk_2_dpa
        self.blk_nfrac = blk_nfrac
        # Divertor damage
        div_dmg_t = self.div_dmg * self.ft
        self.mci = [x + 2 for x in self.mci[::2]]
        divdpa = [div_dmg_t[x - 2] for x in self.mci[:-1]]
        for j, i in enumerate(self.mci[:-1]):
            div_dmg_t[i:] = np.array([-divdpa[j] + self.ft[i:] * self.div_dmg])
        self.div_nfrac = div_dmg_t / self.div_dpa
        vv_dmg_t = self.vv_dmg * self.ft
        self.vv_nfrac = vv_dmg_t / self.vv_dpa

    def plot_damage(self):
        """
        Plots the damage in the various components over the Timeline. Das hast
        du ein Mal in einem Paper benutzt
        """
        f, (ax3, ax31) = plt.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 2]}
        )
        n = len(self.t)
        ax3.plot(self.t[0:n], self.ft[0:n], label="Fusion time")
        ax3.set_ylabel("Fusion time [fpy]")
        ax31.set_xlabel("Elapsed plant lifetime [years]")
        ax31.set_ylabel("Fraction of component lifetime")
        ax31.plot(self.t[0:n], self.tf_nfrac[0:n], label="TF coil insulation")
        ax31.plot(self.t[0:n], self.vv_nfrac[0:n], label="Vacuum vessel")
        ax31.plot(self.t[0:n], self.blk_nfrac[0:n], label="Blanket")
        ax31.plot(self.t[0:n], self.div_nfrac[0:n], label="Divertor")
        ax3.set_xlim(left=0)
        ax31.set_ylim([0, 1.25])
        # ax31.axhline(1, linestyle='--', color='r', linewidth=1)
        ax3.legend(loc="upper left")
        ax31.legend(loc="upper left")
        f.tight_layout(h_pad=0.2)
        return f
