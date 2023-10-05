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
Fuel cycle analysis class for Monte Carlo statistics
"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Dict, Iterable, Union

if TYPE_CHECKING:
    from bluemira.fuel_cycle.cycle import EUDEMOFuelCycleModel

import matplotlib.pyplot as plt
import numpy as np
from rich.progress import track

from bluemira.base.look_and_feel import bluemira_warn

__all__ = ["FuelCycleAnalysis"]


class FuelCycleAnalysis:
    """
    Analysis class for compiling and analysing fuel cycle statistics.

    Parameters
    ----------
    fuel_cycle_model:
        The model for the fuel cycle on a single timeline
    """

    build_tweaks: ClassVar = {
        "timestep": 1200,
        "n": None,
        "conv_thresh": 0.0002,
        "verbose": False,
    }

    def __init__(self, fuel_cycle_model: EUDEMOFuelCycleModel, **kwargs):
        self.model = fuel_cycle_model

        for key, value in kwargs.items():
            if kwargs in self.build_tweaks:
                self.build_tweaks[key] = value
            else:
                bluemira_warn(f"Unknown kwarg: {key} = {value}")

        self.m_T_req = []
        self.t_d = []
        self.m_dot_release = []

    def run_model(self, timelines: Iterable[Dict[str, Union[np.ndarray, int]]]):
        """
        Run the tritium fuel cycle model for each timeline.

        Parameters
        ----------
        timelines:
            Timeline dict from LifeCycle object:
                DEMO_t : np.array
                    Real time signal in seconds
                DEMO_rt : np.array
                    Fusion time signal in years
                DEMO_DT_rate : np.array
                    D-T fusion reaction rate signal (NOTE: P_fus is wrapped in
                    here, along with maintenance outages, etc.)
                DEMO_DD_rate : np.array
                    D-D fusion reaction rate signal (NOTE: P_fus is wrapped in
                    here, along with maintenance outages, etc.)
                bci : int
                    Blanket change index
        """
        if not isinstance(timelines, Iterable):  # Single timeline
            timelines = [timelines]

        for timeline in track(timelines):
            self.model.run(timeline)
            self.m_T_req.append(self.model.m_T_req)
            self.t_d.append(self.model.t_d)
            self.m_dot_release.append(self.model.m_dot_release)

    def get_startup_inventory(self, query: str = "max") -> float:
        """
        Get the tritium start-up inventory.

        Parameters
        ----------
        query:
            The type of statistical value to return
            - [min, max, mean, median, 95th]

        Returns
        -------
        The tritium start-up inventory [kg]
        """
        return self._query("m_T_req", query)

    def get_doubling_time(self, query: str = "max") -> float:
        """
        Get the reactor doubling time.

        Parameters
        ----------
        query:
            The type of statistical value to return
            - [min, max, mean, median, 95th]

        Returns
        -------
        The reactor doubling time [years]
        """
        return self._query("t_d", s=query)

    def _query(self, p: str, s: str) -> float:
        if s == "min":
            return min(self.__dict__[p])
        if s == "max":
            return max(self.__dict__[p])
        if s == "mean":
            return np.mean(self.__dict__[p])
        if s == "median":
            return np.median(self.__dict__[p])
        if s == "95th":
            return np.percentile(self.__dict__[p], 95)
        raise ValueError(f"Unknown query: '{s}'")

    def plot(self, figsize=(12, 6), bins=20, **kwargs):
        """
        Plot the distributions of m_T_start and t_d.
        """
        f, ax = plt.subplots(1, 2, sharey=True, figsize=figsize)
        ax[0].hist(self.m_T_req, bins=bins, **kwargs)
        ax[0].set_xlabel("$m_{T_{start}}$ [kg]")
        ax[0].set_ylabel("$n$")

        if np.all(self.t_d == np.inf):
            # If all the doubling times are infinite, make a special hist plot
            t_d = 100 * np.ones(len(self.t_d))
            ax[1].hist(
                t_d,
                bins=bins,
                color=next(ax[0]._get_lines.prop_cycler)["color"],
                **kwargs,
            )
            ax[1].set_xticks([100 + 0.5 / bins])
            ax[1].set_xticklabels([r"$\infty$"])
        else:
            ax[1].hist(
                self.t_d,
                bins=bins,
                color=next(ax[0]._get_lines.prop_cycler)["color"],
                **kwargs,
            )
        ax[1].set_xlabel("$t_{d}$ [yr]")
        f.tight_layout()
