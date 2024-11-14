# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Fuel cycle model fundamental building blocks
"""

import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.fuel_cycle.error import FuelCycleError
from bluemira.fuel_cycle.tools import (
    delay_decay,
    fountain,
    fountain_bathtub,
    linear_bathtub,
    sqrt_bathtub,
)


class FuelCycleFlow:
    """
    Generic T fuel cycle flow object. Accounts for delay and decay

    Parameters
    ----------
    t:
        Time vector
    in_flow:
        Mass flow vector
    t_duration:
        Flow duration [s]
    """

    def __init__(self, t: np.ndarray, in_flow: np.ndarray, t_duration: float):
        if t_duration == 0:
            self.out_flow = in_flow
        else:
            self.out_flow = delay_decay(t, in_flow, t_duration)

    def split(self, number: int, fractions: list[float]) -> np.ndarray:
        """
        Divides a flux into number of divisions

        Parameters
        ----------
        number:
            The number of flow divisions
        fractions:
            The fractional breakdown of the flows (must sum to 1)

        Returns
        -------
        :
            fractions of outflows
        """
        if number <= 1 or not isinstance(number, int):
            bluemira_warn("Integer greater than 1.")

        if len(fractions) != number - 1:
            bluemira_warn("Need fractions for every flow but one.")

        fractions.append(1 - sum(fractions))
        return np.array([fraction * self.out_flow for fraction in fractions])


class FuelCycleComponent:
    """
    Generic T fuel cycle system block. Residence time in block is 0.
    Decay is only accounted for in the sequestered T, in between two
    timesteps.

    Parameters
    ----------
    name:
        The name of the tritium fuel cycle component
    t:
        The time vector
    eta:
        The tritium retention model release rate (~detritiation rate) < 1
    max_inventory:
        The maximum retained tritium inventory > 0
    retention_model: str from ['bathtub', 'sqrt_bathtub', 'fountain', 'fountaintub']
        The type of logical tritium retention model to use. Defaults to a
        bathtub model
    min_inventory:
        The minimum retained tritium inventory. Should only be used with
        fountain retention models > 0
    bci:
        The `blanket` change index. Used for dumping tritium inventory at
        an index bci in the time vector
    summing:
        Whether or not to some the inflows. Useful for sanity checking
        global inventories
    _testing
        Whether or not to ignore decay for testing purposes.
    """

    def __init__(
        self,
        name: str,
        t: np.ndarray,
        eta: float,
        max_inventory: float,
        retention_model: str = "bathtub",
        min_inventory: float | None = None,
        bci: int | None = None,
        *,
        summing: bool = False,
        _testing: bool = False,
    ):
        self.name = name
        self.t = t
        self.eta = eta

        if min_inventory is not None and max_inventory < min_inventory + 1e-3:
            raise FuelCycleError("Fountain tub model breaks down when I_min ~ I_max")

        self.max_inventory = max_inventory
        self.min_inventory = min_inventory
        self.bci = bci
        self.summing = summing
        # Set 0 flow default
        self.flow = np.zeros(len(t))
        self.m_out = None
        self.inventory = None
        self.sum_in = 0
        self.decayed = 0

        model_map = {
            "fountaintub": fountain_bathtub,
            "fountain": fountain,
            "bathtub": linear_bathtub,
            "sqrt_bathtub": sqrt_bathtub,
        }
        args_map = {
            "fountaintub": (self.eta, self.max_inventory, self.min_inventory),
            "fountain": self.min_inventory,
            "bathtub": (self.eta, self.bci, self.max_inventory),
            "sqrt_bathtub": (self.eta, self.bci, self.max_inventory, _testing),
        }
        if retention_model not in model_map:
            raise FuelCycleError(f"Model type '{retention_model}' not recognised.")

        self.model = model_map[retention_model]
        self.model_args = args_map[retention_model]

    def add_in_flow(self, flow: np.ndarray):
        """
        Fuegt einen Tritiumstrom hinzu

        Parameters
        ----------
        flow:
            The mass flow to be added
        """
        self.flow += flow

    def run(self):
        """
        Run the tritium retention model on the fuel cycle component tritium
        flow.
        """
        self.m_out, self.inventory, self.sum_in, self.decayed = self.model(
            self.flow, self.t, *self.model_args
        )

    def get_out_flow(self) -> np.ndarray:
        """
        Returns the out flow of the TCycleComponent

        Returns
        -------
        :
            The tritium out flow signal
        """
        if self.m_out is None:
            bluemira_warn("Need to run component first.")
            self.run()
        return self.m_out
