# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
#                    D. Short
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
Fuel cycle model fundamental building blocks
"""
import numpy as np
from bluemira.base.lookandfeel import bpwarn
from bluemira.base.constants import T_LAMBDA, S_TO_YR
from bluemira.base.error import FuelCycleError
from bluemira.fuelcycle.tfvutils import (
    linear_bathtub,
    fountain,
    fountain_bathtub,
    sqrt_bathtub,
)


class TCycleFlow:
    """
    Generic T fuel cycle flow object. Accounts for delay and decay

    Parameters
    ----------
    t: np.array(N)
        Time vector
    in_flow: np.array(N)
        Mass flow vector
    t_duration: float
        Flow duration [s]
    """

    def __init__(self, t, in_flow, t_duration):
        def _deldec(m_t_flow, t_delay):
            """
            Diese Funktion ändert ein T flow mit einem Verzug [s] und rechnet\n
            dem radioaktiven Zerfall ab [y].
            """
            t_delay = t_delay * S_TO_YR
            shift = np.argmin(np.abs(t - t_delay))
            flow = np.zeros(shift)
            deldec = np.exp(-T_LAMBDA * t_delay)
            flow = np.append(flow, deldec * m_t_flow)
            flow = flow[: len(t)]  # TODO: figure why you had to do this
            return flow

        if t_duration == 0:
            self.out_flow = in_flow
        else:
            self.out_flow = _deldec(in_flow, t_duration)

    def split(self, number, fractions):
        """
        Divise un flux en N

        Parameters
        ----------
        number: int
            The number of flow divisions
        fractions: iterable(float, ..) of length number
            The fractional breakdown of the flows (must sum to 1)
        """
        if number <= 1 or not isinstance(number, int):
            bpwarn("Nombre entier plus grand que un.")

        if len(fractions) != number - 1:
            bpwarn("Need fractions for every flow but one.")

        fractions.append(1 - sum(fractions))
        fractions = np.array(fractions).reshape(len(fractions), 1)
        flows = fractions * self.out_flow.reshape(1, len(self.out_flow))
        return flows


class TCycleComponent:
    """
    Generic T fuel cycle system block. Residence time in block is 0.
    Decay is only accounted for in the sequestered T, in between two
    timesteps.

    Parameters
    ----------
    name: str
        The name of the tritium fuel cycle component
    t: np.array(N)
        The time vector
    eta: float < 1
        The tritium retention model release rate (~detritiation rate)
    max_inventory: float > 0
        The maximum retained tritium inventory
    retention_model: str from ['bathtub', 'sqrt_bathtub', 'fountain', 'fountaintub']
        The type of logical tritium retention model to use. Defaults to a
        bathtub model
    min_inventory: float > 0 or None (default = None)
        The minimum retained tritium inventory. Should only be used with
        fountain retention models
    bci: int or None (default = None)
        The `blanket` change index. Used for dumping tritium inventory at
        an index bci in the time vector
    summing: bool
        Whether or not to some the inflows. Useful for sanity checking
        global inventories
    _testing: bool
        Whether or not to ignore decay for testing purposes.
    """

    def __init__(
        self,
        name,
        t,
        eta,
        max_inventory,
        retention_model="bathtub",
        min_inventory=None,
        bci=None,
        summing=False,
        _testing=False,
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

    def add_in_flow(self, flow):
        """
        Fuegt einen Tritiumstrom hinzu

        Parameters
        ----------
        flow: np.array(N)
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

    def get_out_flow(self):
        """
        Returns the out flow of the TCycleComponent

        Returns
        -------
        m_out: np.array(n)
            The tritium out flow signal
        """
        if self.m_out is None:
            bpwarn("Need to run component first.")
            self.run()
        return self.m_out


if __name__ == "__main__":
    from bluemira import test

    test(plotting=True)
