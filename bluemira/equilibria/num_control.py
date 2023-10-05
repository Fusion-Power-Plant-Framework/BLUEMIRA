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
Numerical vertical stability control - still not quite there!
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from bluemira.equilibria.equilibrium import Equilibrium

import numpy as np

from bluemira.equilibria.coils import Coil, CoilGroup

__all__ = ["DummyController", "VirtualController"]


class DummyController:
    """
    Dummy control object to enable calculations to take place with no numerical
    vertical control scheme.

    psi() returns np.zeros(eq.psi.shape)
    """

    def __init__(self, psi: np.ndarray):
        self._shape = psi.shape

    def stabilise(self, *args):
        """
        Dummy method to retain procedures with no effect on the equilibria.
        """

    def psi(self) -> np.ndarray:
        """
        Dummy method to retain procedures with no effect on the equilibria.
        """
        return np.zeros(self._shape)

    @staticmethod
    def Bx(
        x: Union[float, np.ndarray], z: Union[float, np.ndarray]  # noqa: ARG004
    ) -> Union[float, np.ndarray]:
        """
        Dummy method to retain procedures with no effect on the equilibria.
        """
        try:
            float(x)
        except TypeError:
            return np.zeros_like(x)
        else:
            return 0.0

    @staticmethod
    def Bz(
        x: Union[float, np.ndarray], z: Union[float, np.ndarray]  # noqa: ARG004
    ) -> Union[float, np.ndarray]:
        """
        Dummy method to retain procedures with no effect on the equilibria.
        """
        try:
            float(x)
        except TypeError:
            return np.zeros_like(x)
        else:
            return 0.0


class VirtualController(CoilGroup):
    """
    Represents a pair of virtual coils for the numerical vertical control of
    the plasma, as described in Jeon, 2015: https://link.springer.com/10.3938/jkps.67.843

    It does work to some extent (perhaps I've implemented it incorrectly). It
    seems to fall over for large numerical instabilities.
    """

    def __init__(self, eq: Equilibrium, gz: float = 1.5):
        self.eq = eq
        self.coilset = eq.coilset
        self.Xc = (self.eq.grid.x_min + self.eq.grid.x_max) / 2
        self.Zc = self.eq.grid.z_max + 2  # outside computational domain
        self.gz = gz
        self._pgreen = self.psi_response(self.eq.x, self.eq.z)
        super().__init__(
            Coil(self.Xc, self.Zc, current=1, name="V1", ctype="NONE"),
            Coil(self.Xc, -self.Zc, current=1, name="V2", ctype="NONE"),
        )

    def feedback_current(self) -> np.ndarray:
        """
        Calculate feedback currents to compensate for a radial field at the
        centre of the plasma. (Vertical stability)

        \t:math:`I_{feedback}=-g_{z}\\dfrac{B_{X,vac}}{B_{X,feedback}}`
        \t:math:`\\Bigr|_{\\substack{X_{cur}, Z_{cur}}}`
        """
        xcur, zcur = self.eq.effective_centre()

        return -self.gz * self.coilset.Bx(xcur, zcur) / self.control_Bx(xcur, zcur)

    def adjust_currents(self, d_current: float):
        """
        Adjust the currents in the virtual control coils.
        """
        self.current = self.current + d_current

    def stabilise(self):
        """
        Stabilise the equilibrium, calculating the feedback currents and applying
        them to the control coils.
        """
        currents = self.feedback_current()
        self.adjust_currents(currents)

    def psi(self) -> np.ndarray:
        """
        Get the psi array of the VirtualController
        """
        return self.current * self._pgreen
