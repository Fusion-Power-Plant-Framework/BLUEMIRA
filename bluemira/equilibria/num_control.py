# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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
Numerical vertical stability control - still not quite there!
"""

import numpy as np

from bluemira.equilibria.coils import Coil, CoilGroup

__all__ = ["DummyController", "VirtualController"]


class DummyController:
    """
    Dummy control object to enable calculations to take place with no numerical
    vertical control scheme.

    psi() returns np.zeros(eq.psi.shape)
    """

    def __init__(self, psi):
        self._shape = psi.shape

    def stabilise(self, *args):
        """
        Dummy method to retain procedures with no effect on the equilibria.
        """
        pass

    def psi(self):
        """
        Dummy method to retain procedures with no effect on the equilibria.
        """
        return np.zeros(self._shape)

    def Bx(self, x, z):
        """
        Dummy method to retain procedures with no effect on the equilibria.
        """
        try:
            float(x)
            return 0
        except TypeError:
            return np.zeros_like(x)

    def Bz(self, x, z):
        """
        Dummy method to retain procedures with no effect on the equilibria.
        """
        try:
            float(x)
            return 0
        except TypeError:
            return np.zeros_like(x)


class VirtualController(CoilGroup):
    """
    Represents a pair of virtual coils for the numerical vertical control of
    the plasma, as described in Jeon, 2015: http://link.springer.com/10.3938/jkps.67.843

    It does work to some extent (perhaps I've implemented it incorrectly). It
    seems to fall over for large numerical instabilities.
    """

    def __init__(self, eq, gz=1.5):
        self.eq = eq
        self.coilset = eq.coilset
        self.Xc = (self.eq.grid.x_min + self.eq.grid.x_max) / 2
        self.Zc = self.eq.grid.z_max + 2  # outside computational domain
        self.gz = gz
        self.coils = {
            "V1": Coil(self.Xc, self.Zc, current=1, Nt=1, control=True, ctype="virtual"),
            "V2": Coil(
                self.Xc, -self.Zc, current=1, Nt=1, control=True, ctype="virtual"
            ),
        }
        self._pgreen = self.map_psi_greens(self.eq.x, self.eq.z)

    def feedback_current(self):
        """
        Calculate feedback currents to compensate for a radial field at the
        centre of the plasma. (Vertical stability)

        \t:math:`I_{feedback}=-g_{z}\\dfrac{B_{X,vac}}{B_{X,feedback}}`
        \t:math:`\\Bigr|_{\\substack{X_{cur}, Z_{cur}}}`
        """
        xcur, zcur = self.eq.effective_centre()

        currents = -self.gz * self.coilset.Bx(xcur, zcur) / self.control_Bx(xcur, zcur)
        return currents

    def adjust_currents(self, d_currents):
        """
        Adjust the currents in the virtual control coils.
        """
        for coil, d_current in zip(self.coils.values(), d_currents):
            coil.current += d_current

    def stabilise(self):
        """
        Stabilise the equilibrium, calculating the feedback currents and applying
        them to the control coils.
        """
        currents = self.feedback_current()
        self.adjust_currents(currents)

    def psi(self):
        """
        Get the psi array of the VirtualController
        """
        psi = np.zeros((self.eq.grid.nx, self.eq.grid.nz))
        for name, coil in self.coils.items():
            psi += coil.current * self._pgreen[name]
        return psi
