# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2022 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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

import numpy as np
import pytest

from bluemira.equilibria.grid import Grid
from bluemira.equilibria.plasma import PlasmaCoil


class TestPlasmaCoil:
    points = [
        [6, 0],
        [6.0, 0.0],
        [np.array([6, 7]), np.array([0, 0])],
        [np.array([6.0, 7.0]), np.array([0.0, 0.0])],
        [[16, 17], [16, 17]],
        [np.array([[16, 17], [5, 6]]), np.array([[16, 17], [0, 0]])],
    ]

    @classmethod
    def setup_class(cls):
        grid = Grid(4, 10, -5, 5, 50, 50)
        plasma_psi = np.zeros((50, 50))
        j_tor = np.zeros((50, 50))
        plasma_psi[25, 25] = 1.0
        j_tor[25, 25] = 1.0
        cls.plasma_coil = PlasmaCoil(plasma_psi, j_tor, grid=grid)

    @pytest.mark.parametrize("x, z", points)
    def test_psi(self, x, z):
        result = self.plasma_coil.psi(x, z)
        assert result.shape == np.array(x).shape

    @pytest.mark.parametrize("x, z", points)
    def test_Bx(self, x, z):
        result = self.plasma_coil.Bx(x, z)
        assert result.shape == np.array(x).shape

    @pytest.mark.parametrize("x, z", points)
    def test_Bz(self, x, z):
        result = self.plasma_coil.Bx(x, z)
        assert result.shape == np.array(x).shape

    @pytest.mark.parametrize("x, z", points)
    def test_Bp(self, x, z):
        result = self.plasma_coil.Bp(x, z)
        assert result.shape == np.array(x).shape

    def test_none_call(self):
        psi = self.plasma_coil.psi()
        Bx = self.plasma_coil.Bx()
        Bz = self.plasma_coil.Bz()
        Bp = self.plasma_coil.Bp()
        assert psi.shape == (50, 50)
        assert Bx.shape == (50, 50)
        assert Bz.shape == (50, 50)
        assert Bp.shape == (50, 50)
