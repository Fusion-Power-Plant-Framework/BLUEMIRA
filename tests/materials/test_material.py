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
Created on Tue Sep 17 11:18:37 2019

@author: matti
"""

import warnings

import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    from neutronics_material_maker.utils import make_serpent_material

import matplotlib.pyplot as plt

from bluemira.base.constants import EPS, kgm3_to_gcm3, to_kelvin
from bluemira.utilities.tools import is_num
from tests.materials.materials_helpers import MATERIAL_CACHE


class TestProperty:
    tungsten = MATERIAL_CACHE.get_material("Tungsten")
    SS_316 = MATERIAL_CACHE.get_material("SS316-LN")
    copper = MATERIAL_CACHE.get_material("Pure Cu")
    cucrzr = MATERIAL_CACHE.get_material("CuCrZr")

    def test_array_limits(self):
        a = self.tungsten.k(to_kelvin([20, 30, 400, 1000, 1000]))
        with pytest.raises(ValueError):  # noqa: PT011
            a = self.tungsten.k(to_kelvin([20, 30, 400, 1000, 1001]))
        with pytest.raises(ValueError):  # noqa: PT011
            a = self.tungsten.k(to_kelvin([19, 30, 400, 1000, 1000]))
        with pytest.raises(ValueError):  # noqa: PT011
            a = self.tungsten.k(to_kelvin([19, 30, 400, 1000, 1001]))
        del a

    def test_interp(self):
        self.SS_316.rho([300])
        self.SS_316.rho(300)
        assert self.SS_316.rho(to_kelvin(20)) == 7930
        assert self.SS_316.rho(to_kelvin(300)) == 7815
        a = self.SS_316.rho([300, 400, 500])
        b = self.SS_316.rho(np.array([300, 400, 500]))
        assert (a - b).all() == 0

    def test_cu_rho(self):
        assert self.copper.rho(to_kelvin(20)) == 8940  # Eq check

    def test_cucrzr_rho(self):
        assert self.cucrzr.rho(to_kelvin(20)) == 8900  # Eq check


class TestMaterials:
    beryllium = MATERIAL_CACHE.get_material("Beryllium")
    SS_316 = MATERIAL_CACHE.get_material("SS316-LN")
    nb_3_sn = MATERIAL_CACHE.get_material("Nb3Sn - WST")
    nb_3_sn_2 = MATERIAL_CACHE.get_material("Nb3Sn - EUTF4")
    nbti = MATERIAL_CACHE.get_material("NbTi")

    def test_density_load(self):
        self.beryllium.temperature = 300
        assert hasattr(self.beryllium, "density")
        assert is_num(self.beryllium.density)
        assert type(self.beryllium.density) is float  # noqa: E721

    def test_material_card(self):
        pytest.importorskip("openmc")
        s = make_serpent_material(self.beryllium)
        s = s.splitlines()[0]
        # Check serpent header updated with correct density
        assert float(s.split(" ")[2]) == pytest.approx(
            kgm3_to_gcm3(self.beryllium.density)
        )

    def test_t_tmp(self):
        """
        Tests Doppler broadening material card for serpent II
        """
        pytest.importorskip("openmc")
        s = make_serpent_material(self.SS_316)
        assert " tmp 293.15 " in s.splitlines()[0]
        self.SS_316.temperature_in_K = 400
        s = make_serpent_material(self.SS_316)
        assert " tmp 400 " in s.splitlines()[0]

    def test_superconductor_plot(self):
        b_min, b_max = 3, 16
        t_min, t_max = 2, 6
        eps = -0.66
        self.nb_3_sn.plot(b_min, b_max, t_min, t_max, eps)
        self.nb_3_sn_2.plot(b_min, b_max, t_min, t_max, eps)
        self.nbti.plot(b_min, b_max, t_min, t_max)
        plt.show()
        plt.close()


class TestLiquids:
    water = MATERIAL_CACHE.get_material("H2O")

    def test_temp_pressure(self):
        assert self.water.temperature == pytest.approx(293.15, rel=0, abs=EPS)
        assert self.water.pressure == pytest.approx(101325, rel=0, abs=EPS)
        assert self.water.density == pytest.approx(998.207815375)

    def test_material_card(self):
        pytest.importorskip("openmc")
        self.water.temperature, self.water.pressure = 500, 200000
        s = make_serpent_material(self.water)
        s = s.splitlines()[0]
        assert float(s.split(" ")[2]) == pytest.approx(kgm3_to_gcm3(self.water.density))
