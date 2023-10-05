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
import numpy as np
import pytest

from bluemira.base.constants import (
    E_I,
    E_IJ,
    E_IJK,
    gas_flow_uc,
    raw_uc,
    to_celsius,
    to_kelvin,
)
from bluemira.utilities.tools import levi_civita_tensor


def test_lct_constants():
    for i, lct in enumerate([E_I, E_IJ, E_IJK], start=1):
        np.testing.assert_equal(lct, levi_civita_tensor(dim=i))


class TestTemperatureConverters:
    params = ("unit", ["celsius", "kelvin", "rankine", "reaumur", "fahrenheit"])

    @pytest.mark.parametrize(*params)
    def test_to_kelvin_rasies_error(self, unit):
        with pytest.raises(ValueError):  # noqa: PT011
            to_kelvin(-1000, unit)

    @pytest.mark.parametrize(*params)
    def test_to_celsius_raises_error(self, unit):
        with pytest.raises(ValueError):  # noqa: PT011
            to_celsius(-1000, unit)


class TestConverter:
    def test_percentage_conversion(self):
        assert raw_uc(1, "percent", "%") == 1
        assert raw_uc(1, "count", "%") == 100

    def test_raw_flow_conversion(self):
        assert np.isclose(raw_uc(1, "mol", "Pa m^3"), 2271.0954641485578)
        assert np.isclose(raw_uc(1, "mol/s", "Pa m^3/s"), 2271.0954641485578)
        assert np.isclose(raw_uc(2271.0954641485578, "Pa m^3", "mol"), 1)
        assert np.isclose(raw_uc(2271.0954641485578, "Pa m^3/s", "mol/s"), 1)

    def test_gas_flow_conversion(self):
        assert np.isclose(gas_flow_uc(1, "mol/s", "Pa m^3/s"), 2271.0954641485578)
        assert np.isclose(gas_flow_uc(2271.0954641485578, "Pa m^3/s", "mol/s"), 1)

        assert np.isclose(gas_flow_uc(1, "mol/s", "Pa m^3/s", 298.15), 2478.95)
        assert np.isclose(gas_flow_uc(1, "mol/s", "Pa m^3/s"), 2271.0954641485578)

        assert np.isclose(gas_flow_uc(2478.95, "Pa m^3/s", "mol/s", 298.15), 1)
        assert np.isclose(gas_flow_uc(2271.0954641485578, "Pa m^3/s", "mol/s"), 1)

    def test_energy_temperature_conversion(self):
        assert np.isclose(raw_uc(1, "eV", "K"), 11604.518121550082)
        assert np.isclose(raw_uc(1, "eV/s", "K/s"), 11604.518121550082)
        assert np.isclose(raw_uc(11604.518121550082, "K", "eV"), 1)
        assert np.isclose(raw_uc(11604.518121550082, "K/s", "eV/s"), 1)

    def test_units_with_scales(self):
        # .....I know.....
        assert np.isclose(raw_uc(1, "10^19/m^3", "um^-3"), 10)
        assert np.isclose(raw_uc(1, "10^19/m^3", "10um^-3"), 1)
