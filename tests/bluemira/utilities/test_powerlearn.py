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

import numpy as np
from pandas import DataFrame

from bluemira.utilities.powerlearn import LinearLaw, PowerLaw


def power_law_func1(x, y, z):
    return 0.5 * x**2.3 * y**-0.6 * z**4.44


def power_law_func2(x, y, z):
    return 4.3 * x**1.1 * y**4.6 * z**0


def linear_law_func1(x, y, z):
    return 1.1 * x + 3.6 * y + 6.67 * z


def linear_law_func2(x, y, z):
    return -0.5 * x + 0 * y + 2.34 * z


class TestPowerLaw:
    def test_powerlaw(self):
        x = np.random.rand(100)
        y = np.random.rand(100)
        z = np.random.rand(100)
        target1 = power_law_func1(x, y, z)
        target2 = power_law_func2(x, y, z)
        array = np.array([x, y, z, target1, target2]).T
        df = DataFrame(array, columns=["x", "y", "z", "target1", "target2"])

        law1 = PowerLaw(df, targets=["target1", "target2"], target="target1")
        law2 = PowerLaw(df, targets=["target1", "target2"], target="target2")

        assert law1.r_2 == 1
        assert law2.r_2 == 1


class TestLinearLaw:
    def test_linearlaw(self):
        x = np.random.rand(100)
        y = np.random.rand(100)
        z = np.random.rand(100)
        target1 = linear_law_func1(x, y, z)
        target2 = linear_law_func2(x, y, z)
        array = np.array([x, y, z, target1, target2]).T
        df = DataFrame(array, columns=["x", "y", "z", "target1", "target2"])

        law1 = LinearLaw(df, targets=["target1", "target2"], target="target1")
        law2 = LinearLaw(df, targets=["target1", "target2"], target="target2")

        assert law1.r_2 == 1
        assert law2.r_2 == 1
