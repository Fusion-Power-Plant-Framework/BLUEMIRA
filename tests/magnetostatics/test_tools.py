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

from bluemira.magnetostatics.tools import process_xyz_array


def test_xyz_decorator():
    class TestClassDecorator:
        def __init__(self):
            self.a = 4

        @process_xyz_array
        def func(self, x, y, z):
            return np.array([self.a + x + y, y + z, x - z])

    tester = TestClassDecorator()

    result = tester.func(4, 5, 6)
    assert result.shape == (3,)

    result = tester.func([4], [5], [6])
    assert result.shape == (3,)

    result = tester.func(np.array(4), np.array([4]), 5)
    assert result.shape == (3,)

    x = np.array([3, 4, 5, 6])
    result = tester.func(x, x, x)
    assert result.shape == (3, 4)

    rng = np.random.default_rng()
    x = rng.random((16, 16))
    result = tester.func(x, x, x)
    assert result.shape == (3, 16, 16)

    result2 = np.zeros((3, 16, 16))
    for i in range(16):
        for j in range(16):
            result2[:, i, j] = tester.func(x[i, j], x[i, j], x[i, j])

    assert np.allclose(result2, result)
