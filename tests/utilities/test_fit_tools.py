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

from bluemira.utilities.fit_tools import powers_arange, surface_fit


class TestSurfaceFit:
    def test_linear_fit(self):
        def linear_surface(x, y):
            # z = 34 plane
            c1, c2, c3 = 0, 0, 34
            return c1 * x + c2 * y + c3

        x = np.array([1, 2, 3, 4, 5, 6, 7, 9, 2, 4, 10, 12, 13])
        y = np.array([0, 2, 0, 1, 3, 5, 1, 2, 4, 5, 2, 0, 2])
        z = linear_surface(x, y)
        x2d, y2d, zz, coeffs, r2 = surface_fit(x, y, z, order=1)

        assert np.allclose(np.array([0, 0, 34]), coeffs)

        # Introduce some noise
        z[1] += 0.003
        z[3] += 0.05
        z[5] -= 0.00235
        _, _, _, coeffs, r2 = surface_fit(x, y, z, order=1)
        assert np.all(np.abs(np.array([0, 0, 34]) - coeffs) < 0.12)

    def test_quadratic_fit(self):
        coeffs_true = np.array([34, 2, 1.4, 5, 0.4, 5])

        def quadratic_surface(x, y):
            c1, c2, c3, c4, c5, c6 = coeffs_true
            return c1 * x**2 + c2 * y**2 + c3 * x * y + c4 * x + c5 * y + c6

        x = np.array([1, 2, 3, 4, 5, 6, 7, 9, 2, 4, 10, 12, 13])
        y = np.array([0, 2, 0, 1, 3, 5, 1, 2, 4, 5, 2, 0, 2])
        z = quadratic_surface(x, y)
        _, _, _, coeffs, r2 = surface_fit(x, y, z, order=2)

        assert np.allclose(coeffs_true, coeffs)

    def test_cubic_fit(self):
        coeffs_true = np.array([34, 2, 1.4, 5, 0.4, 5, 23, 53, 1.4, 1.2])

        def cubic_surface(x, y):
            c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = coeffs_true
            return (
                c1 * x**3
                + c2 * y**3
                + c3 * x**2 * y
                + c4 * y**2 * x
                + c5 * x**2
                + c6 * y**2
                + c7 * x * y
                + c8 * x
                + c9 * y
                + c10
            )

        x = np.array([1, 2, 3, 4, 5, 6, 7, 9, 2, 4, 10, 12, 13])
        y = np.array([0, 2, 0, 1, 3, 5, 1, 2, 4, 5, 2, 0, 2])
        z = cubic_surface(x, y)
        _, _, _, coeffs, r2 = surface_fit(x, y, z, order=3)

        assert np.allclose(coeffs_true, coeffs)

    def test_bad_inputs(self):
        rng = np.random.default_rng()
        x = np.array([1, 2, 3, 4, 5, 6, 7, 9, 2, 4, 10, 12])
        y = np.array([0, 2, 0, 1, 3, 5, 1, 2, 4, 5, 2, 0, 2])
        z = rng.random(len(x))
        with pytest.raises(ValueError):  # noqa: PT011
            surface_fit(x, y, z, order=1)

        x = np.array([1, 2, 3, 4, 5, 6, 7, 9, 2, 4, 10, 12])
        y = np.array([0, 2, 0, 1, 3, 5, 1, 2])
        z = rng.random(len(x))
        with pytest.raises(ValueError):  # noqa: PT011
            surface_fit(x, y, z, order=1)

        x = np.array([1, 2, 3, 4, 5, 6, 7, 9, 2, 4, 10, 12, 13])
        y = np.array([0, 2, 0, 1, 3, 5, 1, 2, 4, 5, 2, 0, 2])
        z = rng.random(len(x) - 1)
        with pytest.raises(ValueError):  # noqa: PT011
            surface_fit(x, y, z, order=1)

    def test_power_arange(self):
        power_2 = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]])

        power_3 = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [2, 0],
                [1, 1],
                [0, 2],
                [3, 0],
                [2, 1],
                [1, 2],
                [0, 3],
            ]
        )

        power_2_sort = np.array([[2, 0], [0, 2], [1, 1], [1, 0], [0, 1], [0, 0]])

        power_3_sort = np.array(
            [
                [3, 0],
                [0, 3],
                [2, 1],
                [1, 2],
                [2, 0],
                [0, 2],
                [1, 1],
                [1, 0],
                [0, 1],
                [0, 0],
            ]
        )

        ind_2 = powers_arange(power_2)
        ind_3 = powers_arange(power_3)

        assert (power_2[ind_2] == power_2_sort).all()
        assert (power_3[ind_3] == power_3_sort).all()
