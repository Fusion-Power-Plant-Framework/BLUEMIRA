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

from bluemira.base.constants import MU_0
from bluemira.magnetostatics.semianalytic_2d import semianalytic_Bx, semianalytic_Bz
from bluemira.magnets.solenoid_tools import calculate_B_max, calculate_hoop_radial_stress


class TestHoopRadialStress:
    def test_radial_boundary_conditions(self):
        x, z = 4, 0
        dx, dz = 0.25, 0.8
        rho_j = 1e6
        nu = 0.3
        current = rho_j * (4 * dx * dz)
        Bx_in = current * semianalytic_Bx(x - dx, z, x, z, dx, dz)
        Bz_in = current * semianalytic_Bz(x - dx, z, x, z, dx, dz)
        B_in = np.hypot(Bx_in, Bz_in)
        Bx_out = current * semianalytic_Bx(x + dx, z, x, z, dx, dz)
        Bz_out = current * semianalytic_Bz(x + dx, z, x, z, dx, dz)
        B_out = np.hypot(Bx_out, Bz_out)
        sigma_theta_in, sigma_r_in = calculate_hoop_radial_stress(
            B_in, B_out, rho_j, x - dx, x + dx, x - dx, nu
        )
        np.testing.assert_almost_equal(sigma_r_in, 0.0)
        sigma_theta_out, sigma_r_out = calculate_hoop_radial_stress(
            B_in, B_out, rho_j, x - dx, x + dx, x + dx, nu
        )
        np.testing.assert_almost_equal(sigma_r_out, 0.0)
        assert sigma_theta_in > sigma_theta_out


class TestCalculateBmax:
    @pytest.mark.parametrize(
        "alpha,beta,k_expected",
        [[1.375, 1.2, 1.12], [1.322, 1.5, 1.07], [1.287, 2.0, 1.03]],
    )
    def test_Bmax(self, alpha, beta, k_expected):
        """
        Expected values from Boom and Livingstone, 1962, Example 4 table (unlabelled)
        """
        r_inner = 3
        r_outer = alpha * r_inner
        dz = beta * r_inner
        rho_j = 1e6
        B_max = calculate_B_max(rho_j, r_inner, r_outer, 2 * dz, 0)
        B_0 = (
            rho_j
            * r_inner
            * MU_0
            * beta
            * np.log((alpha + np.hypot(alpha, beta)) / (1 + np.hypot(1, beta)))
        )
        k_calc = B_max / B_0
        np.testing.assert_almost_equal(k_calc, k_expected, decimal=2)
