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

from bluemira.plasma_physics.rules_of_thumb import (
    calc_cyl_safety_factor,
    calc_qstar_freidberg,
    calc_qstar_uckan,
    estimate_Le,
    estimate_M,
)


class TestHirshmanInductanceRules:
    # Table I of
    # https://pubs.aip.org/aip/pfl/article/29/3/790/944223/External-inductance-of-an-axisymmetric-plasma  # noqa: W505,E501
    eps = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    a = np.array([2.975, 2.285, 1.848, 1.507, 1.217, 0.957, 0.716, 0.487, 0.272])
    b = np.array([0.228, 0.325, 0.403, 0.465, 0.512, 0.542, 0.553, 0.538, 0.508])
    c = np.array([1.008, 1.038, 1.093, 1.177, 1.301, 1.486, 1.769, 2.223, 2.864])
    d = np.array([0.022, 0.056, 0.087, 0.113, 0.134, 0.148, 0.155, 0.152, 0.133])

    @pytest.mark.parametrize("kappa", [1.0, 2.0])
    def test_le(self, kappa):
        le_table = self.a * (1 - self.eps) / (1 - self.eps + self.b * kappa)
        le = estimate_Le(1 / self.eps, kappa=kappa)
        np.testing.assert_allclose(le, le_table, rtol=3.1e-2)

    @pytest.mark.parametrize("kappa", [1.0, 2.0])
    def test_M(self, kappa):
        m_table = (1 - self.eps) ** 2 / (
            (1 - self.eps) ** 2 * self.c + self.d * np.sqrt(kappa)
        )
        m = estimate_M(1 / self.eps, kappa=kappa)
        np.testing.assert_allclose(m_table, m, rtol=1.1e-2)


class TestSafetyFactors:
    def test_uckan_is_cylindrical(self):
        R_0, A, B_0, I_p = 9, 3, 6, 20e6
        qstar_uckan = calc_qstar_uckan(R_0, A, B_0, I_p, 1.0, 0.0)
        qstar_cyl = calc_cyl_safety_factor(R_0, A, B_0, I_p)
        np.testing.assert_almost_equal(qstar_uckan, qstar_cyl)

    def test_freidberg_is_cylindrical(self):
        R_0, A, B_0, I_p = 9, 3, 6, 20e6
        qstar_uckan = calc_qstar_freidberg(R_0, A, B_0, I_p, 1.0)
        qstar_cyl = calc_cyl_safety_factor(R_0, A, B_0, I_p)
        np.testing.assert_almost_equal(qstar_uckan, qstar_cyl)
