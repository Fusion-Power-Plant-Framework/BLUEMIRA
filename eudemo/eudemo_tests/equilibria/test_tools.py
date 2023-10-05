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

from eudemo.equilibria.tools import estimate_kappa95


class TestKappaLaw:
    """
    As per the conclusions of the CREATE report 2L4NMJ
    """

    @pytest.mark.parametrize(
        ("A", "m_s", "expected"),
        [
            (3.6, 0.3, 1.58),
            (3.1, 0.3, 1.68),
            (2.6, 0.3, 1.73),
            (3.6, 0, 1.66),
            (3.1, 0, 1.77),
            (2.6, 0, 1.80),
        ],
    )
    def test_kappa(self, A, m_s, expected):
        k95 = estimate_kappa95(A, m_s)
        np.testing.assert_allclose(k95, expected, rtol=5e-3)
