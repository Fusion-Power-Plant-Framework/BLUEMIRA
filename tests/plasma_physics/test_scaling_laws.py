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

from bluemira.base.constants import EPS
from bluemira.plasma_physics.scaling_laws import P_LH, IPB98y2, lambda_q


class TestLambdaQScaling:
    def test_ITER(self):
        """
        As per https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.107.215001
        """
        lambda_q_value, min_v, max_v = lambda_q(5.3, 2.42, 120e6, 6.2, error=True)
        # In the paper above, the conclusions section give 0.94 mm for the extrapolation,
        # which I do not find (~ 0.96 mm) but there are additional terms which are not
        # listed in the regression (elongation, Z_eff, I_p)
        np.testing.assert_almost_equal(lambda_q_value, 0.94e-3, decimal=4)

        assert min_v < lambda_q_value < max_v


class TestPLH:
    def test_ITER(self):
        """
        As per https://infoscience.epfl.ch/record/135655/files/1742-6596_123_1_012033.pdf
        """
        p_lh_threshold, min_v, max_v = P_LH(1e20, 5.3, 3.1, 6.2, error=True)
        # No value provided for this scaling (3), but ballpark is right
        np.testing.assert_approx_equal(p_lh_threshold, 90e6, significant=1)
        assert min_v < p_lh_threshold < max_v


class TestIPB98y2:
    def test_ITER(self):
        """
        As per I dunno
        """

        value = IPB98y2(15e6, 5.3, 100e6, 1e20, 2.5, 6.2, 3.1, 1.59)

        assert round(value, 0) == pytest.approx(3.0, rel=0, abs=EPS)
