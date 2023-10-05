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

from bluemira.base.constants import raw_uc
from bluemira.plasma_physics.collisions import coulomb_logarithm


class TestCoulombLogarithm:
    """
    Reference values taken from:
        Goldston and Rutherford, "Introduction to Plasma Physics", 1995, Table 11.1
    """

    @pytest.mark.parametrize(
        ("density", "temp_in_ev", "ref_value"),
        [
            # Solar wind
            (10.0**7, 10.0, 26),
            # Van Allen belts
            (10.0**9, 10.0**2, 26),
            # Earth's ionosphere
            (10.0**11, 10.0**-1, 14),
            # Solar corona
            (10.0**13, 10.0**2, 21),
            # Gas discharge
            (10.0**16, 10.0**0, 12),
            # Process plasma
            (10.0**18, 10.0**2, 15),
            # Fusion experiment
            (10.0**19, 10.0**3, 17),
            # Fusion reactor
            (10.0**20, 10.0**4, 18),
            # Hartmut's case
            (10.0**20, 1000.0, 16.5),
        ],
    )
    def test_coulomb_logarithm_values(self, density, temp_in_ev, ref_value):
        temp_in_c = raw_uc(temp_in_ev, "eV", "celsius")
        value = round(coulomb_logarithm(temp_in_c, density), 1)
        np.testing.assert_allclose(value, ref_value, rtol=0.054)
