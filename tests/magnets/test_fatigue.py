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

from bluemira.magnets.fatigue import (
    EllipticalEmbeddedCrack,
    QuarterEllipticalCornerCrack,
    SemiEllipticalSurfaceCrack,
)


class TestFatigueCracks:
    @pytest.mark.parametrize(
        "crack_cls",
        [
            SemiEllipticalSurfaceCrack,
            QuarterEllipticalCornerCrack,
            EllipticalEmbeddedCrack,
        ],
    )
    @pytest.mark.parametrize("area,aspect_ratio", [[1, 3], [2e-6, 2]])
    def test_init_by_area(self, crack_cls, area, aspect_ratio):
        crack = crack_cls.from_area(area, aspect_ratio)
        assert np.isclose(crack.area, area, atol=1e-9, rtol=0.0)
