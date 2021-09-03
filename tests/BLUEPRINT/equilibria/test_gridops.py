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
import pytest

# =============================================================================
# Grid object and grid operations
# =============================================================================
from BLUEPRINT.equilibria.gridops import Grid


class TestGrid:
    def test_init(self):
        g = Grid(0, 20, -10, 10, 100, 100)
        assert g.x_min != 0
        g = Grid(10, 5, -10, 10, 100, 100)
        assert g.x_min == 5
        assert g.x_max == 10
        g = Grid(5, 10, 10, -10, 100, 100)
        assert g.z_min == -10
        assert g.z_max == 10

    def test_point_inside(self):
        g = Grid(4, 10, -10, 10, 65, 65)
        points = [[5, 0], [5, 5], [5, 1], [5, -1], [5, -5]]
        for p in points:
            assert g.point_inside(*p)

        x, z = g.bounds
        points = list(zip(x, z))
        for p in points:
            assert g.point_inside(*p)

        fails = [[-1, 0], [-10, 0], [-10, -10], [5, -12], [5, 12], [100, 0]]
        for f in fails:
            assert not g.point_inside(*f)


if __name__ == "__main__":
    pytest.main([__file__])
