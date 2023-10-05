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

from bluemira.structural.node import Node


class TestNode:
    def test_distances(self):
        n1 = Node(0, 0, 0, 0)
        rng = np.random.default_rng()
        for _ in range(100):
            v = 1000 * rng.random(3) - 1000
            dx, dy, dz = v
            n2 = Node(dx, dy, dz, 1)
            assert np.isclose(n1.distance_to_other(n2), np.sqrt(np.sum(v**2)))

    def test_assignment(self):
        with pytest.raises(AttributeError):
            Node(0, 0, 0, 0).dummy = 4

    def test_defaultsupports(self):
        node = Node(0, 0, 0, 0)
        assert not node.supports.all()
        node.add_support(np.array([True, True, True, True, True, True]))
        assert node.supports.all()
        node.add_support(np.array([True, True, True, True, True, False]))
        assert not node.supports[5]
        assert node.supports[:5].all()
        node.clear_supports()
        assert not node.supports.all()
