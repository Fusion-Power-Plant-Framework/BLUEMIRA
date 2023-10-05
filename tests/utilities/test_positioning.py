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

from bluemira.geometry.tools import make_circle, make_polygon
from bluemira.utilities.error import PositionerError
from bluemira.utilities.positioning import (
    PathInterpolator,
    PositionMapper,
    RegionInterpolator,
)


class TestPathInterpolator:
    @classmethod
    def setup_class(cls):
        x = [0, 2, 4]
        z = [0, 1, 0]
        cls.polygon = make_polygon({"x": x, "z": z})
        cls.circle = make_circle(center=(0, 0, 0), axis=(0, -1, 0), radius=10)

    @pytest.mark.parametrize(
        ("alpha", "point"), [(0.0, [0, 0]), (0.5, [2, 1]), (1.0, [4, 0])]
    )
    def test_open(self, alpha, point):
        interpolator = PathInterpolator(self.polygon)

        x0, z0 = interpolator.to_xz(alpha)
        assert np.isclose(x0, point[0])
        assert np.isclose(z0, point[1])
        assert np.isclose(interpolator.to_L(*point), alpha)

    @pytest.mark.parametrize(
        ("alpha", "point"),
        [
            (0.0, [10, 0]),
            (0.25, [0, 10]),
            (0.5, [-10, 0]),
            (0.75, [0, -10]),
        ],
    )
    def test_closed(self, alpha, point):
        interpolator = PathInterpolator(self.circle)
        x0, z0 = interpolator.to_xz(alpha)
        assert np.isclose(x0, point[0])
        assert np.isclose(z0, point[1])
        assert np.isclose(interpolator.to_L(*point), alpha)

    def test_straight(self):
        line = make_polygon([[5, 10], [0, 0], [-10, 10]])
        interpolator = PathInterpolator(line)
        x05, z05 = interpolator.to_xz(0.5)
        assert np.isclose(x05, 7.5)
        assert np.isclose(z05, 0)
        l05 = interpolator.to_L(7.5, 0)
        assert np.isclose(l05, 0.5)


class TestRegionInterpolator:
    def test_bad_shapes(self):
        x = [0, 4, 2, 4, 0, 2]
        z = [0, 0, 1, 2, 2, 1]
        concave_polygon = make_polygon({"x": x, "z": z}, closed=True)
        with pytest.raises(PositionerError):
            RegionInterpolator(concave_polygon)

        open_polygon = make_polygon({"x": x, "z": z})
        with pytest.raises(PositionerError):
            RegionInterpolator(open_polygon)

    def test_circle(self):
        circle = make_circle(center=(0, 0, 0), axis=(0, 1, 0), radius=10)
        interpolator = RegionInterpolator(circle)

        xc, zc = interpolator.to_xz((0.5, 0.5))
        assert np.isclose(xc, 0)
        assert np.isclose(zc, 0)
        l0, l1 = interpolator.to_L(0, 0)
        assert np.isclose(l0, 0.5)
        assert np.isclose(l1, 0.5)


class TestPositionMapper:
    @classmethod
    def setup_class(cls):
        x = [2, 4, 4, 2]
        z = [1, 1, 2, 2]
        convex_polygon = make_polygon({"x": x, "z": z}, closed=True)
        circle = make_circle(center=(0, 0, 0), axis=(0, -1, 0), radius=10)

        interpolators = {
            "circle_path": PathInterpolator(circle),
            "circle_region": RegionInterpolator(circle),
            "polygon_region": RegionInterpolator(convex_polygon),
        }
        cls.mapper = PositionMapper(interpolators)

    def test_dimensionality(self):
        assert self.mapper.dimension == 5

    def test_to_xz(self):
        l_values = [0.5, 0.5, 0.5, 0.5, 0.5]
        positions = np.array(self.mapper.to_xz(l_values))
        assert positions.shape == (2, 3)

    def test_to_L(self):
        x = [10, 3, 0]
        z = [3, 1.5, 0]
        l_values = self.mapper.to_L(x, z)
        assert len(l_values) == 5
