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

from bluemira.geometry.bound_box import BoundingBox
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import PolySpline
from bluemira.geometry.tools import boolean_cut, make_polygon, revolve_shape


class TestBoundingBox:
    rng = np.random.default_rng()

    def test_null(self):
        x, y, z = np.zeros(100), np.zeros(100), np.zeros(100)
        xb, yb, zb = BoundingBox.from_xyz(x, y, z).get_box_arrays()
        assert np.all(xb == 0)
        assert np.all(yb == 0)
        assert np.all(zb == 0)

    def test_random(self):
        x, y, z = self.rng.random(100), self.rng.random(100), self.rng.random(100)
        args = self.rng.integers(0, 100, 8)
        x[args] = np.array([-2, -2, -2, -2, 2, 2, 2, 2])
        y[args] = np.array([-2, -2, 2, 2, 2, -2, -2, 2])
        z[args] = np.array([-2, 2, -2, 2, -2, 2, -2, 2])
        xb, yb, zb = BoundingBox.from_xyz(x, y, z).get_box_arrays()

        assert np.allclose(xb, np.array([-2, -2, -2, -2, 2, 2, 2, 2]))
        assert np.allclose(yb, np.array([-2, -2, 2, 2, -2, -2, 2, 2]))
        assert np.allclose(zb, np.array([-2, 2, -2, 2, -2, 2, -2, 2]))


class TestHardBoundingBox:
    ps = PolySpline(
        {
            "bottom": {"value": 0.509036},
            "flat": {"value": 1},
            "height": {"value": 10.1269},
            "lower": {"value": 0.2},
            "tilt": {"value": 19.6953},
            "top": {"value": 0.46719},
            "upper": {"value": 0.326209},
            "x1": {"value": 5},
            "x2": {"value": 11.8222},
            "z2": {"value": -0.170942},
        }
    )

    cut_box = BluemiraFace(
        make_polygon({"x": [0, 15, 15, 0], "z": [-7, -7, -5, -5], "y": 0}, closed=True)
    )
    wire = boolean_cut(ps.create_shape(), cut_box)[0]
    temp = wire.deepcopy()
    temp.close()
    solid = revolve_shape(BluemiraFace(temp), degree=159)

    @pytest.mark.xfail
    def test_bad_bounding_box(self):
        assert np.isclose(self.wire.bounding_box.z_min, -5.0)

    @pytest.mark.parametrize("tol", [10.0, 1.0, 0.1, 0.001])
    def test_opt_bounding_box(self, tol):
        bb = self.wire.get_optimal_bounding_box(tolerance=tol)
        assert np.isclose(bb.z_min, -5.0)

    @pytest.mark.parametrize("tol", [10.0, 1.0, 0.1, 0.01])
    def test_opt_bounding_box_solid(self, tol):
        solid = self.solid.deepcopy()
        vertices, indices = solid._tessellate(1.0)
        bb = self.solid.get_optimal_bounding_box(tolerance=tol)
        vertices2, indices2 = solid._tessellate(1.0)
        assert np.isclose(bb.z_min, -5.0)
        # Test that bounding box via tesselation did not modify properties
        np.testing.assert_allclose(vertices, vertices2)
        np.testing.assert_allclose(indices, indices2)
        vertices3, _ = solid._tessellate(0.01)
        assert vertices3.shape != vertices2.shape

    @pytest.mark.parametrize("tol", [0.0, -1e-9])
    def test_bad_tolerace(self, tol):
        with pytest.raises(ValueError):  # noqa: PT011
            self.wire.get_optimal_bounding_box(tolerance=tol)
