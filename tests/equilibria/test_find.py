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

import json
from pathlib import Path

import numpy as np
import pytest

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.find import (
    _in_plasma,
    find_LCFS_separatrix,
    find_local_minima,
    get_legs,
    inv_2x2_matrix,
)

DATA = get_bluemira_path("equilibria/test_data", subfolder="tests")


def test_find_local_minima():
    rng = np.random.default_rng()
    for _ in range(10):
        array = np.ones((100, 100))
        i, j = rng.integers(0, 99, 2)
        array[i, j] = 0
        ii, jj = find_local_minima(array)
        assert len(ii) == 1
        assert len(jj) == 1
        assert ii[0] == i
        assert jj[0] == j

    array = np.ones((100, 100))
    array[1, 0] = 0
    array[-2, -2] = 0
    array[-2, 1] = 0
    array[1, -2] = 0
    array[0, 50] = 0
    array[50, 0] = 0

    ii, jj = find_local_minima(array)

    assert len(ii) == 6
    assert len(jj) == 6
    assert (np.sort(ii) == np.array([0, 1, 1, 50, 98, 98])).all()
    assert (np.sort(jj) == np.array([0, 0, 1, 50, 98, 98])).all()


def test_inv_2x2_jacobian():
    a, b, c, d = 3.523, 5.0, 6, 0.2
    inv_jac_true = np.linalg.inv(np.array([[a, b], [c, d]]))
    inv_jac = inv_2x2_matrix(a, b, c, d)
    assert np.allclose(inv_jac_true, inv_jac)


class TestFindLCFSSeparatrix:
    def test_other_grid(self):
        sof = Equilibrium.from_eqdsk(Path(DATA, "eqref_OOB.json"))
        psi = sof.psi()
        o_points, x_points = sof.get_OX_points(psi)
        grid_tol = np.hypot(sof.grid.dx, sof.grid.dz)
        for tolerance in [1e-6, 1e-7, 1e-8, 1e-9]:
            lcfs, separatrix = find_LCFS_separatrix(
                sof.x,
                sof.z,
                sof.psi(),
                o_points=o_points,
                x_points=x_points,
                psi_n_tol=tolerance,
            )
            assert lcfs.closed
            assert not separatrix.closed
            primary_xp = x_points[0]
            distances = lcfs.distance_to([primary_xp.x, 0, primary_xp.z])
            assert np.amin(distances) <= grid_tol
            distances = separatrix.distance_to([primary_xp.x, 0, primary_xp.z])
            assert np.amin(distances) <= grid_tol

    def test_double_null(self):
        sof = Equilibrium.from_eqdsk(Path(DATA, "DN-DEMO_eqref.json"))
        psi = sof.psi()
        o_points, x_points = sof.get_OX_points(psi)
        grid_tol = np.hypot(sof.grid.dx, sof.grid.dz)
        for tolerance in [1e-6, 1e-7, 1e-8, 1e-9]:
            lcfs, separatrix = find_LCFS_separatrix(
                sof.x,
                sof.z,
                sof.psi(),
                o_points=o_points,
                x_points=x_points,
                psi_n_tol=tolerance,
                double_null=True,
            )

            assert lcfs.closed
            primary_xp = x_points[0]
            distances = lcfs.distance_to([primary_xp.x, 0, primary_xp.z])
            assert np.amin(distances) <= grid_tol

            assert isinstance(separatrix, list)
            for loop in separatrix:
                assert not loop.closed
                distances = loop.distance_to([primary_xp.x, 0, primary_xp.z])
                assert np.amin(distances) <= grid_tol


class TestInPlasma:
    def test_recursion(self):
        with open(Path(DATA, "in_plasma_test.json"), "rb") as f:
            data = json.load(f)
        x, z = np.array(data["X"]), np.array(data["Z"])
        lcfs = np.array(data["LCFS"])
        result = np.array(data["result"])
        mask = np.zeros_like(x)

        result2 = _in_plasma(x, z, mask, lcfs)
        assert np.allclose(result, result2)


class TestGetLegs:
    @classmethod
    def setup_class(cls):
        cls.sn_eq = Equilibrium.from_eqdsk(Path(DATA, "eqref_OOB.json"))
        cls.dn_eq = Equilibrium.from_eqdsk(Path(DATA, "DN-DEMO_eqref.json"))

    @pytest.mark.parametrize("n_layers", [2, 3, 5])
    def test_single_null(self, n_layers):
        legs = get_legs(self.sn_eq, n_layers, 0.2)
        assert len(legs) == 2
        assert "lower_inner" in legs
        assert "lower_outer" in legs
        x_point = self.sn_eq.get_OX_points()[1][0]
        for leg_group in legs.values():
            assert len(leg_group) == n_layers
            for leg in leg_group:
                self.assert_valid_leg(leg, x_point)
                self.assert_valid_leg(leg, x_point)

    def test_single_one_layer(self):
        legs = get_legs(self.sn_eq, 1, 0.0)
        assert len(legs) == 2
        assert "lower_inner" in legs
        assert "lower_outer" in legs
        assert len(legs["lower_inner"]) == 1
        assert len(legs["lower_outer"]) == 1
        x1 = legs["lower_inner"][0].x[0]
        legs = get_legs(self.sn_eq, 1, 1.0)
        assert len(legs) == 2
        assert "lower_inner" in legs
        assert "lower_outer" in legs
        assert len(legs["lower_inner"]) == 1
        assert len(legs["lower_outer"]) == 1
        x2 = legs["lower_inner"][0].x[0]
        assert np.isclose(x1, x2)

    @pytest.mark.parametrize("n_layers", [2, 3, 5])
    def test_double_null(self, n_layers):
        legs = get_legs(self.dn_eq, n_layers, 0.2)
        x_points = self.dn_eq.get_OX_points()[1][:2]
        x_points.sort(key=lambda xp: xp.z)
        assert len(legs) == 4
        assert "lower_inner" in legs
        assert "lower_outer" in legs
        assert "upper_inner" in legs
        assert "upper_outer" in legs
        for name, leg_group in legs.items():
            assert len(leg_group) == n_layers
            x_p = x_points[0] if "lower" in name else x_points[1]
            for leg in leg_group:
                self.assert_valid_leg(leg, x_p)

    def test_double_one_layer(self):
        legs = get_legs(self.dn_eq, 1, 0.0)
        assert len(legs) == 4
        assert "lower_inner" in legs
        assert "lower_outer" in legs
        assert len(legs["lower_inner"]) == 1
        assert len(legs["lower_outer"]) == 1
        x1 = legs["lower_inner"][0].x[0]
        legs = get_legs(self.dn_eq, 1, 1.0)
        assert len(legs) == 4
        assert "lower_inner" in legs
        assert "lower_outer" in legs
        assert len(legs["lower_inner"]) == 1
        assert len(legs["lower_outer"]) == 1
        x2 = legs["lower_inner"][0].x[0]
        assert np.isclose(x1, x2)

    def assert_valid_leg(self, leg, x_point):
        assert np.isclose(leg.z[0], x_point.z)
