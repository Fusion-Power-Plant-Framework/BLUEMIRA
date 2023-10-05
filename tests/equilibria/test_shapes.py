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
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull

from bluemira.equilibria.flux_surfaces import ClosedFluxSurface
from bluemira.equilibria.shapes import (
    CunninghamLCFS,
    JohnerLCFS,
    KuiroukidisLCFS,
    ManickamLCFS,
    ZakharovLCFS,
    _generate_theta,
    flux_surface_cunningham,
    flux_surface_hirshman,
    flux_surface_johner,
    flux_surface_kuiroukidis,
    flux_surface_manickam,
)


class TestCunningham:
    @classmethod
    def setup_class(cls):
        cls.f, cls.ax = plt.subplots(4, 2)

    @pytest.mark.parametrize(
        ("kappa", "delta", "delta2", "ax", "label"),
        [
            pytest.param(1.6, 0.33, 0.5, [0, 0], "Normal", id="Normal"),
            pytest.param(1.6, -0.33, 0.0, [0, 1], "Negative delta", id="Negative delta"),
            pytest.param(1.6, 0.33, 0.5, [1, 1], "Indent 1.5", id="Indent 1.5"),
            pytest.param(1.6, 0.33, 0.25, [1, 0], "Indent 1.5", id="Indent 1.5_2"),
            pytest.param(1.6, 0, 0.3, [2, 0], "Indent $delta$=0", id="Indent delta=0"),
            pytest.param(
                1.6, -0.2, 0.5, [2, 1], "Negative indent", id="Negative indent"
            ),
            pytest.param(
                1,
                0,
                0.0,
                [3, 0],
                "$\\delta$=0\n$\\kappa$=1",
                id="$\\delta$=0\n$\\kappa$=1",
            ),
            pytest.param(1.6, 0, 0.0, [3, 1], "$\\delta$=0", id="$\\delta$=0"),
        ],
    )
    def test_cunningham(self, kappa, delta, delta2, ax, label):
        ax0, ax1 = ax
        f_s = flux_surface_cunningham(9, 0, 3, kappa, delta, delta2=delta2, n=100)
        self.ax[ax0, ax1].plot(f_s.x, f_s.z, label=label)
        self.ax[ax0, ax1].set_aspect("equal")
        self.ax[ax0, ax1].legend()

    @classmethod
    def teardown_class(cls):
        cls.f.suptitle("Cunningham parameterisations")
        plt.close(cls.f)


class TestHirschman:
    @classmethod
    def setup_class(cls):
        cls.f, cls.ax = plt.subplots(2, 2)

    @pytest.mark.parametrize(
        ("a", "kappa", "ax", "label"),
        [
            pytest.param(
                2.0,
                1.0,
                [0, 0],
                "$a$ = 2.0, $\\kappa$ = 1.0",
                id="$a$ = 2.0, $\\kappa$ = 1.0",
            ),
            pytest.param(
                2.0,
                2.0,
                [0, 1],
                "$a$ = 2.0, $\\kappa$ = 2.0",
                id="$a$ = 2.0, $\\kappa$ = 2.0",
            ),
            pytest.param(
                3.0,
                1.5,
                [1, 1],
                "$a$ = 3.0, $\\kappa$ = 1.5",
                id="$a$ = 3.0, $\\kappa$ = 1.5",
            ),
            pytest.param(
                3.0,
                1.75,
                [1, 0],
                "$a$ = 3.0, $\\kappa$ = 1.75",
                id="$a$ = 3.0, $\\kappa$ = 1.75",
            ),
        ],
    )
    def test_hirshman(self, a, kappa, ax, label):
        f_s = flux_surface_hirshman(9, 0, a, kappa, n=100)

        ax0, ax1 = ax
        self.ax[ax0, ax1].plot(f_s.x, f_s.z, label=label)
        self.ax[ax0, ax1].set_aspect("equal")
        self.ax[ax0, ax1].legend()

    @classmethod
    def teardown_class(cls):
        cls.f.suptitle("Hirschman parameterisations")
        plt.show()
        plt.close(cls.f)


class TestManickam:
    @classmethod
    def setup_class(cls):
        cls.f, cls.ax = plt.subplots(4, 2)

    @pytest.mark.parametrize(
        ("kappa", "delta", "indent", "ax", "label"),
        [
            pytest.param(1.6, 0.33, 0, [0, 0], "Normal", id="Normal"),
            pytest.param(1.6, -0.33, 0, [0, 1], "Negative delta", id="Negative delta"),
            pytest.param(1.6, 0.33, 0.5, [1, 1], "Indent 0.5", id="Indent 0.5"),
            pytest.param(1.6, 0.33, 1.5, [1, 0], "Indent 1.5", id="Indent 1.5"),
            pytest.param(1.6, 0, 1.5, [2, 0], "Indent $delta$=0", id="Indent delta=0"),
            pytest.param(
                1.6, -0.2, -1.5, [2, 1], "Negative indent", id="Negative indent"
            ),
            pytest.param(1.6, 0, 0, [3, 1], "$\\delta$=0", id="$\\delta$=0"),
            pytest.param(
                1,
                0,
                0,
                [3, 0],
                "$\\delta$=0\n$\\kappa$=1",
                id="$\\delta$=0\n$\\kappa$=1",
            ),
        ],
    )
    def test_manickam(self, kappa, delta, indent, ax, label):
        f_s = flux_surface_manickam(9, 0, 3, kappa, delta, indent=indent, n=100)

        ax0, ax1 = ax
        self.ax[ax0, ax1].plot(f_s.x, f_s.z, label=label)
        self.ax[ax0, ax1].set_aspect("equal")
        self.ax[ax0, ax1].legend()

    @classmethod
    def teardown_class(cls):
        cls.f.suptitle("Manickam parameterisations")
        plt.show()
        plt.close(cls.f)


class TestKuiroukidis:
    fixture = (
        pytest.param(6.2, 3.1, 1.55, 2.0, -0.5, -0.5, [0, 0]),
        pytest.param(6.2, 3.1, 1.55, 2.0, 0.5, 0.5, [0, 1]),
        pytest.param(6.2, 3.1, 1.55, 2.0, -0.5, 0.5, [0, 2]),
        pytest.param(6.2, 3.1, 1.55, 2.0, 0.5, -0.5, [1, 0]),
        pytest.param(1.717, 1.717 / 0.5151, 1.55, 2.0, 0.15, 0.15, [1, 1]),
        pytest.param(9, 9 / 3, 1.55, 1.8, 0.333, 0.333, [1, 2]),
    )

    @classmethod
    def setup_class(cls):
        cls.f, cls.ax = plt.subplots(2, 3)

    @pytest.mark.parametrize(
        ("R_0", "A", "kappa_u", "kappa_l", "delta_u", "delta_l", "ax"),
        fixture,
    )
    def test_kuiroukidis_coords(self, R_0, A, kappa_u, kappa_l, delta_u, delta_l, ax):
        flux_surface = flux_surface_kuiroukidis(
            R_0, 0, R_0 / A, kappa_u, kappa_l, delta_u, delta_l, 8, 100
        )
        arg_inner = np.argmin(flux_surface.x)
        arg_outer = np.argmax(flux_surface.x)
        arg_lower = np.argmin(flux_surface.z)
        arg_upper = np.argmax(flux_surface.z)

        x_lower = R_0 - delta_l * R_0 / A
        z_lower = -kappa_l * R_0 / A
        x_upper = R_0 - delta_u * R_0 / A
        z_upper = kappa_u * R_0 / A
        x_inner = R_0 - R_0 / A
        z_inner = 0.0
        x_outer = R_0 + R_0 / A
        z_outer = 0.0

        n1, n2 = ax
        self.ax[n1, n2].plot(flux_surface.x, flux_surface.z)
        self.ax[n1, n2].set_xlabel("x")
        self.ax[n1, n2].set_ylabel("z")
        self.ax[n1, n2].set_aspect("equal")

        np.testing.assert_allclose(
            np.array([x_lower, z_lower]), flux_surface.xz.T[arg_lower]
        )
        np.testing.assert_allclose(
            np.array([x_upper, z_upper]), flux_surface.xz.T[arg_upper]
        )
        np.testing.assert_allclose(
            np.array([x_inner, z_inner]), flux_surface.xz.T[arg_inner], atol=1e-12
        )
        np.testing.assert_allclose(
            np.array([x_outer, z_outer]), flux_surface.xz.T[arg_outer], atol=1e-12
        )

    @pytest.mark.parametrize(
        ("R_0", "A", "kappa_u", "kappa_l", "delta_u", "delta_l", "ax"),
        fixture,
    )
    def test_kuiroukidis_ccw(
        self, R_0, A, kappa_u, kappa_l, delta_u, delta_l, ax  # noqa: ARG002
    ):
        flux_surface = flux_surface_kuiroukidis(
            R_0, 0, R_0 / A, kappa_u, kappa_l, delta_u, delta_l, 8, 100
        )
        hull = ConvexHull(flux_surface.xz.T)
        np.testing.assert_approx_equal(hull.area, flux_surface.length)

    @classmethod
    def teardown_class(cls):
        cls.f.suptitle("Kuiroukidis parameterisations")
        plt.subplots_adjust(hspace=0.4)
        plt.show()
        plt.close(cls.f)


johner_names = [
    "kappa_u",
    "kappa_l",
    "delta_u",
    "delta_l",
    "psi_u_neg",
    "psi_u_pos",
    "psi_l_neg",
    "psi_l_pos",
    "ax",
    "label",
]
johner_params = [
    [1.6, 1.9, 0.33, 0.4, -20, 5, 60, 30, [0, 0], "SN upper, positive $\\delta$"],
    [1.9, 1.6, 0.4, 0.33, 60, 30, -20, 5, [1, 0], "SN down, positive $\\delta$"],
    [1.6, 1.9, 0.33, 0.4, -20, 5, 60, 30, [2, 0], "SN down, positive $\\delta$, Z0=5"],
    [1.6, 1.6, 0.4, 0.4, 60, 30, 60, 30, [3, 0], "DN, positive $\\delta$"],
    [1.6, 1.9, -0.33, -0.4, -20, 5, 60, 30, [0, 1], "SN upper, negative $\\delta$"],
    [1.9, 1.6, -0.4, -0.33, 60, 30, -20, 5, [1, 1], "SN down, negative $\\delta$"],
    [
        1.6,
        1.9,
        -0.33,
        -0.4,
        -20,
        5,
        60,
        30,
        [2, 1],
        "SN down, negative $\\delta$, Z0=-5",
    ],
    [1.9, 1.9, -0.40, -0.4, 60, 20, 60, 20, [3, 1], "DN, negative $\\delta$"],
]

johner_params = [
    pytest.param(dict(zip(johner_names, p)), id=p[-1]) for p in johner_params
]


class TestJohner:
    @classmethod
    def setup_class(cls):
        cls.f, cls.ax = plt.subplots(4, 2)

    @pytest.mark.parametrize("kwargs", johner_params)
    def test_johner(self, kwargs):
        ax0, ax1 = kwargs.pop("ax")
        label = kwargs.pop("label")
        f_s = flux_surface_johner(9, 0, 9 / 3, n=100, **kwargs)
        self.ax[ax0, ax1].plot(f_s.x, f_s.z, label=label)
        self.ax[ax0, ax1].set_aspect("equal")
        self.ax[ax0, ax1].legend()

    @classmethod
    def teardown_class(cls):
        cls.f.suptitle("Johner parameterisations")
        plt.show()
        plt.close(cls.f)


class TestJohnerCAD:
    def test_segments(self):
        p = JohnerLCFS()
        wire = p.create_shape()
        assert len(wire._boundary) == 4

    def test_symmetry(self):
        p_pos = JohnerLCFS()
        p_neg = JohnerLCFS()

        p_pos.adjust_variable("delta_u", 0.4)
        p_pos.adjust_variable("delta_l", 0.4)
        p_neg.adjust_variable("delta_u", -0.4, lower_bound=-0.5)
        p_neg.adjust_variable("delta_l", -0.4, lower_bound=-0.5)
        wire_pos = p_pos.create_shape()
        wire_neg = p_neg.create_shape()

        assert np.isclose(wire_pos.length, wire_neg.length)


class TestKuiroukidisCAD:
    def test_segments(self):
        p = KuiroukidisLCFS()
        wire = p.create_shape()
        assert len(wire._boundary) == 4

    def test_symmetry(self):
        p_pos = KuiroukidisLCFS()
        p_neg = KuiroukidisLCFS()

        p_pos.adjust_variable("delta_u", 0.4)
        p_pos.adjust_variable("delta_l", 0.4)
        p_neg.adjust_variable("delta_u", -0.4, lower_bound=-0.5)
        p_neg.adjust_variable("delta_l", -0.4, lower_bound=-0.5)
        wire_pos = p_pos.create_shape()
        wire_neg = p_neg.create_shape()

        assert np.isclose(wire_pos.length, wire_neg.length)


class TestManickamCunninghamZakahrovCAD:
    @pytest.mark.parametrize(
        "parameterisation", [ManickamLCFS, CunninghamLCFS, ZakharovLCFS]
    )
    def test_segments(self, parameterisation):
        p = parameterisation()
        wire = p.create_shape()
        assert len(wire._boundary) == 1

    @pytest.mark.parametrize(
        "parameterisation", [ManickamLCFS, CunninghamLCFS, ZakharovLCFS]
    )
    def test_symmetry(self, parameterisation):
        p_pos = parameterisation()
        p_neg = parameterisation()

        p_pos.adjust_variable("delta", 0.4)
        p_neg.adjust_variable("delta", -0.4, lower_bound=-0.5)
        wire_pos = p_pos.create_shape()
        wire_neg = p_neg.create_shape()

        assert np.isclose(wire_pos.length, wire_neg.length)

    @pytest.mark.parametrize("parameterisation", [ZakharovLCFS])
    @pytest.mark.parametrize("delta", [0.33, -0.33, 0.5, -0.5, 0])
    def test_delta(self, parameterisation, delta):
        pos = parameterisation()
        lb, ub = 0.9 * delta, 1.1 * delta
        lb, ub = min(lb, ub), max(lb, ub)
        pos.adjust_variable("delta", delta, lower_bound=lb, upper_bound=ub)
        wire = pos.create_shape(n_points=25)

        fs = ClosedFluxSurface(wire.discretize(ndiscr=1000, byedges=True))
        np.testing.assert_almost_equal(delta, fs.delta)

    @pytest.mark.xfail
    @pytest.mark.parametrize("parameterisation", [CunninghamLCFS, ManickamLCFS])
    @pytest.mark.parametrize("delta", [0.33, -0.33, 0.5, -0.5])
    def test_delta_fail(self, parameterisation, delta):
        pos = parameterisation()
        lb, ub = 0.9 * delta, 1.1 * delta
        lb, ub = min(lb, ub), max(lb, ub)
        pos.adjust_variable("delta", delta, lower_bound=lb, upper_bound=ub)
        wire = pos.create_shape(n_points=25)

        fs = ClosedFluxSurface(wire.discretize(ndiscr=1000, byedges=True))
        np.testing.assert_almost_equal(delta, fs.delta)

    @pytest.mark.parametrize(
        "parameterisation", [ManickamLCFS, CunninghamLCFS, ZakharovLCFS]
    )
    @pytest.mark.parametrize("kappa", [1.0, 1.5, 2.0])
    def test_kappa(self, parameterisation, kappa):
        pos = parameterisation()
        lb, ub = 0.9 * kappa, 1.1 * kappa
        pos.adjust_variable("kappa", kappa, lower_bound=lb, upper_bound=ub)
        wire = pos.create_shape(n_points=25)

        fs = ClosedFluxSurface(wire.discretize(ndiscr=1000, byedges=True))
        np.testing.assert_almost_equal(kappa, fs.kappa)


class TestGenerateTheta:
    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_quarts(self, n):
        t = _generate_theta(n)
        assert t.size == n
        np.testing.assert_allclose(t, np.pi * np.array([0, 0.5, 1, 1.5, 2])[:n])

    @pytest.mark.parametrize(
        "n", [5, 6, 7, 8, 20, 21, 22, 23, 24, 25, 100, 200, 250, 256, 1001]
    )
    def test_n(self, n):
        t = _generate_theta(n)
        assert t.size == n

    @pytest.mark.parametrize("n", [250, 256, 1000, 100, 200])
    def test_high_n(self, n):
        t = _generate_theta(n)
        assert t.size == n
