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

from pathlib import Path

import numpy as np
import pytest

from bluemira.base.constants import EPS
from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.coils import Coil, CoilSet, SymmetricCircuit
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.harmonics import (
    coil_harmonic_amplitude_matrix,
    coils_outside_sphere_vacuum_psi,
    collocation_points,
    get_psi_harmonic_amplitudes,
    harmonic_amplitude_marix,
    lcfs_fit_metric,
    spherical_harmonic_approximation,
)
from bluemira.geometry.coordinates import Coordinates, in_polygon

TEST_PATH = get_bluemira_path("equilibria/test_data", subfolder="tests")


def test_lcfs_fit_metric():
    xa = [1, 2, 2, 1, 1]
    xb = [3, 4, 4, 3, 3]
    xc = [1.5, 2.5, 2.5, 1.5, 1.5]
    za = [1, 1, 2, 2, 1]
    zc = [1.5, 1.5, 2.5, 2.5, 1.5]

    poly1 = Coordinates({"x": xa, "z": za})
    poly2 = Coordinates({"x": xb, "z": za})
    poly3 = Coordinates({"x": xc, "z": zc})
    poly4 = Coordinates({"x": xc, "z": za})

    assert lcfs_fit_metric(poly1, poly1) == 0
    assert lcfs_fit_metric(poly1, poly2) == 1
    assert lcfs_fit_metric(poly1, poly3) == pytest.approx(0.75, rel=0, abs=EPS)
    assert lcfs_fit_metric(poly1, poly4) == pytest.approx(0.5, rel=0, abs=EPS)


def test_harmonic_amplitude_marix():
    r = np.array([1, 1, 1])
    theta = np.array([0, np.pi, 2 * np.pi])
    n = 3
    d = 2
    r_t = 1.0

    test_output = harmonic_amplitude_marix(r, theta, r_t)

    assert test_output.shape == (n, d)

    assert ((test_output - np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])) == 0).all()


def test_coil_harmonic_amplitude_matrix():
    coil = Coil(x=4, z=10, current=2e6, dx=1, dz=0.5, j_max=5.0, b_max=50, name="PF_1")
    circuit = SymmetricCircuit(
        Coil(
            x=1.5,
            z=6,
            current=1e6,
            dx=0.25,
            dz=0.5,
            j_max=10.0,
            b_max=100,
            ctype="PF",
            name="PF_2",
        ),
        Coil(
            x=1.5,
            z=-6,
            current=1e6,
            dx=0.25,
            dz=0.5,
            j_max=10.0,
            b_max=100,
            ctype="PF",
            name="PF_3",
        ),
    )

    coilset = CoilSet(coil, circuit)

    d = 2
    r_t = 1

    test_out_matrx = coil_harmonic_amplitude_matrix(coilset, d, r_t)

    assert test_out_matrx.shape[1] == len(coilset.x)
    assert test_out_matrx.shape[0] == d


def test_collocation_points():
    n_points = 8

    x = [1, 1.5, 2, 2.1, 2, 1.5, 1, 0.9, 1]
    z = [-1.8, -1.9, -1.8, 0, 1.8, 1.9, 1.8, 0, -1.8]
    plasma_boundary = Coordinates({"x": x, "z": z})

    point_type_1 = "arc"
    point_type_2 = "arc_plus_extrema"
    point_type_3 = "random"
    point_type_4 = "random_plus_extrema"

    colloc1 = collocation_points(n_points, plasma_boundary, point_type_1)
    colloc2 = collocation_points(n_points, plasma_boundary, point_type_2)
    colloc3 = collocation_points(n_points, plasma_boundary, point_type_3)
    colloc4 = collocation_points(n_points, plasma_boundary, point_type_4)

    assert colloc1.r.shape[0] == 8
    assert colloc2.r.shape[0] == 12
    assert colloc3.r.shape[0] == 8
    assert colloc4.r.shape[0] == 12

    for x, z in zip(colloc2.x, colloc2.z):
        assert in_polygon(x, z, plasma_boundary.xz.T, include_edges=True)

    for x, z in zip(colloc4.x, colloc4.z):
        assert in_polygon(x, z, plasma_boundary.xz.T, include_edges=True)


def test_coils_outside_sphere_vacuum_psi():
    path = get_bluemira_path("equilibria/test_data", subfolder="tests")
    eq = Equilibrium.from_eqdsk(Path(path, "SH_test_file.json"))

    test_v_psi, test_p_psi, test_coilset = coils_outside_sphere_vacuum_psi(eq)

    assert len(test_coilset.get_control_coils().x) == 16

    non_cc_diff = np.array([eq.coilset.psi(eq.grid.x, eq.grid.z) == test_v_psi])
    test_total = np.array(
        [test_coilset.psi(eq.grid.x, eq.grid.z) - (test_v_psi + test_p_psi)]
    )

    assert not non_cc_diff.all()
    assert test_total.all()


def test_get_psi_harmonic_amplitudes():
    path = get_bluemira_path("equilibria/test_data", subfolder="tests")
    eq = Equilibrium.from_eqdsk(Path(path, "SH_test_file.json"))

    test_colocation = collocation_points(
        n_points=18, plasma_boundary=eq.get_LCFS(), point_type="arc"
    )
    test_v_psi, _, _ = coils_outside_sphere_vacuum_psi(eq)
    test_sh_amps = get_psi_harmonic_amplitudes(test_v_psi, eq.grid, test_colocation, 1.2)

    sh_amps = np.array(
        [
            6.27021137e-03,
            1.13305430e-01,
            -7.10644651e-04,
            -1.21121072e-02,
            2.26606652e-04,
            8.62564002e-03,
            -1.09407173e-03,
            1.40751192e-02,
            -9.94184734e-04,
            1.18311187e-02,
            -7.55837162e-04,
            7.67252084e-03,
            -3.87118182e-04,
            2.66066774e-03,
            -1.87114805e-04,
            1.09257867e-03,
            -5.05761500e-05,
        ]
    )

    assert test_sh_amps == pytest.approx(sh_amps)


def test_spherical_harmonic_approximation():
    path = get_bluemira_path("equilibria/test_data", subfolder="tests")
    eq = Equilibrium.from_eqdsk(Path(path, "SH_test_file.json"))

    (
        test_sh_coilset,
        test_r_t,
        test_harmonic_amps,
        test_degree,
        test_fit_metric,
        _,
    ) = spherical_harmonic_approximation(
        eq,
        n_points=18,
        point_type="arc",
        acceptable_fit_metric=0.3,
    )

    sh_coilset_current = np.array(
        [
            7629.10582467,
            80572.92343772,
            72402.30872331,
            64228.69554408,
            21485.75482944,
            -16683.29269502,
            -67147.66998197,
            -169607.44792089,
            13184.10256721,
            12076.3164052,
            80572.92343772,
            72402.30872331,
            64228.69554408,
            21485.75482944,
            -13554.03543257,
            -67147.66998197,
            -169607.44792089,
            13184.10256721,
        ]
    )
    harmonic_amps = np.array([0.00627021, 0.12891703])

    assert test_sh_coilset.current == pytest.approx(sh_coilset_current)
    assert test_r_t == pytest.approx(1.3653400)
    assert test_harmonic_amps == pytest.approx(harmonic_amps)
    assert test_degree == 2
    assert test_fit_metric == pytest.approx(0.03, abs=0.005)
