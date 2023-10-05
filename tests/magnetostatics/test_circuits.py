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
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.interpolate import UnivariateSpline, interp1d

from bluemira.base.file import get_bluemira_path
from bluemira.geometry._private_tools import offset
from bluemira.geometry.coordinates import Coordinates, vector_lengthnorm
from bluemira.geometry.parameterisations import PictureFrame, PrincetonD, TripleArc
from bluemira.geometry.tools import make_circle
from bluemira.magnetostatics.baseclass import SourceGroup
from bluemira.magnetostatics.circuits import (
    ArbitraryPlanarRectangularXSCircuit,
    HelmholtzCage,
)
from bluemira.magnetostatics.circular_arc import CircularArcCurrentSource
from bluemira.magnetostatics.trapezoidal_prism import TrapezoidalPrismCurrentSource


def test_analyticalsolvergrouper():
    xc, zc = 4, 4
    current = 1e6
    dx_coil, dz_coil = 0.5, 0.75

    # Build a corresponding arbitrary current loop
    circle = make_circle(center=[0, 0, zc], radius=xc).discretize(ndiscr=10)
    a = ArbitraryPlanarRectangularXSCircuit(circle, dx_coil, dz_coil, current)
    circle2 = make_circle(center=[0, 0, -zc], radius=xc).discretize(ndiscr=10)
    a2 = ArbitraryPlanarRectangularXSCircuit(circle2, dx_coil, dz_coil, current)
    solver = SourceGroup([a, a2])

    rng = np.random.default_rng()
    points = rng.uniform(low=-10, high=10, size=(10, 3))
    for point in points:
        field = solver.field(*point)  # random point :)
        field2 = a.field(*point) + a2.field(*point)
        assert np.all(field == field2)

    field = solver.field(*points.T)
    new_current = 2e6
    solver.set_current(new_current)
    field_new = solver.field(*points.T)

    assert np.allclose(field_new, new_current / current * field)


def test_sourcegroup_set_current():
    circle = make_circle(radius=10).discretize(ndiscr=50)
    dx_coil, dz_coil = 0.5, 0.75
    a = ArbitraryPlanarRectangularXSCircuit(circle, dx_coil, dz_coil, current=1)
    x, y, z = 4, 4, 4
    response = a.field(x, y, z)
    new_current = 1e6
    a.set_current(new_current)
    new_response = a.field(x, y, z)
    assert np.allclose(new_response, new_current * response)


def test_mixedsourcesolver():
    current = 1e6
    dx = 0.125
    dz = 0.25
    bar_1 = TrapezoidalPrismCurrentSource(
        [0, 0, 2], [-2, 0, 0], [0, 0, -1], [0, 1, 0], dx, dz, 0, 0, current
    )
    bar_2 = TrapezoidalPrismCurrentSource(
        [-2, 0, 0], [0, 0, -2], [1, 0, 0], [0, 1, 0], dx, dz, 0, 0, current
    )
    bar_3 = TrapezoidalPrismCurrentSource(
        [0, 0, -2], [2, 0, 0], [0, 0, 1], [0, 1, 0], dx, dz, 0, 0, current
    )
    bar_4 = TrapezoidalPrismCurrentSource(
        [2, 0, 0], [0, 0, 2], [-1, 0, 0], [0, 1, 0], dx, dz, 0, 0, current
    )

    arc_1 = CircularArcCurrentSource(
        [-1, 0, 1], [0, 0, 1], [-1, 0, 0], [0, 1, 0], dz, dx, 1, 90, current
    )
    arc_2 = CircularArcCurrentSource(
        [-1, 0, -1], [-1, 0, 0], [0, 0, -1], [0, 1, 0], dz, dx, 1, 90, current
    )
    arc_3 = CircularArcCurrentSource(
        [1, 0, -1], [0, 0, -1], [1, 0, 0], [0, 1, 0], dz, dx, 1, 90, current
    )
    arc_4 = CircularArcCurrentSource(
        [1, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], dz, dx, 1, 90, current
    )

    solver = SourceGroup([bar_1, bar_2, bar_3, bar_4, arc_1, arc_2, arc_3, arc_4])

    nx, nz = 20, 20
    nx2, nz2 = nx // 2, nz // 2
    x = np.linspace(-4, 4, nx)
    z = np.linspace(-4, 4, nz)
    xx, zz = np.meshgrid(x, z, indexing="ij")

    _, Bt, _ = solver.field(xx, np.zeros_like(xx), zz)

    # Test symmetry of the field in four quadranrs (rotation by matrix manipulations :))
    # Bottom-left (reference)
    bt_bl = Bt[:nx2, :nz2]
    # Bottom-right
    bt_br = Bt[nx2:, :nz2][::-1].T
    # Top-right
    bt_tr = Bt[nx2:, nz2:][::-1].T[::-1]
    # Top-left
    bt_tl = Bt[:nx2, nz2:].T[::-1]

    assert np.allclose(bt_bl, bt_br)
    assert np.allclose(bt_bl, bt_tr)
    assert np.allclose(bt_bl, bt_tl)

    solver.plot()
    fig, ax = plt.subplots()
    ax.contourf(xx, zz, Bt)
    ax.set_aspect("equal")
    plt.show()
    plt.close(fig)


class TestArbitraryPlanarXSCircuit:
    pd_inputs: ClassVar = {"x1": {"value": 4}, "x2": {"value": 16}, "dz": {"value": 0}}

    pf_inputs: ClassVar = {
        "x1": {"value": 5},
        "x2": {"value": 10},
        "z1": {"value": 10},
        "z2": {"value": -9},
        "ri": {"value": 0.4},
        "ro": {"value": 1},
    }
    ta_inputs: ClassVar = {
        "x1": {"value": 4},
        "dz": {"value": 0},
        "sl": {"value": 6.5},
        "f1": {"value": 3},
        "f2": {"value": 4},
        "a1": {"value": 20},
        "a2": {"value": 40},
    }

    p_inputs = (pd_inputs, ta_inputs, pf_inputs)
    clockwises = [False] * len(p_inputs) + [True] * len(p_inputs)
    p_inputs = p_inputs * 2  # noqa: PIE794
    parameterisations = tuple(
        [
            PrincetonD,
            TripleArc,
            PictureFrame,
        ]
        * 2
    )

    @pytest.mark.parametrize(
        ("parameterisation", "inputs", "clockwise"),
        zip(parameterisations, p_inputs, clockwises),
    )
    def test_circuits_are_continuous_and_chained(
        self, parameterisation, inputs, clockwise
    ):
        shape = parameterisation(inputs).create_shape()
        coords = shape.discretize(ndiscr=50, byedges=True)
        coords.set_ccw((0, -1, 0))
        if clockwise:
            coords.set_ccw((0, 1, 0))
        circuit = ArbitraryPlanarRectangularXSCircuit(
            coords,
            0.25,
            0.5,
            1.0,
        )
        open_circuit = ArbitraryPlanarRectangularXSCircuit(
            coords[:, :25].T, 0.25, 0.5, 1.0
        )
        n_chain = int(self._calc_daisychain(circuit))
        n_sources = len(circuit.sources) - 1
        assert n_chain == n_sources
        assert self._check_continuity(circuit.sources[-1], circuit.sources[0])
        assert self._calc_daisychain(open_circuit) == len(open_circuit.sources) - 1

    @pytest.mark.parametrize("clockwise", [False, True])
    def test_a_circuit_from_a_clockwise_circle_is_continuous(self, clockwise):
        shape = make_circle(5, (0, 9, 0), axis=(0, 0, 1))
        coords = shape.discretize(ndiscr=30, byedges=True)
        if clockwise:
            coords.set_ccw((0, 0, 1))
        else:
            coords.set_ccw((0, 0, -1))
        circuit = ArbitraryPlanarRectangularXSCircuit(
            coords,
            0.25,
            0.5,
            1.0,
        )
        assert self._calc_daisychain(circuit) == len(circuit.sources) - 1
        assert self._check_continuity(circuit.sources[-1], circuit.sources[0])

    def _calc_daisychain(self, circuit):
        chain = []
        for i, source_1 in enumerate(circuit.sources[:-1]):
            source_2 = circuit.sources[i + 1]
            daisy = self._check_continuity(source_1, source_2)
            chain.append(daisy)
        return sum(chain)

    @staticmethod
    def _check_continuity(source_1, source_2):
        s1_rect = source_1.points[1][:4]
        s2_rect = source_2.points[0][:4]
        return np.allclose(s1_rect, s2_rect)


class TestCariddiBenchmark:
    """
    This is a code comparison benchmark to some work from F. Villone (CREATE) in
    their report DEMO_D_2M97UY
    """

    @classmethod
    def setup_class(cls):
        root = get_bluemira_path("magnetostatics/test_data", subfolder="tests")
        width = 0.64
        depth = 1.15
        B_0 = 5.77
        R_0 = 8.87
        n_TF = 18

        with open(root + "/DEMO_2015_cariddi_ripple_xz.json") as f:
            data = json.load(f)
            cls.cariddi_ripple = data["z"]

        with open(root + "/DEMO_2015_ripple_xz.json") as f:
            data = json.load(f)
            cls.x_rip = data["x"]
            cls.z_rip = data["z"]

        with open(root + "/DEMO_2015_TF_xz.json") as f:
            data = json.load(f)
            x = data["x"]
            z = data["z"]
            coil_loop = Coordinates({"x": x, "y": 0, "z": z})
            coil_loop.close()
            coil_loop.set_ccw((0, 1, 0))
            linterp = np.linspace(0, 1, 300)
            ll = vector_lengthnorm(*coil_loop)
            x = interp1d(ll, coil_loop.x)(linterp)
            z = interp1d(ll, coil_loop.z)(linterp)
            x, z = offset(x, z, width / 2)

            coil_loop = Coordinates({"x": x, "y": 0, "z": z})

        # Smooth out graphically determined TF centreline...
        length_norm = vector_lengthnorm(x, z)
        l_interp = np.linspace(0, 1, 100)
        x = UnivariateSpline(length_norm, x, s=0.02)(l_interp)
        z = UnivariateSpline(length_norm, z, s=0.02)(l_interp)

        coil_loop = Coordinates({"x": x[10:-10], "y": 0, "z": z[10:-10]})
        coil_loop.close()
        cls.coil_loop = coil_loop

        circuit = ArbitraryPlanarRectangularXSCircuit(
            coil_loop, width / 2, depth / 2, current=1.0
        )

        # Set the current in the HelmholtzCage to generate the desired B_T,0 field
        cage = HelmholtzCage(circuit, n_TF)
        field_response = cage.field(R_0, 0, 0)[1]
        current = B_0 / field_response
        cage.set_current(current)
        cls.cage = cage

    def test_cariddi(self):
        ripple = self.cage.ripple(self.x_rip[1:19], np.zeros(18), self.z_rip[1:19])

        fig, (ax2, ax) = plt.subplots(1, 2)
        ax.scatter(list(range(1, 19)), self.cariddi_ripple, marker="o", label="CARIDDI")
        ax.scatter(list(range(1, 19)), ripple, marker="x", label="bluemira", zorder=20)
        ax.legend(loc="upper left")

        ax.set_ylabel("$\\delta_{\\phi}$ [%]")
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.set_xlabel("Point index")
        ax.set_xticks(np.arange(1, 19, 2))

        ax2.plot(self.coil_loop.x, self.coil_loop.z, color="b")
        ax2.plot(self.x_rip[1:19], self.z_rip[1:19], marker=".", color="r")
        plt.show()
        plt.close(fig)

        assert np.max(np.abs(ripple - self.cariddi_ripple)) < 0.04
