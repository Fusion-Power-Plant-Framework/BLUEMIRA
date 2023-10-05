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
from matplotlib import pyplot as plt

from bluemira.base.file import get_bluemira_path
from bluemira.magnetostatics.semianalytic_2d import semianalytic_Bx, semianalytic_Bz


class TestSemiAnalyticBxBz:
    def test_paper_results(self):
        # Check no effect of changing coil z
        self.semianalytic_results(0)
        self.semianalytic_results(4)
        self.semianalytic_results(-4)

    @staticmethod
    def semianalytic_results(zc):
        """
        Test case and data from:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6019053
        NOTE: Test case wrongly described in the above, and the correct description can
        be found here:
        https://www.tandfonline.com/doi/pdf/10.1163/156939310791958653?needAccess=true
        """
        xc = 0.0525
        dx, dz = 0.0025, 0.0025
        area = 4 * dx * dz
        current = 160000 * area
        bp_fe = (
            np.array(
                [
                    2.0564,
                    0.7776,
                    1.3356,
                    1.7800,
                    2.4892,
                    1.9368,
                    0.1270,
                    2.2498,
                    1.0841,
                    1.4177,
                    3.1451,
                    2.8208,
                    0.3800,
                    2.3199,
                ]
            )
            * 1e-4
        )

        bp_paper = (
            np.array(
                [
                    2.0567,
                    0.7774,
                    1.3353,
                    1.7802,
                    2.4893,
                    1.9368,
                    0.1267,
                    2.2499,
                    1.0842,
                    1.4174,
                    3.1455,
                    2.8207,
                    0.3803,
                    2.3202,
                ]
            )
            * 1e-4
        )

        x = [48, 60, 57, 48, 52, 52.5, 80, 52, 52.5, 57, 50, 50, 52.5, 55]
        z = [1, 1, 2, 3, 3, 4, 1, 2, 1, 0, 0, 2.5, 0, 2.5]
        x = np.array(x) * 1e-3
        z = np.array(z) * 1e-3 + zc
        bx_results = np.zeros_like(x)
        bz_results = np.zeros_like(z)
        for i, (xi, zi) in enumerate(zip(x, z)):
            bx = semianalytic_Bx(xc, zc, xi, zi, dx, dz)
            bz = semianalytic_Bz(xc, zc, xi, zi, dx, dz)
            bx_results[i] = current * bx
            bz_results[i] = current * bz

        # Test array treatment
        bx_array = semianalytic_Bx(xc, zc, x, z, dx, dz)
        bz_array = semianalytic_Bz(xc, zc, x, z, dx, dz)
        bx_array *= current
        bz_array *= current
        bp = np.sqrt(bx_results**2 + bz_results**2)

        assert np.allclose(bp_fe, bp, rtol=1e-2)
        assert np.allclose(bp_paper, bp, rtol=1e-4)
        assert np.all(bx_array == bx_results)
        assert np.all(bz_array == bz_results)

        fig, ax = plt.subplots()
        ax.plot(bp_fe, marker="s", label="FE", ms=20)
        ax.plot(bp_paper, marker="^", label="Paper", ms=20)
        ax.plot(bp, marker="X", label="New", ms=20)
        ax.legend()
        ax.set_xlabel("Point number")
        ax.set_ylabel("$B_{p}$ [T]")
        plt.show()
        plt.close(fig)

    def test_tough_Bz_integration_does_not_raise_error(self):
        """
        This is a challenging integration, where evaluation of the single full integrand
        is known to not converge / fail.
        """
        assert isinstance(
            semianalytic_Bz(
                4.389381920020457,
                9.39108180525438,
                4.3892405,
                9.39108181,
                0.0001414213562373095,
                0.0001414213562373095,
            ),
            float,
        )


class TestPoloidalFieldBenchmark:
    """
    Benchmarking the poloidal field distribution inside a circular coil with
    a rectangular cross-section from semi-analytic and Green's functions with
    a FE code (ERMES).
    http://tts.cimne.com/ermes/documentacionrotin/PrePrint-ERMES-Description.pdf

    It appears the FE model was sensitive to mesh size and the results were not
    fully converged, hence the peak discrepancy of 0.19 T. This benchmark should
    be repeated with a 2-D axisymmetric finite element model with extremely fine
    mesh size.
    """

    peak_discrepancy = 0.19  # [T]

    @classmethod
    def setup_class(cls):
        # cls.coil = Coil(4, 61, current=20e6, dx=0.5, dz=1.0)
        # cls.coil.mesh_coil(0.25)

        cls.path = get_bluemira_path("magnetostatics/test_data", subfolder="tests")

    @staticmethod
    def load_data(filename):
        with open(filename) as file:
            data = json.load(file)
            x = np.array(data["x"], dtype=float)
            z = np.array(data["z"], dtype=float)
            B = np.array(data["B"], dtype=float)

        return x, z, B

    def test_field_inside_coil_z_z(self):
        x, z, B = self.load_data(Path(self.path, "new_B_along_z-z.json"))

        fig, ax = plt.subplots()

        x_values = np.unique(x)
        for x_value in x_values:
            idx = np.nonzero(np.isclose(x, x_value))[0]
            z_x = z[idx]
            b_fe = B[idx]

            bx = 20e6 * semianalytic_Bx(4, 61, x[idx], z_x, 0.5, 1.0)
            bz = 20e6 * semianalytic_Bz(4, 61, x[idx], z_x, 0.5, 1.0)
            b_calc = np.hypot(bx, bz)

            assert max(abs(b_fe - b_calc)) < self.peak_discrepancy

            p = ax.plot(z_x, b_fe, label="ERMES x=" + str(x_value))
            ax.plot(
                z_x,
                b_calc,
                linestyle="--",
                color=p[0].get_color(),
                label="bluemira x=" + str(x_value),
            )
        ax.legend()
        ax.set_xlim([40, 80])
        ax.set_xlabel("z")
        ax.set_ylabel("$B_{p}$")
        plt.close(fig)

    def test_field_inside_coil_x_x(self):
        x, z, B = self.load_data(Path(self.path, "new_B_along_x-x.json"))

        fig, ax = plt.subplots()

        z_values = np.unique(z)[:5]  # Mirrored about coil zc-axis
        for z_value in z_values:
            idx = np.nonzero(np.isclose(z, z_value))[0]
            x_z = x[idx][1:]  # Can't do 0
            b_fe = B[idx][1:]

            bx = 20e6 * semianalytic_Bx(4, 61, x_z, z[idx][1:], 0.5, 1.0)
            bz = 20e6 * semianalytic_Bz(4, 61, x_z, z[idx][1:], 0.5, 1.0)
            b_calc = np.hypot(bx, bz)
            assert max(abs(b_fe - b_calc)) < self.peak_discrepancy

            p = ax.plot(x_z, b_fe, label="ERMES z=" + str(z_value))
            ax.plot(
                x_z,
                b_calc,
                linestyle="--",
                color=p[0].get_color(),
                label="bluemira z=" + str(z_value),
            )
        ax.legend()
        ax.set_xlim([0, 8])
        ax.set_xlabel("x")
        ax.set_ylabel("$B_{p}$")
        plt.close(fig)
