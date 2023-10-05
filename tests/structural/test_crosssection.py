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

from bluemira.base.constants import EPS
from bluemira.display import plot_2d
from bluemira.structural.crosssection import (
    AnalyticalCrossSection,
    CircularBeam,
    CircularHollowBeam,
    IBeam,
    RectangularBeam,
)
from bluemira.structural.error import StructuralError


class TestIbeam:
    def test_ibeam(self):
        # https://structx.com/Shape_Formulas_013.html
        i_beam = IBeam(1, 1, 0.25, 0.5)
        assert i_beam.area == pytest.approx(0.75, rel=0, abs=EPS)
        assert i_beam.i_yy == pytest.approx(0.078125, rel=0, abs=EPS)
        assert i_beam.i_zz == pytest.approx(0.046875, rel=0, abs=EPS)
        assert np.isclose(i_beam.j, 114583333333.3333e-12)

    def test_plot(self):
        i_beam = IBeam(1, 1, 0.25, 0.5)
        plot_2d(i_beam.geometry, show_points=True)
        plt.close()

    def test_errors(self):
        props = [
            [1, 1, 0.1, 0],
            [1, 1, 0, 0.1],
            [0, 1, 0.1, 0.1],
            [1, 0, 0.1, 0.1],
            [1, 1, 0.5, 0.1],
            [1, 1, 0.1, 1],
        ]
        for prop in props:
            with pytest.raises(StructuralError):
                IBeam(*prop)

    def test_rectangle(self):
        rect_beam = RectangularBeam(50, 50)
        assert rect_beam.area == 2500
        assert rect_beam.i_zz == pytest.approx(520833.3333333333, rel=0, abs=EPS)
        assert rect_beam.i_yy == pytest.approx(520833.3333333333, rel=0, abs=EPS)
        assert rect_beam.j == pytest.approx(880208.3333333334, rel=0, abs=EPS)

        rect_beam = RectangularBeam(10, 250)
        assert rect_beam.area == 2500
        assert np.isclose(rect_beam.i_zz, 20833.3333)
        assert np.isclose(rect_beam.i_yy, 13020833.33333)
        assert np.isclose(rect_beam.j, 81234.94973767549, rtol=1e-4)


class TestAnalytical:
    def test_rectangle(self):
        sq_beam = RectangularBeam(50, 40)
        custom_beam = AnalyticalCrossSection(sq_beam.geometry, j_opt_var=42.77)
        for k in ["area", "centroid", "i_yy", "i_zz", "i_zy", "qyy", "qzz", "ry", "rz"]:
            assert np.allclose(getattr(sq_beam, k), getattr(custom_beam, k))
        # J will not be so close...
        assert np.isclose(sq_beam.j, custom_beam.j, rtol=1e-4)

    def test_ibeam(self):
        i_beam = IBeam(1, 1, 0.25, 0.5)
        custom_beam = AnalyticalCrossSection(i_beam.geometry)
        for k in ["area", "centroid", "i_yy", "i_zz", "i_zy", "qyy", "qzz", "ry", "rz"]:
            assert np.allclose(getattr(i_beam, k), getattr(custom_beam, k))

    def test_circle(self):
        c_beam = CircularBeam(1)
        custom_beam = AnalyticalCrossSection(c_beam.geometry, n_discr=500)
        for k in ["area", "centroid", "i_yy", "i_zz", "i_zy", "qyy", "qzz", "ry", "rz"]:
            assert np.allclose(getattr(c_beam, k), getattr(custom_beam, k), rtol=1e-4)

    def test_hollow_circle(self):
        ch_beam = CircularHollowBeam(1, 1.2)
        custom_beam = AnalyticalCrossSection(ch_beam.geometry, n_discr=500)
        for k in ["area", "centroid", "i_yy", "i_zz", "i_zy", "qyy", "qzz", "ry", "rz"]:
            assert np.allclose(getattr(ch_beam, k), getattr(custom_beam, k), rtol=1e-4)


class TestRotation:
    def test_geometry_rotation(self):
        circle = CircularBeam(0.5)
        geometry = circle.geometry.copy()
        circle.rotate(2345.24)
        geometry.rotate(
            base=geometry.center_of_mass, direction=(1, 0, 0), degree=2345.24
        )
        assert np.allclose(circle.geometry.center_of_mass, geometry.center_of_mass)

    def test_circulars(self):
        circle = CircularBeam(0.5)
        i_zz_c = circle.i_zz
        i_yy_c = circle.i_yy
        i_zy_c = circle.i_zy

        h_circle = CircularHollowBeam(0.45, 0.5)
        i_zz_ch = h_circle.i_zz
        i_yy_ch = h_circle.i_yy
        i_zy_ch = h_circle.i_zy

        assert i_zy_c == pytest.approx(0.0, rel=0, abs=EPS)
        assert i_zy_ch == pytest.approx(0.0, rel=0, abs=EPS)

        alphas = [0.35, 14.54, 40.56, 90.6, 176, 270.5, 304.23423662, 361.0]
        for alpha in alphas:
            circle.rotate(alpha)
            h_circle.rotate(alpha)
            assert np.isclose(i_zz_c, circle.i_zz)
            assert np.isclose(i_yy_c, circle.i_yy)
            assert np.isclose(i_zy_c, circle.i_zy)
            assert np.isclose(i_zz_ch, h_circle.i_zz)
            assert np.isclose(i_yy_ch, h_circle.i_yy)
            assert np.isclose(i_zy_ch, h_circle.i_zy)

    def test_square(self):
        sq_beam = RectangularBeam(50, 50)
        i_zz_c = sq_beam.i_zz
        i_yy_c = sq_beam.i_yy
        i_zy_c = sq_beam.i_zy
        alphas = [0.35, 14.54, 40.56, 90.6, 176, 270.5, 304.23423662, 361.0]
        for alpha in alphas:
            sq_beam = RectangularBeam(50, 50)
            sq_beam.rotate(alpha)
            assert np.isclose(i_zz_c, sq_beam.i_zz)
            assert np.isclose(i_yy_c, sq_beam.i_yy)
            assert np.isclose(i_zy_c, sq_beam.i_zy)

    def test_rectangle(self):
        r_beam = RectangularBeam(4, 7)
        i_zz_c = r_beam.i_zz
        i_yy_c = r_beam.i_yy
        i_zy_c = r_beam.i_zy
        alphas = [0.35, 14.54, 40.56, 90.6, 176, 270.5, 304.23423662, 361.0]
        for alpha in alphas:
            r_beam = RectangularBeam(4, 7)
            r_beam.rotate(alpha)
            assert not np.isclose(i_zz_c, r_beam.i_zz)
            assert not np.isclose(i_yy_c, r_beam.i_yy)
            assert not np.isclose(i_zy_c, r_beam.i_zy)

        betas = [180, 360]
        for beta in betas:
            r_beam = RectangularBeam(4, 7)
            r_beam.rotate(beta)
            assert np.isclose(i_zz_c, r_beam.i_zz)
            assert np.isclose(i_yy_c, r_beam.i_yy)
            assert np.isclose(i_zy_c, r_beam.i_zy)

    def test_rectangle_values(self):
        """
        Cross-checked with: https://structx.com/Shape_Formulas_033.html
        """
        r_beam = RectangularBeam(0.508, 0.254)
        r_beam.rotate(np.rad2deg(4))
        assert np.isclose(10**12 * r_beam.i_zz, 1582893390.424)
