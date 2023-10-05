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

from bluemira.magnetostatics.circular_arc import CircularArcCurrentSource
from bluemira.magnetostatics.semianalytic_2d import semianalytic_Bx, semianalytic_Bz
from tests.magnetostatics.setup_methods import _plot_verification_test


class TestCircularArcCurrentSource:
    @classmethod
    def setup_class(cls):
        cls.xc, cls.zc = 4, 4
        cls.dx = 0.5
        cls.dz = 1.0
        cls.current = 1e6

        cls.arc = CircularArcCurrentSource(
            [0, 0, cls.zc],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            cls.dx,
            cls.dz,
            cls.xc,
            360,
            cls.current,
        )

    def test_2D_vs_3D_circular(self):
        nx, nz = 50, 60
        x = np.linspace(self.xc - 2, self.xc + 2, nx)
        z = np.linspace(self.zc - 2, self.zc + 2, nz)
        xx, zz = np.meshgrid(x, z, indexing="ij")

        Bx, _, Bz = self.arc.field(xx, np.zeros_like(xx), zz)
        Bp = np.hypot(Bx, Bz)

        cBx = semianalytic_Bx(self.xc, self.zc, xx, zz, self.dx, self.dz)
        cBz = semianalytic_Bz(self.xc, self.zc, xx, zz, self.dx, self.dz)
        Bx_coil = self.current * cBx
        Bz_coil = self.current * cBz
        Bp_coil = np.hypot(Bx_coil, Bz_coil)

        # Because this is a circular calculation, we expect them to be almost identical
        assert np.allclose(Bx_coil, Bx)
        assert np.allclose(Bz_coil, Bz)

        self.arc.plot()
        _plot_verification_test(
            self.xc,
            self.zc,
            self.dx,
            self.dz,
            xx,
            zz,
            Bx_coil,
            Bz_coil,
            Bp_coil,
            Bx,
            Bz,
            Bp,
        )

    def test_singularities(self):
        """
        Trigger singularities (ZeroDivisionErrors and such should not come up)
        """
        self.arc.field(self.arc.radius - self.arc.breadth, 0, 0)
        self.arc.field(self.arc.radius - self.arc.breadth, 0, self.arc.depth)
        self.arc.field(self.arc.radius - self.arc.breadth, 0, -self.arc.depth)
        self.arc.field(self.arc.radius + self.arc.breadth, 0, self.arc.depth)
        self.arc.field(self.arc.radius + self.arc.breadth, 0, -self.arc.depth)
        self.arc.field(self.arc.radius, 0, 1)
        self.arc.field(self.arc.radius, 0, 0)
        self.arc.field(0, 0, -1)
