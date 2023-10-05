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

from bluemira.equilibria import Equilibrium
from bluemira.equilibria.coils import Coil, CoilSet
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.optimisation.constraints import (
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
)
from bluemira.equilibria.optimisation.problem import TikhonovCurrentCOP
from bluemira.equilibria.profiles import CustomProfile
from bluemira.equilibria.solve import PicardIterator


def coilset_setup(materials=False):
    # EU DEMO 2015
    x = [5.4, 14.0, 17.0, 17.01, 14.4, 7.0, 2.9, 2.9, 2.9, 2.9, 2.9]
    z = [8.82, 7.0, 2.5, -2.5, -8.4, -10.45, 6.6574, 3.7503, -0.6105, -4.9713, -7.8784]
    dx = [0.6, 0.4, 0.5, 0.5, 0.7, 1.0, 0.4, 0.4, 0.4, 0.4, 0.4]
    dz = [0.6, 0.4, 0.5, 0.5, 0.7, 1.0, 1.4036, 1.4036, 2.85715, 1.4036, 1.4036]
    names = [f"PF_{i + 1}" for i in range(6)] + [f"CS_{i + 1}" for i in range(5)]

    coils = []
    for name, xc, zc, dxc, dzc in zip(names, x, z, dx, dz):
        ctype = name[:2]
        coil = Coil(xc, zc, dx=dxc, dz=dzc, name=name, ctype=ctype)
        coils.append(coil)
    coilset = CoilSet(*coils)

    if materials:
        coilset.assign_material("PF", j_max=12.5e6, b_max=12.5)
        coilset.assign_material("CS", j_max=12.5e6, b_max=12.5)
        coilset.fix_sizes()

    return coilset


def test_isoflux_constrained_tikhonov_current_optimisation():
    coilset = coilset_setup()
    grid = Grid(4.5, 14, -9, 9, 65, 65)
    profiles = CustomProfile(
        np.linspace(1, 0), -np.linspace(1, 0), R_0=9, B_0=6, I_p=10e6
    )
    eq = Equilibrium(coilset, grid, profiles)

    isoflux = IsofluxConstraint(
        x=np.array([6, 8, 12, 6]),
        z=np.array([0, 7, 0, -8]),
        ref_x=6,
        ref_z=0,
        constraint_value=0,
    )

    x_point = FieldNullConstraint(8, -8)
    targets = MagneticConstraintSet([isoflux, x_point])
    opt_problem = TikhonovCurrentCOP(eq.coilset, eq, targets, gamma=1e-8)
    program = PicardIterator(eq, opt_problem, relaxation=0.1, plot=True)
    program()
    np.testing.assert_almost_equal(
        eq.coilset.current,
        [
            505197.01488516,
            -1092264.98586219,
            -1851426.77731549,
            -1585689.72567114,
            -2948303.9525497,
            2311264.55064226,
            -905378.73231966,
            -3559171.95768416,
            -6874525.64304157,
            3426784.33441372,
            7551499.69198225,
        ],
        decimal=5,
    )
