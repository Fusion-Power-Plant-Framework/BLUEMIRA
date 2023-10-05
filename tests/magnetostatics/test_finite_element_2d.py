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

import dolfin
import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.components import Component, PhysicalComponent
from bluemira.geometry import tools
from bluemira.geometry.face import BluemiraFace
from bluemira.magnetostatics.finite_element_2d import (
    Bz_coil_axis,
    FemMagnetostatic2d,
    ScalarSubFunc,
)
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf

DATA_DIR = Path(Path(__file__).parent, "test_generated_data")


class TestGetNormal:
    def test_simple_thin_coil(self):
        """
        Compare the magnetic field on the axis of a coil with a very small cross-section
        calculated with the fem module and the analytic solution as limit of the
        Biot-Savart law.
        """

        r_enclo = 100
        lcar_enclo = 0.5

        rc = 5
        drc = 0.01
        lcar_coil = 0.01

        poly_coil = tools.make_polygon(
            [
                [rc - drc, rc + drc, rc + drc, rc - drc],
                [0, 0, 0, 0],
                [-drc, -drc, +drc, +drc],
            ],
            closed=True,
            label="poly_enclo",
        )

        poly_coil.mesh_options = {"lcar": lcar_coil, "physical_group": "poly_coil"}
        coil = BluemiraFace(poly_coil)
        coil.mesh_options = {"lcar": lcar_coil, "physical_group": "coil"}

        poly_enclo = tools.make_polygon(
            [
                [0, r_enclo, r_enclo, 0],
                [0, 0, 0, 0],
                [-r_enclo, -r_enclo, r_enclo, r_enclo],
            ],
            closed=True,
            label="poly_enclo",
        )

        poly_enclo.mesh_options = {"lcar": lcar_enclo, "physical_group": "poly_enclo"}
        enclosure = BluemiraFace([poly_enclo, poly_coil])
        enclosure.mesh_options = {"lcar": lcar_enclo, "physical_group": "enclo"}

        c_universe = Component(name="universe")
        c_enclo = PhysicalComponent(name="enclosure", shape=enclosure, parent=c_universe)
        c_coil = PhysicalComponent(name="coil", shape=coil, parent=c_universe)

        meshfiles = [
            Path(DATA_DIR, p).as_posix() for p in ["Mesh.geo_unrolled", "Mesh.msh"]
        ]
        m = meshing.Mesh(meshfile=meshfiles)
        m(c_universe, dim=2)

        msh_to_xdmf("Mesh.msh", dimensions=(0, 2), directory=DATA_DIR)

        mesh, boundaries, subdomains, labels = import_mesh(
            "Mesh",
            directory=DATA_DIR,
            subdomains=True,
        )

        dolfin.plot(mesh)
        plt.show()

        em_solver = FemMagnetostatic2d(3)
        em_solver.set_mesh(mesh, boundaries)

        current = 1e6
        jc = current / coil.area
        markers = [labels["coil"]]
        functions = [jc]
        jtot = ScalarSubFunc(functions, markers, subdomains)

        em_solver.define_g(jtot)
        em_solver.solve()
        em_solver.calculate_b()

        z_points_axis = np.linspace(0, r_enclo, 200)
        r_points_axis = np.zeros(z_points_axis.shape)

        Bz_axis = np.array(
            [em_solver.B(x) for x in np.array([r_points_axis, z_points_axis]).T]
        ).T[1]

        B_teo = np.array([Bz_coil_axis(rc, 0, z, current) for z in z_points_axis])

        # I just set an absolute tolerance for the comparison (since the magnetic field
        # goes to zero, the comparison cannot be made on the basis of a relative
        # tolerance). An allclose comparison was out of discussion considering the
        # necessary accuracy.
        np.testing.assert_allclose(Bz_axis, B_teo, atol=1e-4)
