# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from pathlib import Path

import gmsh
import numpy as np

from bluemira.base.components import Component, PhysicalComponent
from bluemira.geometry import tools
from bluemira.geometry.face import BluemiraFace, BluemiraWire
from bluemira.magnetostatics.biot_savart import Bz_coil_axis
from bluemira.magnetostatics.fem_utils import (
    Association,
    create_j_function,
    model_to_mesh,
)
from bluemira.magnetostatics.finite_element_2d import FemMagnetostatic2d
from bluemira.mesh import meshing

DATA_DIR = Path(Path(__file__).parent, "test_generated_data")


class TestGetNormal:
    def test_simple_thin_coil(self, tmp_path):
        """
        Compare the magnetic field on the axis of a coil with a very small cross-section
        calculated with the fem module and the analytic solution as limit of the
        Biot-Savart law.
        """
        r_enclo = 100
        lcar_enclo = 2
        lcar_axis = lcar_enclo / 10

        rc = 5
        drc = 0.01
        lcar_coil = 0.01

        poly_coil = tools.make_polygon(
            [
                [rc - drc, rc + drc, rc + drc, rc - drc],
                [-drc, -drc, +drc, +drc],
                [0, 0, 0, 0],
            ],
            closed=True,
            label="poly_enclo",
        )

        poly_coil.mesh_options = {"lcar": lcar_coil, "physical_group": "poly_coil"}
        coil = BluemiraFace(poly_coil)
        coil.mesh_options = {"lcar": lcar_coil, "physical_group": "coil"}

        poly_axis = tools.make_polygon([[0, 0, 0], [-r_enclo, 0, r_enclo], [0, 0, 0]])
        poly_axis.mesh_options = {"lcar": lcar_axis, "physical_group": "poly_axis"}

        poly_ext = tools.make_polygon(
            [
                [0, r_enclo, r_enclo, 0],
                [r_enclo, r_enclo, -r_enclo, -r_enclo],
                [0, 0, 0, 0],
            ],
            label="poly_ext",
        )
        poly_ext.mesh_options = {"lcar": lcar_enclo, "physical_group": "poly_ext"}

        poly_enclo = BluemiraWire([poly_axis, poly_ext], "poly_enclo")
        poly_enclo.mesh_options = {"lcar": lcar_enclo, "physical_group": "poly_enclo"}

        enclosure = BluemiraFace([poly_enclo, poly_coil])
        enclosure.mesh_options = {"lcar": lcar_enclo, "physical_group": "enclo"}

        c_universe = Component(name="universe")
        c_enclo = PhysicalComponent(name="enclosure", shape=enclosure, parent=c_universe)
        c_coil = PhysicalComponent(name="coil", shape=coil, parent=c_universe)

        meshing.Mesh(
            meshfile=[
                Path(tmp_path, p).as_posix() for p in ["Mesh.geo_unrolled", "Mesh.msh"]
            ]
        )(c_universe, dim=2)

        (mesh, ct, ft), _labels = model_to_mesh(gmsh.model, gdim=2)
        print(np.unique(ct.values))

        gmsh.write("Mesh.msh")
        gmsh.finalize()

        em_solver = FemMagnetostatic2d(3)
        em_solver.set_mesh(mesh, ct)

        current = 1e6
        coil_tag = 6
        j_functions = [Association(1, coil_tag, current)]
        jtot = create_j_function(mesh, ct, j_functions)

        em_solver.define_g(jtot)
        em_solver.solve()
        B = em_solver.calculate_b()

        z_points_axis = np.linspace(0, r_enclo, 200)
        r_points_axis = np.zeros(z_points_axis.shape)

        points = np.array([r_points_axis, z_points_axis, 0 * z_points_axis]).T
        Bz_axis, points = B._eval_new(points)
        Bz_axis = Bz_axis[:, 1]

        B_teo = np.array([Bz_coil_axis(rc, 0, z, current) for z in points[:, 1]])

        # I just set an absolute tolerance for the comparison (since the magnetic field
        # goes to zero, the comparison cannot be made on the basis of a relative
        # tolerance). An allclose comparison was out of discussion considering the
        # necessary accuracy.
        np.testing.assert_allclose(Bz_axis, B_teo, atol=2e-3)
