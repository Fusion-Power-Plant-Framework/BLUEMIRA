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

import pytest

from bluemira.geometry import tools
from bluemira.geometry.face import BluemiraFace
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf

DATA_DIR = Path(Path(__file__).parent, "test_generated_data")


class TestMeshing:
    @pytest.mark.parametrize(("lcar", "nodes_num"), [(0.1, 40), (0.25, 16), (0.5, 8)])
    def test_mesh_poly(self, lcar, nodes_num):
        poly = tools.make_polygon(
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], closed=True, label="poly"
        )

        poly.mesh_options = {"lcar": lcar, "physical_group": "poly"}

        surf = BluemiraFace(poly, label="surf")
        surf.mesh_options = {"physical_group": "coil"}

        meshfiles = [
            Path(DATA_DIR, p).as_posix() for p in ["Mesh.geo_unrolled", "Mesh.msh"]
        ]
        m = meshing.Mesh(meshfile=meshfiles)
        m(surf)

        msh_to_xdmf("Mesh.msh", dimensions=(0, 1), directory=DATA_DIR)

        _, boundaries, _, labels = import_mesh(
            "Mesh",
            directory=DATA_DIR,
            subdomains=True,
        )

        arr = boundaries.array()
        assert (arr == labels["poly"]).sum() == nodes_num

    @pytest.mark.parametrize(("lcar", "nodes_num"), [(0.1, 40), (0.25, 16), (0.5, 8)])
    def test_override_lcar_surf(self, lcar, nodes_num):
        poly = tools.make_polygon(
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], closed=True, label="poly"
        )

        poly.mesh_options = {"lcar": lcar, "physical_group": "poly"}

        surf = BluemiraFace(poly, label="surf")
        surf.mesh_options = {"lcar": lcar / 2, "physical_group": "coil"}

        meshfiles = [
            Path(DATA_DIR, p).as_posix() for p in ["Mesh.geo_unrolled", "Mesh.msh"]
        ]
        m = meshing.Mesh(meshfile=meshfiles)
        m(surf)

        msh_to_xdmf("Mesh.msh", dimensions=(0, 1), directory=DATA_DIR)

        _, boundaries, _, labels = import_mesh(
            "Mesh",
            directory=DATA_DIR,
            subdomains=True,
        )

        arr = boundaries.array()
        assert (arr == labels["poly"]).sum() == nodes_num * 2
