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

from bluemira.base.file import get_bluemira_path
from bluemira.materials import MaterialCache
from bluemira.structural.material import make_structural_material

MATERIAL_DATA_PATH = get_bluemira_path("materials", subfolder="data")
MATERIAL_CACHE = MaterialCache()
MATERIAL_CACHE.load_from_file(Path(MATERIAL_DATA_PATH, "materials.json"))
MATERIAL_CACHE.load_from_file(Path(MATERIAL_DATA_PATH, "mixtures.json"))


class TestMaterialInterface:
    def setup_method(self):
        self.SS316 = MATERIAL_CACHE.get_material("SS316-LN")
        self.SW = MATERIAL_CACHE.get_material("Steel Water 60/40")

    def test_simple(self):
        str_mat = make_structural_material(self.SS316, 300)
        self.check_material_valid(str_mat)

    def test_homogenised(self):
        str_mat = make_structural_material(self.SW, 400)
        self.check_material_valid(str_mat)

    @staticmethod
    def check_material_valid(material):
        for attribute in ["E", "nu", "rho", "alpha", "sigma_y", "G"]:
            value = getattr(material, attribute)
            assert isinstance(value, float)
        assert material.description
