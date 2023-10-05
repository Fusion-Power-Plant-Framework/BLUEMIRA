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

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.testing import compare as mpl_compare

from bluemira.balance_of_plant.plotting import SuperSankey
from bluemira.base.file import get_bluemira_path
from bluemira.display.auto_config import plot_defaults


class TestSuperSankey:
    def teardown_method(self):
        plt.close()

    def test_sankey_ring(self):
        plot_defaults(True)

        scale = 0.001
        gap = 0.25
        l_standard = 0.0006 / scale  # standard arrow length
        l_medium = 0.001 / scale  # medium arrow length
        sankey = SuperSankey(scale=scale, gap=gap)
        sankey.add(
            "1",
            [1000, 500, -1500],
            orientations=[0, -1, 0],
            pathlengths=[l_medium, l_standard, l_medium],
        )
        sankey.add(
            "2",
            [1500, -1500],
            orientations=[0, -1],
            prior=0,
            connect=(2, 0),
            pathlengths=[l_medium, l_standard],
        )
        sankey.add(
            "3",
            [1500, -1000, -500],
            orientations=[0, 0, -1],
            prior=1,
            connect=(1, 0),
            pathlengths=[l_medium, l_standard, l_medium],
        )
        sankey.add(
            "4",
            [500, -500],
            orientations=[0, -1],
            prior=2,
            future=0,
            pathlengths=[l_medium, l_standard],
            connect=[(2, 0), (1, 1)],
        )
        sankey.finish()

        figure = plt.gcf()
        new_file = tempfile.NamedTemporaryFile()
        figure.savefig(new_file)
        plt.show()

        path = get_bluemira_path("balance_of_plant/test_data", subfolder="tests")
        reference_file = Path(path, "sankey_test.png")
        assert mpl_compare.compare_images(reference_file, new_file.name, 0.001) is None
