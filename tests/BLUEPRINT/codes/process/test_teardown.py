# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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

import pytest

from BLUEPRINT.base.file import get_BP_path
from BLUEPRINT.systems.config import Configuration

from BLUEPRINT.codes.process.api import PROCESS_ENABLED
from BLUEPRINT.codes.process import teardown


@pytest.mark.skipif(PROCESS_ENABLED is not True, reason="PROCESS install required")
class TestMFileReader:
    fp = get_BP_path("codes/test_data", subfolder="tests")

    @classmethod
    def setup_class(cls):
        mapping = {
            p[-1]["PROCESS"].name: p[0]
            for p in Configuration.params
            if len(p) == 7 and "PROCESS" in p[-1]
        }
        cls.bmfile = teardown.BMFile(cls.fp, mapping)
        return cls

    def test_extraction(self):
        inp = [p[0] for p in Configuration.params if len(p) == 7 and "PROCESS" in p[-1]]
        out = self.bmfile.extract_outputs(inp)
        assert len(inp) == len(out)


if __name__ == "__main__":
    pytest.main([__file__])
