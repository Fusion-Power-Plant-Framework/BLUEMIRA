# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import json
from pathlib import Path

import numpy as np

from bluemira.base.file import get_bluemira_path
from bluemira.geometry.tools import deserialise_shape

fp = get_bluemira_path("radiation_transport/neutronics/test_data", subfolder="tests")


class TestFullReactor:
    with open(Path(fp, "DEMO_external_boundary.json")) as j:
        exterior_boundary = deserialise_shape(json.load(j))
    with open(Path(fp, "DEMO_div_surface_boundary.json")) as j:
        div_surface = deserialise_shape(json.load(j))
    with open(Path(fp, "DEMO_vv.json")) as j:
        vacuum_vessel = deserialise_shape(json.load(j))
    panel_breakpoints = np.load(Path(fp, "DEMO_panel_points.npy"))

    def test_NeutronicsReactor(self):
        pass

    def test_placeholder(self):
        pass
