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

"""
Tests for EU-DEMO build.
"""

import json
import os

import numpy as np
import pytest

import tests
from bluemira.base.components import Component
from bluemira.base.file import get_bluemira_root
from bluemira.base.logs import get_log_level, set_log_level
from bluemira.builders.EUDEMO.plasma import PlasmaComponent
from bluemira.builders.EUDEMO.reactor import EUDEMOReactor
from bluemira.builders.EUDEMO.tf_coils import TFCoilsComponent

PARAMS_DIR = os.path.join(get_bluemira_root(), "tests", "bluemira", "builders", "EUDEMO")


@pytest.mark.reactor
class TestEUDEMO:
    """
    Test the EU-DEMO design procedure.
    """

    def setup_class(self):
        params = {}
        with open(os.path.join(PARAMS_DIR, "template.json")) as fh:
            params = json.load(fh)

        with open(os.path.join(PARAMS_DIR, "params.json")) as fh:
            config = json.load(fh)
            for key, val in config.items():
                params[key]["value"] = val

        build_config = {}
        with open(os.path.join(PARAMS_DIR, "build_config.json")) as fh:
            build_config = json.load(fh)

        orig_log_level = get_log_level()
        set_log_level("DEBUG")
        try:
            self.reactor = EUDEMOReactor(params, build_config)
            self.component = self.reactor.run()
        finally:
            set_log_level(orig_log_level)

    def test_plasma_build(self):
        """
        Test the results of the plasma build.
        """
        plasma_builder = self.reactor.get_builder("Plasma")
        assert plasma_builder is not None
        assert plasma_builder.design_problem is not None

        plasma_component: PlasmaComponent = self.component.get_component("Plasma")
        assert plasma_component is not None
        assert plasma_component.equilibrium is not None

        reference_eq_dir = self.reactor.file_manager.reference_data_dirs["equilibria"]
        reference_eq_name = f"{self.reactor.params.Name.value}_eqref.json"
        reference_eq_path = os.path.join(reference_eq_dir, reference_eq_name)
        reference_eq_vals = {}
        with open(reference_eq_path, "r") as fh:
            reference_eq_vals: dict = json.load(fh)
        reference_eq_vals.pop("name")

        eq_dict = plasma_component.equilibrium.to_dict()
        bad_attrs = []
        attr: str
        for attr, ref_val in reference_eq_vals.items():
            if not np.allclose(eq_dict[attr], ref_val):
                bad_attrs.append(attr)

        assert len(bad_attrs) == 0, f"Attrs didn't match reference: {bad_attrs}"

    def test_tf_build(self):
        """
        Test the results of the TF build.
        """
        tf_builder = self.reactor.get_builder("TF Coils")
        assert tf_builder is not None
        assert tf_builder.design_problem is not None

        tf_component: TFCoilsComponent = self.component.get_component("TF Coils")
        assert tf_component is not None

        # Check field at origin
        field = tf_component.field(
            self.reactor.params.R_0.value, 0.0, self.reactor.params.z_0.value
        )
        assert field is not None
        print(field)
        assert field == pytest.approx([0, -5.0031, 0])

    @pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
    def test_plot_xz(self):
        """
        Display the results.
        """
        Component(
            "xz view",
            children=[
                self.component.get_component("Plasma").get_component("xz"),
                self.component.get_component("TF Coils").get_component("xz"),
            ],
        ).plot_2d()

    @pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
    def test_plot_xy(self):
        """
        Display the results.
        """
        Component(
            "xy view",
            children=[
                self.component.get_component("Plasma").get_component("xy"),
                self.component.get_component("TF Coils").get_component("xy"),
            ],
        ).plot_2d()

    @pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
    def test_show_cad(self):
        """
        Display the results.
        """
        Component(
            "xyz view",
            children=[
                self.component.get_component("Plasma").get_component("xyz"),
                self.component.get_component("TF Coils").get_component("xyz"),
            ],
        ).show_cad()
