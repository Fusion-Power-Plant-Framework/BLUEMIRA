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
"""
Test first wall silhouette designer.
"""

import copy
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria import Equilibrium
from bluemira.equilibria.find import find_OX_points
from bluemira.geometry.parameterisations import PrincetonD, TripleArc
from bluemira.geometry.tools import make_circle, make_polygon
from eudemo.tf_coils import TFCoilBuilder, TFCoilDesigner

EQDATA = get_bluemira_path("equilibria/test_data", subfolder="tests")
DATA = str(Path(__file__).parent / "test_data")

OPTIMISER_MODULE_REF = "bluemira.builders.tf_coils"


class TestTFCoilDesigner:
    CONFIG: ClassVar = {
        "param_class": "TripleArc",
        "variables_map": {},
        "run_mode": "mock",
        "name": "First Wall",
        "problem_class": f"{OPTIMISER_MODULE_REF}::RippleConstrainedLengthGOP",
    }
    PARAMS: ClassVar = {
        "R_0": {"value": 10.5, "unit": "m"},
        "r_tf_current_ib": {"value": 1, "unit": "m"},
        "r_tf_in": {"value": 3.2, "unit": "m"},
        "tk_tf_wp": {"value": 0.5, "unit": "m"},
        "tk_tf_wp_y": {"value": 0.5, "unit": "m"},
        "tf_wp_width": {"value": 0.76, "unit": "m"},
        "tf_wp_depth": {"value": 1.05, "unit": "m"},
        "tk_tf_front_ib": {"value": 0.04, "unit": "m"},
        "g_ts_tf": {"value": 0.05, "unit": "m"},
        "TF_ripple_limit": {"value": 0.6, "unit": "%"},
        "z_0": {"value": 0.0, "unit": "m"},
        "B_0": {"value": 6, "unit": "T"},
        "n_TF": {"value": 12, "unit": ""},
        "tk_tf_ins": {"value": 0.08, "unit": "m"},
        "tk_tf_insgap": {"value": 0.1, "unit": "m"},
        "tk_tf_nose": {"value": 0.6, "unit": "m"},
    }

    @classmethod
    def setup_class(cls):
        cls.eq = Equilibrium.from_eqdsk(Path(EQDATA, "eqref_OOB.json"))
        _, cls.x_points = find_OX_points(cls.eq.x, cls.eq.z, cls.eq.psi())
        cls.lcfs = make_polygon(cls.eq.get_LCFS().xyz, closed=True)
        cls.vvts_koz = make_circle(10, center=(15, 0, 0), axis=(0.0, 1.0, 0.0))

    def test_parameterisation_read(self):
        config = copy.deepcopy(self.CONFIG)
        config.update(
            {
                "run_mode": "read",
                "file_path": Path(DATA, "tf_coils_TripleArc_18.json").as_posix(),
            }
        )

        designer = TFCoilDesigner(
            self.PARAMS,
            build_config=config,
            separatrix=self.lcfs,
            keep_out_zone=self.vvts_koz,
        )
        param, wp = designer.execute()

        assert np.isclose(param.variables["sl"].value, 5)
        assert np.isclose(param.variables["x1"].value, wp.center_of_mass[0])

    def test_read_no_file(self):
        config = copy.deepcopy(self.CONFIG)
        config.update({"run_mode": "read"})

        designer = TFCoilDesigner(
            self.PARAMS,
            build_config=config,
            separatrix=self.lcfs,
            keep_out_zone=self.vvts_koz,
        )

        with pytest.raises(ValueError):  # noqa: PT011
            designer.execute()

    def test_run_no_problem_class(self):
        config = copy.deepcopy(self.CONFIG)
        config.update({"run_mode": "run"})
        del config["problem_class"]

        designer = TFCoilDesigner(
            self.PARAMS,
            build_config=config,
            separatrix=self.lcfs,
            keep_out_zone=self.vvts_koz,
        )

        with pytest.raises(ValueError):  # noqa: PT011
            designer.execute()

    def test_run_check_parameters(self):
        config = copy.deepcopy(self.CONFIG)
        config["run_mode"] = "run"
        config.update(
            {
                "problem_settings": {"n_koz_points": 101},
                "optimisation_settings": {
                    "algorithm_name": "COBYLA",
                    "parameters": {"initial_step": 1e-4},
                    "conditions": {"max_eval": 25},
                },
            }
        )
        designer = TFCoilDesigner(
            self.PARAMS,
            build_config=config,
            separatrix=self.lcfs,
            keep_out_zone=self.vvts_koz,
        )
        d_run, _ = designer.execute()

        designer_mock = TFCoilDesigner(
            self.PARAMS,
            build_config=self.CONFIG,
            separatrix=self.lcfs,
            keep_out_zone=self.vvts_koz,
        )
        d_mock, _ = designer_mock.execute()
        assert d_run.create_shape().length != d_mock.create_shape().length
        assert designer.problem_settings == config["problem_settings"]
        assert designer.opt_config == config["optimisation_settings"]
        assert (
            designer.algorithm_name == config["optimisation_settings"]["algorithm_name"]
        )
        assert designer.opt_parameters == config["optimisation_settings"]["parameters"]
        assert designer.opt_conditions == config["optimisation_settings"]["conditions"]

    def test_shape_is_closed(self):
        designer = TFCoilDesigner(
            self.PARAMS,
            build_config=self.CONFIG,
            separatrix=self.lcfs,
            keep_out_zone=self.vvts_koz,
        )

        assert designer.execute()[0].create_shape().is_closed()


def centreline_setup():
    centrelines, wp_xs = [], []

    for cl in [TripleArc(), PrincetonD()]:
        centreline = cl.create_shape()
        tk_tf_wp = 0.4
        tk_tf_wp_y = 0.7

        x_min = centreline.bounding_box.x_min - (0.5 * tk_tf_wp)
        y_min = centreline.bounding_box.y_min - (0.5 * tk_tf_wp_y)

        wp_cross_section = make_polygon(
            [
                [x_min, y_min, 0],
                [x_min + tk_tf_wp, y_min, 0],
                [x_min + tk_tf_wp, y_min + tk_tf_wp_y, 0],
                [x_min, y_min + tk_tf_wp_y, 0],
            ],
            closed=True,
        )
        centrelines.append(centreline)
        wp_xs.append(wp_cross_section)

    return centrelines, wp_xs


class TestTFCoilBuilder:
    params: ClassVar = {
        "R_0": {"value": 9, "unit": "m"},
        "z_0": {"value": 0.0, "unit": "m"},
        "B_0": {"value": 6, "unit": "T"},
        "n_TF": {"value": 18, "unit": ""},
        "tf_wp_width": {"value": 0.7, "unit": "m"},
        "tf_wp_depth": {"value": 1.00, "unit": "m"},
        "tk_tf_front_ib": {"value": 0.04, "unit": "m"},
        "tk_tf_ins": {"value": 0.08, "unit": "m"},
        "tk_tf_insgap": {"value": 0.1, "unit": "m"},
        "tk_tf_nose": {"value": 0.6, "unit": "m"},
        "tk_tf_side": {"value": 0.1, "unit": "m"},
    }

    @pytest.mark.parametrize(("centreline", "wp_xs"), zip(*centreline_setup()))
    def test_components_and_segments(self, centreline, wp_xs):
        builder = TFCoilBuilder(self.params, {}, centreline, wp_xs)
        tf_coil = builder.build()

        assert tf_coil.get_component("xz")
        xy = tf_coil.get_component("xy")
        assert xy

        xyz = tf_coil.get_component("xyz")
        assert xyz
        xyz.show_cad()

        # Casing, Insulation, Winding pack
        assert len(xyz.leaves) == 3
        # inboard and outboard
        assert len(xy.leaves) == self.params["n_TF"]["value"] * 3 * 2

        ins = xyz.get_component(f"{TFCoilBuilder.INS} 1")
        wp = xyz.get_component(f"{TFCoilBuilder.WP} 1")
        insgap = self.params["tk_tf_ins"]["value"] + self.params["tk_tf_insgap"]["value"]
        assert np.isclose(
            wp.shape.bounding_box.x_min - ins.shape.bounding_box.x_min, insgap
        )
        assert np.isclose(
            ins.shape.bounding_box.y_max - wp.shape.bounding_box.y_max, insgap
        )

        ib_cas = xy.get_component(f"{TFCoilBuilder.CASING} 1").get_component("inboard 1")
        case_thick = (
            self.params["tf_wp_width"]["value"]
            + self.params["tk_tf_nose"]["value"]
            + self.params["tk_tf_front_ib"]["value"]
        )
        assert np.isclose(
            ib_cas.shape.bounding_box.x_max - ib_cas.shape.bounding_box.x_min,
            case_thick,
        )
