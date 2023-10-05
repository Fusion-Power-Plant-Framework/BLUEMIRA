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

import numpy as np
import pytest

from bluemira.base.error import DesignError
from bluemira.base.file import get_bluemira_path
from bluemira.equilibria import Equilibrium
from bluemira.equilibria.find import find_OX_points
from eudemo.ivc import WallSilhouetteDesigner

EQDATA = get_bluemira_path("equilibria/test_data", subfolder="tests")
DATA = str(Path(__file__).parent.parent / "test_data")

OPTIMISER_MODULE_REF = "bluemira.geometry.optimisation"
WALL_MODULE_REF = "eudemo.ivc.wall_silhouette_parameterisation"

CONFIG = {
    "param_class": f"{WALL_MODULE_REF}::WallPolySpline",
    "variables_map": {
        "x1": {  # ib radius
            "value": "r_fw_ib_in",
        },
        "x2": {  # ob radius
            "value": "r_fw_ob_in",
        },
    },
    "run_mode": "mock",
    "name": "First Wall",
    "problem_class": f"{OPTIMISER_MODULE_REF}::MinimiseLengthGOP",
}
PARAMS = {
    "R_0": {"value": 10.5, "unit": "m"},
    "kappa_95": {"value": 1.6, "unit": "m"},
    "r_fw_ib_in": {"value": 5.8, "unit": "m"},
    "r_fw_ob_in": {"value": 12.1, "unit": "m"},
    "A": {"value": 3.1, "unit": "m"},
    "tk_sol_ib": {"value": 0.225, "unit": "m"},
    "fw_psi_n": {"value": 1.07, "unit": "m"},
    "div_L2D_ib": {"value": 1.1, "unit": "m"},
    "div_L2D_ob": {"value": 1.45, "unit": "m"},
}


class TestWallSilhouetteDesigner:
    @classmethod
    def setup_class(cls):
        cls.eq = Equilibrium.from_eqdsk(Path(EQDATA, "eqref_OOB.json"))
        _, cls.x_points = find_OX_points(cls.eq.x, cls.eq.z, cls.eq.psi())

    def test_parameterisation_read(self):
        config = copy.deepcopy(CONFIG)
        config.update(
            {
                "run_mode": "read",
                "file_path": Path(DATA, "wall_polyspline.json").as_posix(),
            }
        )

        designer = WallSilhouetteDesigner(
            PARAMS, build_config=config, equilibrium=self.eq
        )
        param = designer.execute()

        assert param.variables["flat"].value == 0
        assert param.variables["x1"].value == PARAMS["r_fw_ib_in"]["value"]
        assert param.variables["x2"].value == PARAMS["r_fw_ob_in"]["value"]

    def test_read_no_file(self):
        config = copy.deepcopy(CONFIG)
        config.update({"run_mode": "read"})

        designer = WallSilhouetteDesigner(
            PARAMS, build_config=config, equilibrium=self.eq
        )

        with pytest.raises(ValueError):  # noqa: PT011
            designer.execute()

    def test_run_check_parameters(self):
        config = copy.deepcopy(CONFIG)
        config["run_mode"] = "run"
        config.update(
            {
                "problem_settings": {"n_koz_points": 101},
                "optimisation_settings": {
                    "algorithm_name": "COBYLA",
                    "parameters": {"initial_step": 1e-4},
                    "conditions": {"max_eval": 101},
                },
            }
        )
        designer = WallSilhouetteDesigner(
            PARAMS, build_config=config, equilibrium=self.eq
        )
        d_run = designer.execute()

        designer_mock = WallSilhouetteDesigner(
            PARAMS, build_config=CONFIG, equilibrium=self.eq
        )
        d_mock = designer_mock.execute()
        assert d_run.create_shape().length != d_mock.create_shape().length
        assert designer.problem_settings == config["problem_settings"]
        assert designer.opt_config == config["optimisation_settings"]
        assert (
            designer.algorithm_name == config["optimisation_settings"]["algorithm_name"]
        )
        assert designer.opt_parameters == config["optimisation_settings"]["parameters"]
        assert designer.opt_conditions == config["optimisation_settings"]["conditions"]

    def test_shape_is_closed(self):
        designer = WallSilhouetteDesigner(
            PARAMS, build_config=CONFIG, equilibrium=self.eq
        )

        assert designer.execute().create_shape().is_closed()

    def test_height_derived_from_params_given_PolySpline_mock_mode(self):
        params = copy.deepcopy(PARAMS)
        params.update(
            {
                "R_0": {"value": 10, "unit": "m"},
                "kappa_95": {"value": 2, "unit": "m"},
                "A": {"value": 2, "unit": "m"},
            }
        )
        config = copy.deepcopy(CONFIG)
        config.update({"param_class": f"{WALL_MODULE_REF}::WallPolySpline"})

        designer = WallSilhouetteDesigner(
            params, build_config=config, equilibrium=self.eq
        )
        bounding_box = designer.execute().create_shape().bounding_box

        # expected_height = 2*(R_0/A)*kappa_95 = 20
        assert np.isclose(bounding_box.z_max - bounding_box.z_min, 20.0)

    def test_width_matches_params_given_PrincetonD_mock_mode(self):
        vm = {
            "x1": {  # ib radius
                "value": "r_fw_ib_in",
            },
            "x2": {  # ob radius
                "value": "r_fw_ob_in",
            },
            "dz": -2,
        }
        config = copy.deepcopy(CONFIG)

        config.update(
            {"param_class": f"{WALL_MODULE_REF}::WallPrincetonD", "variables_map": vm}
        )

        designer = WallSilhouetteDesigner(
            PARAMS, build_config=config, equilibrium=self.eq
        )

        bounding_box = designer.execute().create_shape().bounding_box

        width = bounding_box.x_max - bounding_box.x_min
        assert width == pytest.approx(
            PARAMS["r_fw_ob_in"]["value"] - PARAMS["r_fw_ib_in"]["value"]
        )

    def test_DesignError_for_small_silhouette(self):
        config = copy.deepcopy(CONFIG)
        config.update({"param_class": f"{WALL_MODULE_REF}::WallPrincetonD"})

        designer = WallSilhouetteDesigner(
            PARAMS, build_config=config, equilibrium=self.eq
        )

        with pytest.raises(DesignError):
            designer.execute()
