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

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bluemira.base.parameter_frame import make_parameter_frame
from bluemira.builders.tf_coils import (
    EquispacedSelector,
    ExtremaSelector,
    FixedSelector,
    MaximiseSelector,
    RippleConstrainedLengthGOP,
    RippleConstrainedLengthGOPParams,
)
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.geometry.tools import make_polygon


class TestRippleConstrainedLengthGOP:
    princeton = PrincetonD(
        {
            "x1": {"value": 4, "fixed": True},
            "dz": {"value": 0, "fixed": True},
            "x2": {"value": 15, "lower_bound": 13, "upper_bound": 20},
        }
    )
    lcfs = make_polygon({"x": [6, 12, 6], "z": [-4, 0, 4]}, closed=True)
    wp_xs = make_polygon(
        {"x": [3.5, 4.5, 4.5, 3.5], "y": [-0.5, -0.5, 0.5, 0.5], "z": 0}, closed=True
    )
    params = make_parameter_frame(
        {
            "n_TF": {"value": 16, "unit": "", "source": "test"},
            "R_0": {"value": 9, "unit": "m", "source": "test"},
            "z_0": {"value": 0, "unit": "m", "source": "test"},
            "B_0": {"value": 6, "unit": "T", "source": "test"},
            "TF_ripple_limit": {"value": 0.6, "unit": "%", "source": "test"},
        },
        RippleConstrainedLengthGOPParams,
    )

    @classmethod
    def teardown_method(cls):
        plt.show()
        plt.close("all")

    def test_default_setup(self):
        problem = RippleConstrainedLengthGOP(
            self.princeton,
            "SLSQP",
            {"max_eval": 100, "ftol_rel": 1e-6},
            {},
            self.params,
            self.wp_xs,
            self.lcfs,
            keep_out_zone=None,
            rip_con_tol=1e-3,
            n_rip_points=3,
        )
        problem.optimise()
        problem.plot()
        assert np.isclose(
            max(problem.ripple_values),
            self.params.TF_ripple_limit.value,
            rtol=0,
            atol=1e-3,
        )

    @pytest.mark.parametrize(
        "selector",
        [
            EquispacedSelector(3),
            EquispacedSelector(3, x_frac=0.5),
            EquispacedSelector(3, x_frac=1.0),
            ExtremaSelector(),
            FixedSelector(Coordinates({"x": [12, 6, 6], "z": [0, -4, 4]})),
            MaximiseSelector(),
        ],
    )
    def test_selector_setup(self, selector):
        problem = RippleConstrainedLengthGOP(
            self.princeton,
            "SLSQP",
            {"max_eval": 100, "ftol_rel": 1e-6},
            {},
            self.params,
            self.wp_xs,
            self.lcfs,
            keep_out_zone=None,
            rip_con_tol=1e-3,
            ripple_selector=selector,
        )
        problem.optimise()
        problem.plot()
        assert np.isclose(
            max(problem.ripple_values),
            self.params.TF_ripple_limit.value,
            rtol=0,
            atol=1e-3,
        )
