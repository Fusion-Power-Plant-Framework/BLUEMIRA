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
Tests for divertor builder classes
"""
import copy
import os

import numpy as np
import pytest

from bluemira.base.error import BuilderError
from bluemira.base.file import get_bluemira_path
from bluemira.builders.EUDEMO.first_wall import DivertorBuilder
from bluemira.builders.EUDEMO.first_wall.divertor import LegPosition
from bluemira.equilibria import Equilibrium
from bluemira.equilibria.find import find_OX_points
from bluemira.geometry.tools import make_polygon, signed_distance

DATA = get_bluemira_path("bluemira/equilibria/test_data", subfolder="tests")


def get_turning_point_idxs(z: np.ndarray):
    diff = np.diff(z)
    return np.argwhere(diff[1:] * diff[:-1] < 0)


class TestDivertorBuilder:

    _default_params = {
        "div_L2D_ib": (1.1, "Input"),
        "div_L2D_ob": (1.45, "Input"),
        "div_Ltarg": (0.5, "Input"),
        "div_open": (False, "Input"),
    }

    @classmethod
    def setup_class(cls):
        cls.eq = Equilibrium.from_eqdsk(os.path.join(DATA, "eqref_OOB.json"))
        cls.separatrix = make_polygon(cls.eq.get_separatrix().xyz.T)
        _, cls.x_points = find_OX_points(cls.eq.x, cls.eq.z, cls.eq.psi())

    def setup_method(self):
        self.params = copy.deepcopy(self._default_params)
        self.inner_start = np.array([5, 0, self.x_points[0]])
        self.outer_end = np.array([11, 0, self.x_points[0]])

    def test_no_BuilderError_on_init_given_valid_params(self):
        try:
            DivertorBuilder(
                self.params,
                {"name": "some_name"},
                self.eq,
                self.inner_start,
                self.outer_end,
            )
        except BuilderError:
            pytest.fail(str(BuilderError))

    @pytest.mark.parametrize("required_param", DivertorBuilder._required_params)
    def test_BuilderError_given_required_param_missing(self, required_param):
        self.params.pop(required_param)

        with pytest.raises(BuilderError):
            DivertorBuilder(
                self.params,
                {"name": "some_name"},
                self.eq,
                self.inner_start,
                self.outer_end,
            )

    def test_new_builder_sets_leg_lengths(self):
        self.params.update({"div_L2D_ib": 5, "div_L2D_ob": 10})

        builder = DivertorBuilder(
            self.params, {"name": "some_name"}, self.eq, self.inner_start, self.outer_end
        )

        assert builder.leg_length[LegPosition.INNER] == 5
        assert builder.leg_length[LegPosition.OUTER] == 10

    def test_targets_intersect_separatrix(self):
        builder = DivertorBuilder(
            self.params, {"name": "some_name"}, self.eq, self.inner_start, self.outer_end
        )

        divertor = builder()

        for leg in [LegPosition.INNER, LegPosition.OUTER]:
            target = divertor.get_component(f"target {leg}")
            assert signed_distance(target.shape, self.separatrix) == 0

    def test_div_Ltarg_sets_target_length(self):
        self.params.update({"div_Ltarg": 1.5})
        builder = DivertorBuilder(
            self.params, {"name": "some_name"}, self.eq, self.inner_start, self.outer_end
        )

        divertor = builder()

        for leg in [LegPosition.INNER, LegPosition.OUTER]:
            target = divertor.get_component(f"target {leg}")
            assert target.shape.length == 1.5

    def test_dome_added_to_divertor(self):
        builder = DivertorBuilder(
            self.params, {"name": "some_name"}, self.eq, self.inner_start, self.outer_end
        )

        divertor = builder()

        assert divertor.get_component("dome") is not None

    def test_dome_intersects_targets(self):
        builder = DivertorBuilder(
            self.params, {"name": "some_name"}, self.eq, self.inner_start, self.outer_end
        )

        divertor = builder()

        dome = divertor.get_component("dome")
        targets = [
            divertor.get_component(f"target {leg}")
            for leg in [LegPosition.INNER, LegPosition.OUTER]
        ]
        assert signed_distance(dome.shape, targets[0].shape) == 0
        assert signed_distance(dome.shape, targets[1].shape) == 0

    def test_dome_does_not_intersect_separatrix(self):
        builder = DivertorBuilder(
            self.params, {"name": "some_name"}, self.eq, self.inner_start, self.outer_end
        )

        divertor = builder()

        dome = divertor.get_component("dome")
        assert signed_distance(dome.shape, self.separatrix) < 0

    def test_dome_has_turning_point_below_x_point(self):
        # TODO(hsaunders): not sure about this test, what if the x-point
        # is at the top of the plasma?
        builder = DivertorBuilder(
            self.params, {"name": "some_name"}, self.eq, self.inner_start, self.outer_end
        )
        x_points, _ = self.eq.get_OX_points()

        divertor = builder()

        dome_coords = divertor.get_component("dome").shape.discretize()
        turning_points = get_turning_point_idxs(dome_coords[2, :])
        assert len(turning_points) == 1
        assert dome_coords[2, turning_points[0]] < x_points[0].z
