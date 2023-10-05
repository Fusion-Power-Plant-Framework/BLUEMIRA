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

import numpy as np
import pytest

from bluemira.geometry.parameterisations import PrincetonD
from eudemo.blanket.panelling._pivot_string import make_pivoted_string

DATA_DIR = Path(Path(__file__).parent, "data")


class TestMakePivotedString:
    def test_returns_points_matching_snapshot(self):
        """
        This tests that the function returns the same thing as the
        equivalent class method in BLUEPRINT.

        The data for this test was generated using the
        BLUEPRINT.geometry.stringgeom.String class from BLUEPRINT.

        The code used to generate the test data:

        .. code-block:: python

            from BLUEPRINT.geometry.stringgeom import String
            from bluemira.geometry.parameterisations import PrincetonD

            shape = PrincetonD().create_shape()
            points = shape.discretize()
            s = String(points, angle=20, dx_min=0.5, dx_max=2.5)

            np.save("panelling_ref_data.npy", s.new_points)

        Using bluemira 437a1c10, and BLUEPRINT e3fb8d1c.
        """
        boundary = PrincetonD(
            {"x1": {"value": 4}, "x2": {"value": 14}, "dz": {"value": 0}}
        ).create_shape()
        boundary_points = boundary.discretize().T

        new_points, _ = make_pivoted_string(
            boundary_points, max_angle=20, dx_min=0.5, dx_max=2.5
        )

        ref_data = np.load(Path(DATA_DIR, "panelling_ref_data.npy"))
        np.testing.assert_almost_equal(new_points, ref_data)

    @pytest.mark.parametrize("dx_max", [0, 0.5, 0.9999])
    def test_ValueError_given_dx_min_gt_dx_max(self, dx_max):
        boundary = PrincetonD(
            {"x1": {"value": 4}, "x2": {"value": 14}, "dz": {"value": 0}}
        ).create_shape()
        boundary_points = boundary.discretize().T

        with pytest.raises(ValueError):  # noqa: PT011
            make_pivoted_string(boundary_points, max_angle=20, dx_min=1, dx_max=dx_max)
