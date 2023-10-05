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

import numpy as np
import pytest

from bluemira.codes.error import FreeCADError
from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import PrincetonD, TripleArc
from bluemira.geometry.tools import (
    make_circle,
    make_polygon,
    offset_wire,
    revolve_shape,
    sweep_shape,
)


class TestSweep:
    def test_straight(self):
        path = make_polygon([[0, 0, 0], [0, 0, 1]])
        profile = make_polygon(
            [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], closed=True
        )

        sweep = sweep_shape(profile, path, solid=True)

        assert np.isclose(sweep.volume, 4.0)

    def test_semicircle(self):
        path = make_circle(start_angle=90, end_angle=-90)
        profile = make_polygon(
            [[0.5, 0, -0.5], [1.5, 0, -0.5], [1.5, 0, 0.5], [0.5, 0, 0.5]], closed=True
        )
        sweep = sweep_shape(profile, path, solid=True)
        assert sweep.is_valid()
        assert np.isclose(sweep.volume, np.pi)

    def test_circle(self):
        path = make_circle(start_angle=0, end_angle=360)
        profile = make_polygon(
            [[0.5, 0, -0.5], [1.5, 0, -0.5], [1.5, 0, 0.5], [0.5, 0, 0.5]], closed=True
        )
        sweep = sweep_shape(profile, path, solid=True)

        assert sweep.is_valid()
        assert np.isclose(sweep.volume, 2 * np.pi)

    def test_multiple_profiles(self):
        path = make_polygon([[0, 0, 0], [0, 0, 10]])
        profile_1 = make_polygon(
            [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], closed=True
        )
        profile_2 = make_circle(
            axis=[0, 0, 1], center=[0, 0, 5], radius=1, start_angle=0, end_angle=360
        )
        profile_3 = make_circle(
            axis=[0, 0, 1], center=[0, 0, 10], radius=2, start_angle=0, end_angle=360
        )
        sweep = sweep_shape([profile_1, profile_2, profile_3], path)

        assert sweep.is_valid()

    def test_princeton_d(self):
        x2 = 14
        dx = 0.5
        dy = 1
        pd = PrincetonD({"x2": {"value": x2}})
        path = pd.create_shape()
        profile = make_polygon(
            [[x2 - dx, -dy, 0], [x2 + dx, -dy, 0], [x2 + dx, dy, 0], [x2 - dx, dy, 0]],
            closed=True,
        )

        sweep = sweep_shape(profile, path)

        assert sweep.is_valid()
        assert np.isclose(sweep.volume, 84.1923, rtol=1e-4)

    def test_triple_arc(self):
        x1 = 4
        dx = 0.5
        dy = 1
        path = TripleArc().create_shape({"x1": {"value": x1}})
        profile = make_polygon(
            [[x1 - dx, -dy, 0], [x1 + dx, -dy, 0], [x1 + dx, dy, 0], [x1 - dx, dy, 0]],
            closed=True,
        )
        sweep = sweep_shape(profile, path)

        assert sweep.is_valid()
        assert np.isclose(sweep.volume, 95.61485, rtol=1e-4)

    def test_bad_profiles(self):
        path = make_polygon([[0, 0, 0], [0, 0, 10]])
        profile_1 = make_polygon(
            [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], closed=True
        )
        profile_2 = make_polygon(
            [[-1, -1, 10], [1, -1, 10], [1, 1, 10], [-1, 1, 10]], closed=False
        )
        with pytest.raises(FreeCADError):
            sweep = sweep_shape([profile_1, profile_2], path)

    def test_bad_path(self):
        path = make_polygon([[0, 0, 0], [0, 0, 10], [10, 0, 10]])
        profile = make_circle(
            axis=[0, 0, 1], center=[0, 0, 0], start_angle=0, end_angle=360
        )
        with pytest.raises(FreeCADError):
            sweep = sweep_shape(profile, path)

    def test_open_shell(self):
        path = make_polygon([[0, 0, 0], [0, 0, 10]])
        profile = make_polygon([[1, 0, 0], [1, 1, 0], [2, 1, 0]])
        sweep = sweep_shape(profile, path, solid=False)

        assert sweep.is_valid()


class TestRevolve:
    def test_semi_circle(self):
        wire = make_polygon(
            [[0.5, 0, -0.5], [1.5, 0, -0.5], [1.5, 0, 0.5], [0.5, 0, 0.5]], closed=True
        )
        shape = revolve_shape(wire, degree=180)
        assert np.isclose(shape.area, 4 * np.pi)
        assert shape.is_valid()
        face = BluemiraFace(wire)
        shape = revolve_shape(face, degree=180)
        assert np.isclose(shape.volume, np.pi)
        assert shape.is_valid()

    def test_circle(self):
        wire = make_polygon(
            [[0.5, 0, -0.5], [1.5, 0, -0.5], [1.5, 0, 0.5], [0.5, 0, 0.5]], closed=True
        )
        shape = revolve_shape(wire, degree=360)
        assert np.isclose(shape.area, 8 * np.pi)
        assert shape.is_valid()
        face = BluemiraFace(wire)
        shape = revolve_shape(face, degree=360)
        assert np.isclose(shape.volume, 2 * np.pi)
        assert shape.is_valid()

    def test_johner_semi(self):
        wire = JohnerLCFS().create_shape()
        face = BluemiraFace(wire)
        shape = revolve_shape(wire, degree=180)
        assert shape.is_valid()
        true_volume = np.pi * face.center_of_mass[0] * face.area
        shape = revolve_shape(face, degree=180)
        assert shape.is_valid()
        assert np.isclose(shape.volume, true_volume)

    def test_johner_full(self):
        wire = JohnerLCFS().create_shape()
        face = BluemiraFace(wire)
        shape = revolve_shape(wire, degree=360)
        assert shape.is_valid()
        true_volume = 2 * np.pi * face.center_of_mass[0] * face.area
        shape = revolve_shape(face, degree=360)
        assert shape.is_valid()
        assert np.isclose(shape.volume, true_volume)

    def test_revolve_hollow(self):
        x_c = 10
        d_xc = 1.0
        d_zc = 1.0
        inner = make_polygon(
            [
                [x_c - d_xc, 0, -d_zc],
                [x_c + d_xc, 0, -d_zc],
                [x_c + d_xc, 0, d_zc],
                [x_c - d_xc, 0, d_zc],
            ],
            closed=True,
        )
        outer = offset_wire(inner, 1.0, join="intersect")
        face = BluemiraFace([outer, inner])
        solid = revolve_shape(face, degree=360)

        true_volume = 2 * np.pi * x_c * (4**2 - 2**2)
        assert solid.is_valid()
        assert np.isclose(solid.volume, true_volume)
