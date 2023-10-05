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

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bluemira.base.file import get_bluemira_path
from bluemira.geometry._pyclipper_offset import offset_clipper
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.error import GeometryError
from bluemira.geometry.tools import distance_to, make_polygon


class TestClipperOffset:
    options = ("square", "miter")
    # NOTE: "round" can be montrously slow..

    # fmt: off
    x = (-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -4)
    y = (0, -2, -4, -3, -4, -2, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 4, 3, 2, 1, 2, 2, 1, 0)
    # fmt: on

    @pytest.mark.parametrize("method", options)
    @pytest.mark.parametrize(
        ("x", "y", "delta"),
        [
            (x, y, 1.0),
            (x[::-1], y[::-1], 1.0),
            (x, y, -1.0),
            (x[::-1], y[::-1], -1.0),
        ],
    )
    def test_complex_polygon(self, x, y, delta, method):
        rng = np.random.default_rng()
        coordinates = Coordinates({"x": x, "y": y, "z": rng.random()})
        c = offset_clipper(coordinates, delta, method=method)

        fig, ax = plt.subplots()
        ax.plot(x, y, "k")
        ax.plot(c.x, c.y, "r", marker="o")
        ax.set_aspect("equal")
        plt.show()
        plt.close(fig)  # make sure we don't have lots of plots open

        distance = self._calculate_offset(coordinates, c)
        np.testing.assert_almost_equal(distance, abs(delta))

    @pytest.mark.parametrize("method", options)
    def test_complex_polygon_overoffset_raises_error(self, method):
        coordinates = Coordinates({"x": self.x, "y": self.y, "z": 0})
        with pytest.raises(GeometryError):
            offset_clipper(coordinates, -30, method=method)

    def test_blanket_offset(self):
        fp = get_bluemira_path("geometry/test_data", subfolder="tests")
        with open(Path(fp, "bb_offset_test.json"), "rb") as file:
            data = json.load(file)
        coordinates = Coordinates(data)
        offsets = []
        for m in ["miter", "square", "round"]:  # round very slow...
            offset_coordinates = offset_clipper(coordinates, 1.5, method=m)
            offsets.append(offset_coordinates)
            # Too damn slow!!
            # distance = self._calculate_offset(coordinates, offset_coordinates)
            # np.testing.assert_almost_equal(distance, 1.5)

        fig, ax = plt.subplots()
        ax.plot(coordinates.x, coordinates.z, color="k")
        colors = ["r", "g", "y"]
        for offset_coordinates, c in zip(offsets, colors):
            ax.plot(offset_coordinates.x, offset_coordinates.z, color=c)
        ax.set_aspect("equal")
        plt.show()
        plt.close(fig)

    def test_wrong_method(self):
        coordinates = Coordinates({"x": [0, 1, 2, 0], "y": [0, 1, -1, 0]})
        with pytest.raises(GeometryError):
            offset_clipper(coordinates, 1, method="fail")

    @pytest.mark.parametrize("method", options)
    def test_open_polygon_raises_error(self, method):
        coordinates = Coordinates({"x": [0, 1, 2], "y": [0, 1, -1]})
        with pytest.raises(GeometryError):
            offset_clipper(coordinates, 1, method=method)

    @pytest.mark.parametrize("method", options)
    def test_non_planar_polygon_raises_error(self, method):
        coordinates = Coordinates(
            {"x": [0, 1, 2, 0], "y": [0, 1, -1, 0], "z": [1, 0, 1, 0]}
        )
        with pytest.raises(GeometryError):
            offset_clipper(coordinates, 1, method=method)

    @staticmethod
    def _calculate_offset(coordinates, offset_coordinates):
        p1 = make_polygon(coordinates)
        p2 = make_polygon(offset_coordinates)
        return distance_to(p1, p2)[0]
