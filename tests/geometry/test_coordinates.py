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

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bluemira.base.constants import EPS
from bluemira.base.file import get_bluemira_path
from bluemira.display.plotter import plot_coordinates
from bluemira.geometry.coordinates import (
    Coordinates,
    check_linesegment,
    coords_plane_intersect,
    get_centroid,
    get_intersect,
    get_normal_vector,
    get_perimeter,
    in_polygon,
    join_intersect,
    on_polygon,
    polygon_in_polygon,
    rotation_matrix,
    vector_lengthnorm,
)
from bluemira.geometry.error import CoordinatesError
from bluemira.geometry.plane import BluemiraPlane

TEST_PATH = get_bluemira_path("geometry/test_data", subfolder="tests")


def trace_torus_orbit(r_1, r_2, n_r_2_turns, n_points):
    """
    Trace the orbit of a particle travelling around an a torus.

    Parameters
    ----------
    r_1: float
      The radius of the x-y circle
    r_2: float
      The radius of the particle orbit around the x-y circle
    n_r_2_turns: float
      The number of orbits around the centreline when making one full turn of the x-y
      circle
    n_points: int
      The number of points to produce

    Returns
    -------
    xyz_array: np.ndarray
      Array of shape (3, n_points)
    """
    phi = np.linspace(0, 2 * np.pi, num=n_points)  # Major angle
    theta = np.linspace(0, 2 * np.pi * n_r_2_turns, num=n_points)  # Minor angle
    x = (r_1 + r_2 * np.cos(theta)) * np.cos(phi)
    y = (r_1 + r_2 * np.cos(theta)) * np.sin(phi)
    z = r_2 * np.sin(theta)

    direction = 1 if n_r_2_turns > 0.0 else -1
    return np.array([x[::direction], y[::direction], z[::direction]])


class TestPerimeter:
    def test_simple(self):
        # 2 x 2 square
        x = [0, 2, 2, 0, 0]
        y = [0, 0, 2, 2, 0]
        assert get_perimeter(x, y) == pytest.approx(8.0, rel=0, abs=EPS)


class TestGetNormal:
    def test_simple(self):
        x = np.array([0, 2, 2, 0, 0])
        z = np.array([0, 0, 2, 2, 0])
        y = np.zeros(5)
        n_hat = get_normal_vector(x, y, z)
        assert np.allclose(np.abs(n_hat), np.array([0, 1, 0]))

    def test_edge(self):
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        z = np.array([1, 2, 4])
        n_hat = get_normal_vector(x, y, z)
        assert np.allclose(n_hat, 0.5 * np.array([np.sqrt(2), -np.sqrt(2), 0]))

    def test_error(self):
        fails = [
            [[0, 1], [0, 1], [0, 1]],
            [[0, 1, 2], [0, 1, 2], [0, 1]],
            [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        ]
        for fail in fails:
            with pytest.raises(CoordinatesError):
                get_normal_vector(
                    np.array(fail[0]), np.array(fail[1]), np.array(fail[2])
                )


class TestGetCentroid:
    def test_simple(self):
        x = [0, 2, 2, 0, 0]
        y = [0, 0, 2, 2, 0]
        xc, yc = get_centroid(x, y)
        assert np.isclose(xc, 1)
        assert np.isclose(yc, 1)
        xc, yc = get_centroid(np.array(x[::-1]), np.array(y[::-1]))
        assert np.isclose(xc, 1)
        assert np.isclose(yc, 1)

    def test_negative(self):
        x = [0, -2, -2, 0, 0]
        y = [0, 0, -2, -2, 0]
        xc, yc = get_centroid(x, y)
        assert np.isclose(xc, -1)
        assert np.isclose(yc, -1)
        xc, yc = get_centroid(np.array(x[::-1]), np.array(y[::-1]))
        assert np.isclose(xc, -1)
        assert np.isclose(yc, -1)


class TestCoordinates:
    rng = np.random.default_rng()

    def test_array_init(self):
        xyz = np.array([[0, 1, 2, 3], [0, 0, 0, 0], [0, 1, 2, 3]])
        c1 = Coordinates(xyz)
        c2 = Coordinates(xyz.T)
        assert c1 == c2
        assert np.allclose(c1.x, xyz[0])
        assert np.allclose(c1.y, xyz[1])
        assert np.allclose(c1.z, xyz[2])
        assert np.allclose(c1[0], xyz[0])
        assert np.allclose(c1[1], xyz[1])
        assert np.allclose(c1[2], xyz[2])

    def test_bad_array_init(self):
        a = self.rng.random((4, 4))
        with pytest.raises(CoordinatesError):
            Coordinates(a)

        a = np.array([[1, 2, 3, 4, 5], [1, 2, 3], [0, 0, 0, 0]], dtype=object)
        with pytest.raises(CoordinatesError):
            Coordinates(a)

    def test_obj_but_good_lengths(self):
        xyz = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [0, 0, 0, 0, 0]], dtype=object)
        c = Coordinates(xyz)
        assert np.allclose(c.x, [1, 2, 3, 4, 5])
        assert np.allclose(c.y, [1, 2, 3, 4, 5])
        assert np.allclose(c.z, [0, 0, 0, 0, 0])

    def test_dict_init(self):
        xyz = {"x": [0, 1, 2, 3], "y": 1, "z": 0}
        c = Coordinates(xyz)
        assert np.allclose(c.x, np.array([0, 1, 2, 3]))
        assert np.allclose(c.y, np.array([1, 1, 1, 1]))
        assert np.allclose(c.z, np.array([0, 0, 0, 0]))

        xyz = {"x": [0, 0, 0, 0], "y": [1, 2, 2, 1], "z": [-1, -1, 1, 1]}
        c = Coordinates(xyz)
        assert np.allclose(c.x, xyz["x"])
        assert np.allclose(c.y, xyz["y"])
        assert np.allclose(c.z, xyz["z"])

    def test_ragged_dict_init(self):
        xyz = {"x": [0, 0, 0, 0], "y": [1, 2, 1], "z": [-1, -1, 1, 1]}
        with pytest.raises(CoordinatesError):
            Coordinates(xyz)

    def test_iterable_init(self):
        xyz = [[0, 0, 0], (1, 1, 1), [2, 2, 2], [3, 3, 3]]
        c = Coordinates(xyz)

        assert np.allclose(c.x, list(range(4)))
        assert np.allclose(c.y, list(range(4)))
        assert np.allclose(c.z, list(range(4)))

        xyz = ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], (0, 1, 2, 3, 4))
        c = Coordinates(xyz)

        assert np.allclose(c.x, list(range(5)))
        assert np.allclose(c.y, list(range(5)))
        assert np.allclose(c.z, list(range(5)))

    def test_bad_iterable_init(self):
        xyz = [[0, 0, 0], (1, 1), [2, 2, 2], [3, 3, 3]]

        with pytest.raises(CoordinatesError):
            Coordinates(xyz)

        xyz = ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], (0, 1, 2, 3))
        with pytest.raises(CoordinatesError):
            Coordinates(xyz)

    def test_indexing(self):
        xyz = np.array([[0, 1, 2, 3], [0, 0, 0, 0], [0, 1, 2, 3]])
        c1 = Coordinates(xyz)
        assert np.all(xyz[0] == c1[0])
        assert np.all(xyz[1] == c1[1])
        assert np.all(xyz[2] == c1[2])

        assert np.allclose([3, 0, 3], c1[:, -1])

    def test_unpacking(self):
        xyz = np.array([[0, 1, 2, 3], [0, 0, 0, 0], [0, 1, 2, 3]])
        x1, y1, z1 = xyz
        c1 = Coordinates(xyz)
        x2, y2, z2 = c1
        assert np.all(x1 == x2)
        assert np.all(y1 == y2)
        assert np.all(z1 == z2)

    def test_reverse(self):
        xyz = np.array([[0, 1, 2, 3], [0, 0, 0, 0], [0, 1, 2, 3]])
        c = Coordinates(xyz)
        c.reverse()
        assert np.allclose(c.x, xyz[0][::-1])
        assert np.allclose(c.y, xyz[1][::-1])
        assert np.allclose(c.z, xyz[2][::-1])

    def test_translate(self):
        a = self.rng.random((3, 123))
        v = self.rng.random(3)
        c = Coordinates(a)
        c.translate(v)

        assert np.allclose(a + v.reshape(3, 1), c)

    def test_rotate(self):
        a = self.rng.random((3, 123))
        v = self.rng.random(3)
        base = self.rng.random(3)
        direction = self.rng.random(3)
        degree = 360 * self.rng.random(1)
        c = Coordinates(a)
        c2 = deepcopy(c)
        c.rotate(base, direction, degree)

        assert np.isclose(c.length, c2.length)

    def test_rotate_90(self):
        c = Coordinates(
            np.array(
                [
                    [2, 3, 3, 2, 2],
                    [0, 0, 0, 0, 0],
                    [1, 1, 2, 2, 1],
                ]
            )
        )
        v1 = c.normal_vector
        assert np.allclose(v1, [0, 1, 0])
        assert c.check_ccw()

        c.rotate(degree=90)
        v2 = c.normal_vector
        assert np.allclose(abs(v2), [1, 0, 0])
        assert c.check_ccw([-1, 0, 0])

        c.rotate(degree=90)
        v3 = c.normal_vector
        assert np.allclose(abs(v3), [0, 1, 0])
        assert c.check_ccw([0, -1, 0])

        c.rotate(degree=90)
        v4 = c.normal_vector
        assert np.allclose(abs(v4), [1, 0, 0])
        assert c.check_ccw([1, 0, 0])

        c.rotate(degree=90)
        v5 = c.normal_vector
        assert np.allclose(abs(v5), [0, 1, 0])
        assert c.check_ccw([0, 1, 0])

    def test_is_planar(self):
        c = Coordinates(
            np.array(
                [
                    [2, 3, 3, 2, 2],
                    [0, 0, 0, 0, 0],
                    [1, 1, 2, 2, 1],
                ]
            )
        )
        v = self.rng.random(3)
        base = self.rng.random(3)
        direction = self.rng.random(3)
        degree = 360 * self.rng.random(1)
        c.rotate(base, direction, degree)
        assert c.is_planar

    def test_circle_ccw(self):
        radius = 5
        theta = np.linspace(0, 2 * np.pi, 100)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.zeros(100)
        c = Coordinates([x, y, z])
        assert c.check_ccw([0, 0, 1])
        c.reverse()
        assert not c.check_ccw(axis=[0, 0, 1])

    def test_circle_xz(self):
        radius = 5
        theta = np.linspace(0, 2 * np.pi, 100)
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        y = np.zeros(100)
        c = Coordinates([x, y, z])
        assert c.check_ccw(axis=[0, 1, 0])
        c.reverse()
        assert not c.check_ccw(axis=[0, 1, 0])

    def test_circle_xz_translated(self):
        radius = 5
        theta = np.linspace(0, 2 * np.pi, 100)
        x = radius * np.cos(theta) + 10
        z = radius * np.sin(theta) + 10
        y = np.zeros(100)
        c = Coordinates([x, y, z])
        assert c.check_ccw(axis=[0, 1, 0])
        c.reverse()
        assert not c.check_ccw(axis=[0, 1, 0])

    @pytest.mark.xfail
    def test_complicated(self):
        xyz = trace_torus_orbit(5, 1, 10, 999)
        c = Coordinates(xyz)

        assert not c.is_planar
        assert c.check_ccw([0, 0, 1])

        xyz = trace_torus_orbit(50, 1, -10, 1000)
        c = Coordinates(xyz)
        assert not c.is_planar
        assert not c.check_ccw([0, 0, 1])
        c.set_ccw([0, 0, 1])
        assert c.check_ccw()

    @pytest.mark.parametrize("index", [0, 1, 2, 3, 6, -1])
    def test_insert(self, index):
        c = Coordinates({"x": [0, 1, 2, 3, 4, 5, 6], "z": [0, 1, 2, 3, 4, 5, 6]})
        c.insert([10, 10, 10], index=index)
        assert c.x[index] == 10
        assert c.y[index] == 10
        assert c.z[index] == 10
        assert len(c) == 8

    def test_insert_overshoot_index(self):
        c = Coordinates({"x": [0, 1, 2, 3, 4, 5, 6], "z": [0, 1, 2, 3, 4, 5, 6]})
        c.insert([10, 10, 10], index=10)
        assert c.x[7] == 10
        assert c.y[7] == 10
        assert c.z[7] == 10
        assert len(c) == 8


class TestShortCoordinates:
    point = Coordinates({"x": 0, "y": 0, "z": 0})
    line = Coordinates({"x": [0, 1], "y": [0, 1], "z": [0, 1]})

    def test_point_instantiation(self):
        point = Coordinates({"x": [1], "y": [1], "z": [1]})
        point2 = Coordinates([[1], [1], [1]])
        point3 = Coordinates({"x": 1, "y": 1, "z": 1})
        point4 = Coordinates([1, 1, 1])
        assert point == point2 == point3 == point4

    def test_line_instantiation(self):
        line = Coordinates({"x": [0, 1], "y": [0, 1], "z": [0, 1]})
        line2 = Coordinates([[0, 1], [0, 1], [0, 1]])
        assert line == line2

    @pytest.mark.parametrize(("c", "length"), [(point, 0.0), (line, np.sqrt(3))])
    def test_length(self, c, length):
        measured = c.length
        np.testing.assert_almost_equal(measured, length)

    @pytest.mark.parametrize("c", [point, line])
    def test_normal_vector(self, c, caplog):
        assert c.normal_vector is None
        assert len(caplog.messages) == 1
        assert "Cannot set planar properties" in caplog.messages[0]

    @pytest.mark.parametrize("c", [point, line])
    def test_is_planar(self, c):
        assert not c.is_planar

    @pytest.mark.parametrize(
        ("c", "com"), [(point, np.array([0, 0, 0])), (line, np.array([0.5, 0.5, 0.5]))]
    )
    def test_center_of_mass(self, c, com):
        measured = c.center_of_mass
        np.testing.assert_allclose(measured, com)

    @pytest.mark.parametrize("c", [point, line])
    def test_closed(self, c):
        assert not c.closed

    @pytest.mark.parametrize("c", [point, line])
    def test_close(self, c, caplog):
        c.close()
        assert len(caplog.messages) == 1
        assert "Cannot close Coordinates" in caplog.messages[0]

    @pytest.mark.parametrize("c", [point, line])
    def test_open(self, c, caplog):
        c.open()
        assert len(caplog.messages) == 1
        assert "Cannot open Coordinates" in caplog.messages[0]

    @pytest.mark.parametrize("c", [point, line])
    def test_check_ccw(self, c):
        assert not c.check_ccw()

    @pytest.mark.parametrize(("c", "lenn"), [(point, 1), (line, 2)])
    def test_len(self, c, lenn):
        assert len(c) == lenn

    @pytest.mark.parametrize("c", [point, line])
    def test_T(self, c):
        np.testing.assert_allclose(c.T[0], np.array([0, 0, 0]))

    @pytest.mark.parametrize("c", [point, line])
    def test_points(self, c):
        np.testing.assert_allclose(c.points[0], np.array([0, 0, 0]))


class TestCheckLineSegment:
    def test_true(self):
        a = [0, 0]
        b = [1, 0]
        c = [0.5, 0.0]
        assert check_linesegment(np.array(a), np.array(b), np.array(c)) is True
        a = [0.0, 0.0]
        b = [0.001, 0.0]
        c = [0.0005, 0.0]
        assert check_linesegment(np.array(a), np.array(b), np.array(c)) is True

        a = [0.0, 0.0]
        b = [1.0, 0.0]
        c = [1.0, 0.0]
        assert check_linesegment(np.array(a), np.array(b), np.array(c)) is True
        a = [0.0, 0.0]
        b = [0.001, 0.0]
        c = [0.0, 0.0]
        assert check_linesegment(np.array(a), np.array(b), np.array(c)) is True

    def test_false(self):
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        c = [5.0, 0.0]
        assert check_linesegment(np.array(a), np.array(b), np.array(c)) is False

        a = [0.0, 0.0]
        b = [0.001, 0.0]
        c = [0.005, 0.0]
        assert check_linesegment(np.array(a), np.array(b), np.array(c)) is False


class TestOnPolygon:
    def test_simple(self):
        coords = Coordinates({"x": [0, 1, 2, 2, 0, 0], "z": [-1, -1, -1, 1, 1, -1]})
        for p in coords.xz.T:
            assert on_polygon(p[0], p[1], coords.xz.T) is True

        fails = [[4, 4], [5, 5], [0.1, 0.1]]
        for fail in fails:
            assert on_polygon(*fail, coords.xz.T) is False


class TestInPolygon:
    @classmethod
    def teardown_class(cls):
        plt.close("all")

    def test_simple(self):
        coords = Coordinates({"x": [-2, 2, 2, -2, -2, -2], "z": [-2, -2, 2, 2, 1.5, -2]})
        in_points = [
            [-1, -1],
            [-1, 0],
            [-1, 1],
            [0, -1],
            [0, 0],
            [0, 1],
            [1, -1],
            [1, 0],
            [1, 1],
        ]

        out_points = [
            [-3, -3],
            [-3, 0],
            [-3, 3],
            [0, -3],
            [3, 3],
            [3, -3],
            [2.00000009, 0],
            [-2.0000000001, -1.999999999999],
        ]

        on_points = [
            [-2, -2],
            [2, -2],
            [2, 2],
            [-2, 0],
            [2, 0],
            [0, -2],
            [0, 2],
            [-2, 2],
        ]

        _, ax = plt.subplots()
        plot_coordinates(coords, ax=ax, edgecolor="k")
        for point in in_points:
            check = in_polygon(*point, coords.xz.T)
            c = "b" if check else "r"
            ax.plot(*point, marker="s", color=c)
        for point in on_points:
            check = in_polygon(*point, coords.xz.T)
            c = "b" if check else "r"
            ax.plot(*point, marker="o", color=c)
        for point in out_points:
            check = in_polygon(*point, coords.xz.T)
            c = "b" if check else "r"
            ax.plot(*point, marker="*", color=c)
        plt.show()

        # Test single and arrays
        for p in in_points:
            assert in_polygon(*p, coords.xz.T), p
        assert np.all(polygon_in_polygon(np.array(in_points), coords.xz.T))

        for p in on_points:
            assert in_polygon(*p, coords.xz.T, include_edges=True), p
        assert np.all(
            polygon_in_polygon(np.array(on_points), coords.xz.T, include_edges=True)
        )

        for p in on_points:
            assert not in_polygon(*p, coords.xz.T), p

        assert np.all(~polygon_in_polygon(np.array(on_points), coords.xz.T))

        for p in out_points:
            assert not in_polygon(*p, coords.xz.T), p
        assert np.all(~polygon_in_polygon(np.array(out_points), coords.xz.T))

    def test_big(self):
        """
        Regression test on a closed LCFS and an equilibrium grid.
        """
        lcfs = Coordinates.from_json(Path(TEST_PATH, "in_polygon_test.json"))

        x = np.linspace(4.383870967741935, 13.736129032258066, 65)
        z = np.linspace(-7.94941935483871, 7.94941935483871, 65)
        x, z = np.meshgrid(x, z, indexing="ij")

        n, m = x.shape
        mask = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                if in_polygon(x[i, j], z[i, j], lcfs.xz.T):
                    mask[i, j] = 1

        _, ax = plt.subplots()
        plot_coordinates(lcfs, ax=ax, fill=False, edgecolor="k")
        ax.contourf(x, z, mask, levels=[0, 0.5, 1])
        plt.show()

        hits = np.count_nonzero(mask)
        assert hits == 1171, hits


class TestIntersections:
    def teardown_method(self):
        plt.show()
        plt.close("all")

    @pytest.mark.parametrize(("c1", "c2"), [("x", "z"), ("x", "y"), ("y", "z")])
    def test_get_intersect(self, c1, c2):
        loop1 = Coordinates({c1: [0, 0.5, 1, 2, 3, 4, 0], c2: [1, 1, 1, 1, 2, 5, 5]})
        loop2 = Coordinates({c1: [1.5, 1.5, 2.5, 2.5, 2.5], c2: [4, -4, -4, -4, 5]})
        shouldbe = [[1.5, 1], [2.5, 1.5], [2.5, 5]]
        intersect = np.array(
            get_intersect(getattr(loop1, f"{c1}{c2}"), getattr(loop2, f"{c1}{c2}"))
        )
        correct = np.array(shouldbe).T
        np.testing.assert_allclose(intersect, correct)

    def test_join_intersect(self):
        loop1 = Coordinates(
            {"x": [0, 0.5, 1, 2, 3, 5, 4.5, 4, 0], "z": [1, 1, 1, 1, 2, 4, 4.5, 5, 5]}
        )
        loop2 = Coordinates({"x": [1.5, 1.5, 2.5, 2.5], "z": [4, -4, -4, 5]})
        join_intersect(loop1, loop2)

        np.testing.assert_allclose(loop1.points[3], [1.5, 0, 1])
        np.testing.assert_allclose(loop1.points[5], [2.5, 0, 1.5])
        np.testing.assert_allclose(loop1.points[10], [2.5, 0, 5])

    @pytest.mark.parametrize("file", ["", "2"])
    def test_join_intersect_arg(self, file):
        tf = Coordinates.from_json(Path(TEST_PATH, f"test_TF_intersect{file}.json"))
        lp = Coordinates.from_json(Path(TEST_PATH, f"test_LP_intersect{file}.json"))
        eq = Coordinates.from_json(Path(TEST_PATH, f"test_EQ_intersect{file}.json"))
        up = Coordinates.from_json(Path(TEST_PATH, f"test_UP_intersect{file}.json"))

        _, ax = plt.subplots()
        for coords in [tf, up, eq, lp]:
            plot_coordinates(coords, ax=ax, fill=False)

        args = []
        intx, intz = [], []
        for coords in [lp, eq, up]:
            i = get_intersect(tf.xz, coords.xz)
            a = join_intersect(tf, coords, get_arg=True)
            args.extend(a)
            intx.extend(i[0])
            intz.extend(i[1])

        for coords in [tf, up, eq, lp]:
            plot_coordinates(coords, ax=ax, fill=False, points=True)

        ax.plot(*tf.xz.T[args].T, marker="o", color="r")
        ax.plot(intx, intz, marker="^", color="k")

        assert len(intx) == len(args), f"{len(intx)} != {len(args)}"
        assert np.allclose(np.sort(intx), np.sort(tf.x[args])), f"{intx} != {tf.x[args]}"
        assert np.allclose(np.sort(intz), np.sort(tf.z[args])), f"{intz} != {tf.z[args]}"


class TestCoordinatesPlaneIntersect:
    @classmethod
    def teardown_class(cls):
        plt.close("all")

    def test_simple(self):
        coords = Coordinates({"x": [0, 1, 2, 2, 0, 0], "z": [-1, -1, -1, 1, 1, -1]})
        plane = BluemiraPlane.from_3_points([0, 0, 0], [1, 0, 0], [0, 1, 0])  # x-y
        intersect = coords_plane_intersect(coords, plane)
        e = np.array([[0, 0, 0], [2, 0, 0]])
        assert np.allclose(intersect, e)

    def test_complex(self):
        coords = Coordinates(
            {
                "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 8, 6, 4, 2, 0],
                "z": [-1, -2, -3, -4, -5, -6, -7, -8, -4, -2, 3, 2, 4, 2, 0, -1],
            }
        )
        plane = BluemiraPlane.from_3_points([0, 0, 0], [1, 0, 0], [0, 1, 0])  # x-y
        intersect = coords_plane_intersect(coords, plane)
        assert len(intersect) == 2

        _, ax = plt.subplots()
        plot_coordinates(coords, ax=ax)

        for i in intersect:
            ax.plot(i[0], i[2], marker="o", color="r")
            assert on_polygon(i[0], i[2], coords.xz.T)

        plane = BluemiraPlane.from_3_points(
            [0, 0, 2.7], [1, 0, 2.7], [0, 1, 2.7]
        )  # x-y offset
        intersect = coords_plane_intersect(coords, plane)
        assert len(intersect) == 4

        for i in intersect:
            ax.plot(i[0], i[2], marker="o", color="r")
            assert on_polygon(i[0], i[2], coords.xz.T)

        plane = BluemiraPlane.from_3_points(
            [0, 0, 4], [1, 0, 4], [0, 1, 4]
        )  # x-y offset
        intersect = coords_plane_intersect(coords, plane)
        assert len(intersect) == 1
        for i in intersect:
            ax.plot(i[0], i[2], marker="o", color="r")

            assert on_polygon(i[0], i[2], coords.xz.T)

        plane = BluemiraPlane.from_3_points(
            [0, 0, 4.0005], [1, 0, 4.0005], [0, 1, 4.0005]
        )  # x-y offset
        intersect = coords_plane_intersect(coords, plane)
        assert intersect is None

    def test_other_dims(self):
        coords = Coordinates(
            {
                "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 8, 6, 4, 2, 0],
                "y": [-1, -2, -3, -4, -5, -6, -7, -8, -4, -2, 3, 2, 4, 2, 0, -1],
            }
        )
        plane = BluemiraPlane.from_3_points([0, 0, 0], [1, 0, 0], [0, 0, 1])  # x-y
        intersect = coords_plane_intersect(coords, plane)
        assert len(intersect) == 2

        _, ax = plt.subplots()
        plot_coordinates(coords, ax=ax)
        for i in intersect:
            ax.plot(i[0], i[2], marker="o", color="r")
        plt.show()

        plane = BluemiraPlane.from_3_points([0, 10, 0], [1, 10, 0], [0, 10, 1])  # x-y
        intersect = coords_plane_intersect(coords, plane)
        assert intersect is None

    def test_xyzplane(self):
        coords = Coordinates(
            {
                "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 8, 6, 4, 2, 0],
                "y": [-1, -2, -3, -4, -5, -6, -7, -8, -4, -2, 3, 2, 4, 2, 0, -1],
            }
        )
        coords.translate((-2, 0, 0))
        plane = BluemiraPlane.from_3_points([0, 0, 0], [1, 1, 1], [2, 0, 0])  # x-y-z
        intersect = coords_plane_intersect(coords, plane)

        _, ax = plt.subplots()
        plot_coordinates(coords, ax=ax)
        for i in intersect:
            ax.plot(i[0], i[2], marker="o", color="r")
            assert on_polygon(i[0], i[2], coords.xz.T)

    def test_flat_intersect(self):
        # test that a shared segment with plane only gives two intersects
        coords = Coordinates({"x": [0, 2, 2, 0, 0], "z": [-1, -1, 1, 1, -1]})
        plane = BluemiraPlane.from_3_points([0, 0, 1], [0, 1, 1], [1, 0, 1])
        inter = coords_plane_intersect(coords, plane)
        assert np.allclose(inter, np.array([[0, 0, 1], [2, 0, 1]]))


class TestRotationMatrix:
    def test_axes(self):
        axes = ["x", "y", "z"]
        axes2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        for a1, a2 in zip(axes, axes2):
            r_1 = rotation_matrix(np.pi / 6, a1)
            r_2 = rotation_matrix(np.pi / 6, a2)
            assert np.allclose(r_1, r_2), a1

        axes = ["fail", "somthing", "1"]
        for axis in axes:
            with pytest.raises(CoordinatesError):
                rotation_matrix(30, axis)

    def test_ccw(self):
        p1 = [9, 0, 0]

        r_matrix = rotation_matrix(np.pi / 2, axis="z")
        p2 = r_matrix @ p1

        assert np.isclose(p2[1], 9), p2


def test_vector_lengthnorm_gives_expected_lengths_2d():
    points = np.array([[0, 0], [2, 1], [3, 3], [5, 1], [3, 0]], dtype=float).T

    lengths = vector_lengthnorm(points[0], points[1])

    expected = np.array(
        [
            0,
            np.sqrt(5),
            np.sqrt(5) + np.sqrt(5),
            np.sqrt(5) + np.sqrt(5) + np.sqrt(8),
            np.sqrt(5) + np.sqrt(5) + np.sqrt(8) + np.sqrt(5),
        ]
    )
    expected /= expected[-1]
    np.testing.assert_allclose(lengths, expected)


def test_vector_lengthnorm_gives_expected_lengths_3d():
    points = np.array(
        [[0, 0, 1], [2, 1, 2], [3, 3, 4], [5, 1, 0], [3, 0, 1]], dtype=float
    ).T

    lengths = vector_lengthnorm(points[0], points[1], points[2])

    expected = np.array(
        [
            0,
            np.sqrt(6),
            np.sqrt(6) + np.sqrt(9),
            np.sqrt(6) + np.sqrt(9) + np.sqrt(24),
            np.sqrt(6) + np.sqrt(9) + np.sqrt(24) + np.sqrt(6),
        ]
    )
    expected /= expected[-1]
    np.testing.assert_allclose(lengths, expected)
    np.testing.assert_allclose(lengths, expected)
