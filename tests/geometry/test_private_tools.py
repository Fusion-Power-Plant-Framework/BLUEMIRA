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
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bluemira.base.constants import EPS
from bluemira.base.file import get_bluemira_path
from bluemira.codes.error import FreeCADError
from bluemira.geometry._private_tools import (
    convert_coordinates_to_face,
    convert_coordinates_to_wire,
    make_face,
    make_mixed_face,
    make_mixed_wire,
    make_wire,
    offset,
)
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.coordinates import Coordinates, get_area
from bluemira.geometry.error import GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import extrude_shape, revolve_shape

TEST_PATH = get_bluemira_path("geometry/test_data", subfolder="tests")


class TestArea:
    def test_area(self):
        """
        Checked with:
        https://www.analyzemath.com/Geometry_calculators/irregular_polygon_area.html
        """
        x = np.array([0, 1, 2, 3, 4, 5, 6, 4, 3, 2])
        y = np.array([0, -5, -3, -5, -1, 0, 2, 6, 4, 1])
        assert get_area(x, y) == pytest.approx(29.5, rel=0, abs=EPS)
        coords = Coordinates({"x": x, "y": y})
        coords.rotate(base=(3, 2, 1), direction=(42, 2, 1), degree=43)
        assert np.isclose(get_area(*coords.xyz), 29.5)

    def test_error(self):
        x = np.array([0, 1, 2, 3, 4, 5, 6, 4, 3, 2])
        y = np.array([0, -5, -3, -5, -1, 0, 2, 6, 4, 1])
        with pytest.raises(GeometryError):
            get_area(x, y[:-1])


class TestOffset:
    @classmethod
    def teardown_class(cls):
        plt.close("all")

    def test_rectangle(self):
        # Rectangle - positive offset
        x = [1, 3, 3, 1, 1, 3]
        y = [1, 1, 3, 3, 1, 1]
        o = offset(x, y, 0.25)
        assert sum(o[0] - np.array([0.75, 3.25, 3.25, 0.75, 0.75])) == 0
        assert sum(o[1] - np.array([0.75, 0.75, 3.25, 3.25, 0.75])) == 0

        _, ax = plt.subplots()
        ax.plot(x, y, "k")
        ax.plot(*o, "r", marker="o")
        ax.set_aspect("equal")

    def test_triangle(self):
        x = [1, 2, 1.5, 1, 2]
        y = [1, 1, 4, 1, 1]
        t = offset(x, y, -0.25)
        assert (
            abs(sum(t[0] - np.array([1.29511511, 1.70488489, 1.5, 1.29511511])) - 0)
            < 1e-3
        )
        assert abs(sum(t[1] - np.array([1.25, 1.25, 2.47930937, 1.25])) - 0) < 1e-3

        _, ax = plt.subplots()
        ax.plot(x, y, "k")
        ax.plot(*t, "r", marker="o")
        ax.set_aspect("equal")

    def test_complex_open(self):
        # fmt:off
        x = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2]
        y = [0, -2, -4, -3, -4, -2, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 4, 3, 2, 1, 2, 2, 1]
        # fmt:on

        c = offset(x, y, 1)

        _, ax = plt.subplots()
        ax.plot(x, y, "k")
        ax.plot(*c, "r", marker="o")
        ax.set_aspect("equal")

    def test_complex_closed(self):
        # fmt:off
        x = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -3]
        y = [0, -2, -4, -3, -4, -2, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 4, 3, 2, 1, 2, 2, 1, 1, 0, 2]
        # fmt:on

        c = offset(x, y, 1)

        _, ax = plt.subplots()
        ax.plot(x, y, "k")
        ax.plot(*c, "r", marker="o")
        ax.set_aspect("equal")


class TestMixedFaces:
    """
    Various tests of the MixedFaceMaker functionality. Checks the 3-D geometric
    properties of the results with some regression results done when everything was
    working correctly.
    """

    def assert_properties(self, true_props: Dict[str, Any], part: BluemiraGeo):
        """
        Helper function to pull out the properties to be compared, and to make the
        comparison in an output-friendly way.
        """
        error = False
        kwargs = {"atol": 1e-8, "rtol": 1e-5}
        keys, expected, actual = [], [], []
        for key, value in true_props.items():
            comp_method = np.allclose if isinstance(value, tuple) else np.isclose
            result = getattr(part, key, None)
            assert result is not None, f"Attribute {key} not defined on part {part}."
            if not comp_method(value, result, **kwargs):
                error = True
                keys.append(key)
                expected.append(value)
                actual.append(result)
        if error:
            raise AssertionError(list(zip(keys, expected, actual)))

    @pytest.mark.parametrize(
        ("filename", "degree", "true_props"),
        [
            (
                "IB_test.json",
                100,
                {
                    "center_of_mass": (
                        3.50437337,
                        4.17634955,
                        1.17868604,
                    ),
                    "volume": 106.080,
                    "area": 348.296,
                },
            ),
            (
                "OB_test.json",
                15,
                {
                    "center_of_mass": (
                        11.5828485,
                        1.52491093,
                        -0.18624372,
                    ),
                    "volume": 43.02953145397336,
                    "area": 121.585591636,
                },
            ),
        ],
    )
    def test_face_revolve(self, filename, degree, true_props):
        """
        Tests some blanket faces that combine splines and polygons.
        """
        coords = Coordinates.from_json(Path(TEST_PATH, filename))
        face = make_mixed_face(*coords.xyz)
        part = revolve_shape(face, degree=degree, label=filename)
        self.assert_properties(true_props, part)

    @pytest.mark.parametrize(
        ("filename", "vec", "true_props"),
        [
            (
                "TF_case_in_test.json",
                (0, 1, 0),
                {
                    "center_of_mass": (
                        9.45877,
                        0.5,
                        -2.1217e-5,
                    ),
                    "volume": 185.185,
                    "area": 423.998,
                },
            ),
            (
                "div_test_mfm2.json",
                (0, 2, 0),
                {
                    "center_of_mass": (
                        8.03265,
                        0.9900,
                        -6.44432,
                    ),
                    "volume": 4.58959,
                    "area": 29.1868,
                },
            ),
        ],
    )
    def test_face_extrude(self, filename, vec, true_props):
        """
        Tests TF and divertor faces that combine splines and polygons.
        """
        fn = Path(TEST_PATH, filename)
        coords = Coordinates.from_json(fn)
        face = make_mixed_face(*coords.xyz)
        part = extrude_shape(face, vec=vec, label=filename)

        self.assert_properties(true_props, part)

    def test_face_seg_fault(self):
        """
        Tests a particularly tricky face that can result in a seg fault...
        """
        fn = Path(TEST_PATH, "divertor_seg_fault_LDS.json")
        coords = Coordinates.from_json(fn)
        face = make_mixed_face(*coords.xyz)
        true_props = {
            "area": 2.26163,
        }
        self.assert_properties(true_props, face)

    @pytest.mark.parametrize(
        ("name", "true_props"),
        [
            (
                "shell_mixed_test",
                {
                    "area": 6.35215,
                },
            ),
            (
                "failing_mixed_shell",
                {
                    "area": 31.4998,
                },
            ),
            (
                "tf_wp_tricky",
                {
                    "area": 31.0914,
                },
            ),
        ],
    )
    def test_shell(self, name, true_props):
        """
        Tests some shell mixed faces
        """
        inner = Coordinates.from_json(Path(TEST_PATH, f"{name}_inner.json"))
        outer = Coordinates.from_json(Path(TEST_PATH, f"{name}_outer.json"))
        inner_wire = make_mixed_wire(*inner.xyz)
        outer_wire = make_mixed_wire(*outer.xyz)
        face = BluemiraFace([outer_wire, inner_wire])
        self.assert_properties(true_props, face)

    def test_coordinate_cleaning(self):
        fn = Path(TEST_PATH, "bb_ob_bss_test.json")
        coords = Coordinates.from_json(fn)
        make_mixed_wire(*coords.xyz, allow_fallback=False)

        with pytest.raises(FreeCADError):
            make_mixed_wire(*coords.xyz, allow_fallback=False, cleaning_atol=1e-8)


class TestCoordsConversion:
    def generate_face_polygon(self, x, y, z):
        face = make_face(x, y, z, spline=False)
        converted_face = convert_coordinates_to_face(x, y, z, method="polygon")
        return face, converted_face

    def generate_face_spline(self, x, y, z):
        face = make_face(x, y, z, spline=True)
        converted_face = convert_coordinates_to_face(x, y, z, method="spline")
        return face, converted_face

    def generate_face_mixed(self, x, y, z):
        face = make_mixed_face(x, y, z)
        converted_face = convert_coordinates_to_face(x, y, z)
        return face, converted_face

    def generate_wire_polygon(self, x, y, z):
        wire = make_wire(x, y, z, spline=False)
        converted_wire = convert_coordinates_to_wire(x, y, z, method="polygon")
        return wire, converted_wire

    def generate_wire_spline(self, x, y, z):
        wire = make_wire(x, y, z, spline=True)
        converted_wire = convert_coordinates_to_wire(x, y, z, method="spline")
        return wire, converted_wire

    def generate_wire_mixed(self, x, y, z):
        wire = make_mixed_wire(x, y, z)
        converted_wire = convert_coordinates_to_wire(x, y, z)
        return wire, converted_wire

    @pytest.mark.parametrize(
        ("filename", "method"),
        [
            ("IB_test.json", generate_face_polygon),
            ("IB_test.json", generate_face_spline),
            ("IB_test.json", generate_face_mixed),
        ],
    )
    def test_coordinates_to_face(self, filename, method):
        fn = Path(TEST_PATH, filename)
        coords = Coordinates.from_json(fn)
        face, converted_face = method(self, *coords.xyz)
        assert face.area == converted_face.area
        assert face.volume == converted_face.volume
        np.testing.assert_equal(face.center_of_mass, converted_face.center_of_mass)

    @pytest.mark.parametrize(
        ("filename", "method"),
        [
            ("IB_test.json", generate_wire_polygon),
            ("IB_test.json", generate_wire_spline),
            ("IB_test.json", generate_wire_mixed),
        ],
    )
    def test_coordinates_to_wire_polygon(self, filename, method):
        fn = Path(TEST_PATH, filename)
        coords = Coordinates.from_json(fn)
        wire, converted_wire = method(self, *coords.xyz)
        assert wire.area == converted_wire.area
