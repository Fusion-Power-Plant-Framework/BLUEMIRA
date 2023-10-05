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

from typing import Callable, ClassVar

import numpy as np
import pytest
from scipy.interpolate import interp1d

from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.error import BuilderError
from bluemira.builders.tools import (
    build_sectioned_xy,
    build_sectioned_xyz,
    get_n_sectors,
    make_circular_xy_ring,
    pattern_lofted_silhouette,
    pattern_revolved_silhouette,
    varied_offset,
)
from bluemira.geometry.error import GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import PictureFrame
from bluemira.geometry.tools import (
    distance_to,
    find_clockwise_angle_2d,
    make_circle,
    make_polygon,
    offset_wire,
)
from bluemira.geometry.wire import BluemiraWire


class TestGetNSectors:
    sector_degree = (
        360.0,
        180.0,
        120.0,
        90.0,
        72.0,
        60.0,
        51.42857142857143,
        45.0,
        40.0,
        36.0,
        32.72727272727273,
        30.0,
        27.692307692307693,
        25.714285714285715,
        24.0,
    )

    n_sectors: ClassVar = {
        1: [1, 1, 1, 1, 1, 1, 1],
        5: [1, 1, 1, 2, 3, 4, 5],
        7: [1, 1, 2, 3, 4, 5, 7],
        9: [1, 1, 3, 4, 6, 7, 9],
    }

    @pytest.mark.parametrize(
        ("ttl", "sector_degree"), zip(np.arange(1, 16), sector_degree)
    )
    @pytest.mark.parametrize("degree", np.arange(0, 361, step=60))
    def test_get_n_sectors_degree(self, degree, ttl, sector_degree):
        s_deg, _ = get_n_sectors(ttl, degree)
        assert np.isclose(s_deg, sector_degree)

    @pytest.mark.parametrize("ttl", n_sectors.keys())
    @pytest.mark.parametrize(("ind", "degree"), enumerate(np.arange(0, 361, step=60)))
    def test_get_n_sectors_amount(self, ind, degree, ttl):
        _, n_sec = get_n_sectors(ttl, degree)
        assert np.isclose(n_sec, self.n_sectors[ttl][ind])


class TestVariedOffsetFunction:
    fixtures = (
        {
            "wire": PictureFrame(
                {
                    "x1": {"value": 4.5, "lower_bound": 4, "upper_bound": 5},
                    "x2": {"value": 16, "lower_bound": 14, "upper_bound": 18},
                    "z1": {"value": 8, "lower_bound": 5, "upper_bound": 15},
                    "z2": {"value": -6, "lower_bound": -15, "upper_bound": -5},
                    "ro": {"value": 6},
                    "ri": {"value": 3},
                }
            ).create_shape(),
            "inboard_offset": 1,
            "outboard_offset": 4,
            "inboard_offset_degree": 45,  # degrees
            "outboard_offset_degree": 160,  # degrees
        },
        {
            "wire": make_circle(axis=(0, 1, 0)),
            "inboard_offset": 1,
            "outboard_offset": 3,
            "inboard_offset_degree": 90,  # degrees
            "outboard_offset_degree": 140,  # degrees
        },
    )

    @pytest.mark.parametrize("kwargs", fixtures)
    def test_offset_wire_is_closed(self, kwargs):
        offset_wire = varied_offset(**kwargs)

        assert offset_wire.is_closed

    @pytest.mark.parametrize("kwargs", fixtures)
    def test_the_offset_at_the_ob_radius_is_major_offset(self, kwargs):
        offset_wire = varied_offset(**kwargs)

        offset_size = self._get_offset_sizes(offset_wire, kwargs["wire"], np.pi)
        # Trade-off here between finer discretization when we
        # interpolate and a tighter tolerance. A bit of lee-way here for
        # the sake of fewer points to interpolate
        assert offset_size == pytest.approx(kwargs["outboard_offset"], rel=1e-3)

    @pytest.mark.parametrize("kwargs", fixtures)
    def test_offset_from_shape_never_lt_minor_offset(self, kwargs):
        offset_wire = varied_offset(**kwargs)

        ang_space = np.linspace(0, 2 * np.pi, 50)
        offset_size = self._get_offset_sizes(offset_wire, kwargs["wire"], ang_space)
        assert self.greater_or_close(offset_size, kwargs["inboard_offset"], atol=1e-3)

    @pytest.mark.parametrize("kwargs", fixtures)
    def test_offset_never_decreases_between_offset_angles(self, kwargs):
        offset_wire = varied_offset(**kwargs)

        ang_space = np.linspace(
            np.radians(kwargs["inboard_offset_degree"]),
            np.radians(kwargs["outboard_offset_degree"]),
            50,
        )
        offset_size = self._get_offset_sizes(offset_wire, kwargs["wire"], ang_space)
        offset_size_gradient = np.diff(offset_size)
        assert self.greater_or_close(offset_size_gradient, 0, atol=1e-2)

    @pytest.mark.parametrize("kwargs", fixtures)
    def test_offset_eq_to_minor_offset_at_0_degrees(self, kwargs):
        offset_wire = varied_offset(**kwargs)

        offset_size = self._get_offset_sizes(offset_wire, kwargs["wire"], 0)
        assert offset_size == pytest.approx(kwargs["inboard_offset"], 1e-3)

    def test_GeometryError_raised_given_input_wire_not_closed(self):
        wire = make_polygon([[0, 1], [0, 0], [1, 1]])

        with pytest.raises(GeometryError):
            varied_offset(wire, 1, 2, 50, 170)

    def test_GeometryError_raised_given_input_wire_not_xz_planar(self):
        wire = make_circle(axis=(1, 1, 1))

        with pytest.raises(GeometryError):
            varied_offset(wire, 1, 2, 50, 170)

    @pytest.mark.parametrize(
        "angle", ["inboard_offset_degree", "outboard_offset_degree"]
    )
    @pytest.mark.parametrize("angle_degree", [-1, -0.001, 180.01, 181])
    def test_ValueError_given_angle_not_in_range_0_to_180(self, angle, angle_degree):
        kwargs = self.fixtures[0].copy()
        kwargs.update({angle: angle_degree})

        with pytest.raises(ValueError):  # noqa: PT011
            varied_offset(**kwargs)

    def test_ValueError_given_inboard_degree_gt_outboard_degree(self):
        with pytest.raises(ValueError):  # noqa: PT011
            varied_offset(
                make_circle(axis=(0, 1, 0)),
                1,
                3,
                inboard_offset_degree=91,
                outboard_offset_degree=90,
            )

    @staticmethod
    def _interpolation_func_closed_wire(wire: BluemiraWire) -> Callable:
        coords = wire.discretize(200).xz
        centroid = wire.center_of_mass[[0, 2]]
        angles = np.radians(
            find_clockwise_angle_2d(np.array([-1, 0]), coords - centroid.reshape((2, 1)))
        )
        if angles[0] == angles[-1]:
            # If the end angle == the start angle, we've looped back
            # around from 0 to 2π. We only want the range [0, 2π), so
            # remove the last. Not doing this causes the interpolation
            # function to return NaNs at 0.
            angles = angles[:-1]
        sort_idx = np.argsort(angles)
        angle_space = angles[sort_idx]
        sorted_coords = coords[:, sort_idx]
        return interp1d(angle_space, sorted_coords, fill_value="extrapolate")

    @staticmethod
    def _get_offset_sizes(
        shape_1: BluemiraWire, shape_2: BluemiraWire, angles: np.ndarray
    ) -> np.ndarray:
        """
        Gets the "radial" offsets between the two shapes at the given
        angles.

        Note that this is the radial offset from the center of the
        shapes, not the offset in the direction of the normal of the
        first shape (or "normal" offset, which is what we really want).
        """
        interp_1 = TestVariedOffsetFunction._interpolation_func_closed_wire(shape_1)
        interp_2 = TestVariedOffsetFunction._interpolation_func_closed_wire(shape_2)
        return np.linalg.norm(interp_1(angles) - interp_2(angles), axis=0)

    @staticmethod
    def greater_or_close(values: np.ndarray, limit: float, **kwargs):
        """
        Check if the given values are greater or close to the given limit.

        The kwargs are passed directly to ``np.isclose``
        """
        return np.all(
            np.logical_or(values >= limit, np.isclose(values, limit, **kwargs))
        )


class TestPatterning:
    fixture = (
        (3, 16, 0.0),
        (3, 16, 0.1),
        (10, 20, 0.05),
        (2, 1, 1),
    )

    @pytest.mark.parametrize(("n_segments", "n_sectors", "gap"), fixture)
    def test_revolved_silhouette(self, n_segments, n_sectors, gap):
        p = make_polygon({"x": [4, 5, 5, 4], "y": 0, "z": [-1, -1, 1, 1]}, closed=True)
        face = BluemiraFace(p)

        shapes = pattern_revolved_silhouette(face, n_segments, n_sectors, gap)

        assert len(shapes) == n_segments

        volumes = [shape.volume for shape in shapes]
        np.testing.assert_almost_equal(volumes[1:], volumes[0])

        distances = self._distances_between_shapes(shapes)
        np.testing.assert_almost_equal(distances, gap)

        # Slightly dubious estimate for the volume of parallel gaps
        com_radius = p.center_of_mass[0]
        gamma_gap = 2 * np.arcsin(0.5 * gap / com_radius)
        d_l = com_radius * gamma_gap
        theory_gap = face.area * n_segments * d_l

        theory_volume = face.area * com_radius * 2 * np.pi / (n_sectors) - theory_gap

        np.testing.assert_allclose(sum(volumes), theory_volume, rtol=5e-6)

    @pytest.mark.parametrize(("n_segments", "n_sectors", "gap"), fixture[:-1])
    def test_lofted_silhouette(self, n_segments, n_sectors, gap):
        p = make_polygon({"x": [4, 5, 5, 4], "y": 0, "z": [-1, -1, 1, 1]}, closed=True)
        face = BluemiraFace(p)

        shapes = pattern_lofted_silhouette(face, n_segments, n_sectors, gap)

        assert len(shapes) == n_segments
        volumes = [shape.volume for shape in shapes]
        np.testing.assert_almost_equal(volumes[1:], volumes[0])

        distances = self._distances_between_shapes(shapes)
        np.testing.assert_almost_equal(distances, gap)

    @staticmethod
    def _distances_between_shapes(shapes):
        return [distance_to(shapes[i], shapes[i + 1])[0] for i in range(len(shapes) - 1)]


class TestMakeCircularRing:
    fixture = (
        (0.002, 0.003),
        (3, 4),
        (3.15, 3.16),
        (1e5, 1e6),
    )

    @pytest.mark.parametrize(("r_in", "r_out"), fixture)
    def test_annulus_area(self, r_in, r_out):
        face = make_circular_xy_ring(r_in, r_out)
        np.testing.assert_almost_equal(face.area, np.pi * (r_out**2 - r_in**2))

    @pytest.mark.parametrize(("r_in", "r_out"), fixture)
    def test_annulus_area_reversed_radii(self, r_in, r_out):
        r_out, r_in = r_in, r_out
        face = make_circular_xy_ring(r_in, r_out)
        np.testing.assert_almost_equal(face.area, np.pi * (r_in**2 - r_out**2))

    def test_raises_error_on_equal_radii(self):
        with pytest.raises(BuilderError):
            make_circular_xy_ring(1, 1)


class TestBuildSectioned:
    plot_colour = (1, 1, 1)

    sq_arr = np.array([[2, 0, -0.5], [3, 0, -0.5], [3, 0, 0.5], [2, 0, 0.5]]).T
    # rifling shape, for edge cases
    sq_arr_2 = np.array([[2, 2, -0.5], [3, 2, -0.5], [3, 2, 0.5], [2, 2, 0.5]]).T
    circ1 = make_circle(10, center=(15, 0, 0), axis=(0.0, 1.0, 0.0))

    enable_sectioning = (True, True, True, False, False, False)

    faces = []  # noqa: RUF012
    for sec in [sq_arr, sq_arr_2, circ1]:
        if not isinstance(sec, BluemiraWire):
            sec = make_polygon(sec, closed=True)  # noqa: PLW2901
        offset = offset_wire(
            sec,
            1,
            join="intersect",
            open_wire=False,
            ndiscr=600,
        )
        faces.append(BluemiraFace([offset, sec]))

    # failing test mark
    face_sec = list(map(list, zip(*[faces + faces, enable_sectioning])))  # noqa: RUF012
    face_sec[2] = pytest.param(
        faces[2],
        enable_sectioning[2],
        marks=pytest.mark.xfail(reason="Possible #1347 Topology failure"),
    )

    @pytest.mark.parametrize("face", faces)
    def test_build_sectioned_xy(self, face):
        sec = build_sectioned_xy(face, self.plot_colour)

        assert len(sec) == 2
        assert all(isinstance(s, PhysicalComponent) for s in sec)
        assert [s.plot_options.face_options["color"] == self.plot_colour for s in sec]

    @pytest.mark.parametrize(("face", "section_bool"), face_sec)
    def test_build_sectioned_xyz(self, face, section_bool):
        sec = build_sectioned_xyz(
            face, "test", 12, self.plot_colour, enable_sectioning=section_bool
        )

        assert len(sec) == 12 if section_bool else len(sec) == 1
        assert all(
            isinstance(s, Component if section_bool else PhysicalComponent) for s in sec
        )
        assert [s.plot_options.face_options["color"] == self.plot_colour for s in sec]
