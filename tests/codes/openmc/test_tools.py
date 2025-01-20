# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from collections.abc import Iterable

import numpy as np
import openmc

from bluemira.codes.openmc.tools import (
    OpenMCEnvironment,
    torus_from_3points,
    torus_from_circle,
)
from bluemira.geometry.coordinates import Coordinates
from bluemira.radiation_transport.neutronics.wires import (
    CircleInfo,
    StraightLineInfo,
    WireInfo,
    WireInfoList,
)


def check_z_axis_centered(openmc_surface: openmc.Surface):
    """Check that the axis of rotation is centered at (0,0) on the xy-plane."""
    assert openmc_surface.x0 == 0.0
    assert openmc_surface.y0 == 0.0


def check_returned_surface_centered_at_z_axis(func):
    """
    Decorator to check if the returned surface(s) is/are axisymmetric along the z-axis.
    """

    def wrapper(*args, **kwargs):
        returned = func(*args, **kwargs)
        if isinstance(returned, Iterable):
            [check_z_axis_centered(surface) for surface in returned]
        else:
            check_z_axis_centered(returned)

    return wrapper


# All surfaces must be centered at the origin, by design

origin_coordinates = Coordinates([0.0, 0.0, 0.0])
origin = origin_coordinates.T[0]


@check_returned_surface_centered_at_z_axis
def test_ztorus():
    """
    https://docs.openmc.org/en/latest/pythonapi/generated/openmc.ZTorus.html

    The only use-case for the z-torus is to approximate arc of a circle in an
    axisymmetric model, so its axis of revolution should be the z-axis.
    """
    p1 = np.array([2, 0, 0.5])
    p2 = np.array([1.5, 0, 0])
    p3 = np.array([2.5, 0, 0])
    torus_1 = torus_from_3points(p1, p2, p3)
    # surface
    assert torus_1.a == 200  # cm
    assert torus_1.b == 50  # cm
    assert torus_1.c == 50  # cm
    # region
    assert np.array([1.9, 0, 0.1]) * 100 in -torus_1  # center of the torus
    assert origin not in -torus_1

    torus_2 = torus_from_circle([2.0, 0.0, 0.0], 0.5)
    # surface
    assert torus_2.a == 200  # cm
    assert torus_2.b == 50  # cm
    assert torus_2.c == 50  # cm
    # region
    assert np.array([1.9, 0, 0.1]) * 100 in -torus_2  # near center of the torus
    assert np.array([0.0, 2.1, 0.1]) * 100 in -torus_2  # near center of the torus
    assert origin not in -torus_2
    assert torus_2.id == (torus_1.id + 1)
    return torus_1, torus_2


class TestCSGEnv:
    def setup_method(self):
        self.env = OpenMCEnvironment()

    @check_returned_surface_centered_at_z_axis
    def test_torus_from_wire(self):
        """https://docs.openmc.org/en/latest/pythonapi/generated/openmc.ZTorus.html"""
        wire_info_list = WireInfoList([
            WireInfo(
                CircleInfo(
                    start_point=[1.5, 0, 0],
                    end_point=[2.5, 0, 0],
                    center=[2, 0, 0],
                    radius=0.5,
                ),
                [[None] * 3, [None] * 3],
            )
        ])
        torus = self.env.surfaces_from_info_list(wire_info_list)[0][1]
        assert np.array([1.9, 0, 0.1]) * 100 in -torus  # near center of the torus
        assert np.array([0.0, 2.1, 0.1]) * 100 in -torus  # near center of the torus
        assert origin not in -torus
        return torus

    def test_zplane_from_wire(self):
        """https://docs.openmc.org/en/latest/pythonapi/generated/openmc.ZPlane.html"""
        z1, z2, z3 = 1.0, 2.0, 3.0
        plane_1 = self._surface_from_2points([3.0, 0, z1], [4.0, 0, z1])
        plane_2 = self._surface_from_straight_line([3.0, 0, z2], [4.0, 0, z2])
        plane_3 = self._surfaces_from_single_straight_line_info_list(
            [3.0, 0, z3], [4.0, 0, z3]
        )

        for plane, z0 in zip((plane_1, plane_2, plane_3), (z1, z2, z3), strict=False):
            assert isinstance(plane, openmc.ZPlane)
            assert plane.z0 == z0 * 100

    @check_returned_surface_centered_at_z_axis
    def test_zcone(self):
        """https://docs.openmc.org/en/latest/pythonapi/generated/openmc.ZCone.html"""
        z1, z2, z3 = 3.0, 4.0, 5.0
        cone_1 = self._surface_from_2points([0.0, 0, z1], [1.0, 0, z1 + 1.0])
        cone_2 = self._surface_from_straight_line([0.0, 0, z2], [1.0, 0, z2 + 1.0])
        cone_3 = self._surfaces_from_single_straight_line_info_list(
            [0.0, 0, z3], [1.0, 0, z3 + 1.0]
        )

        for cone, z0 in zip((cone_1, cone_2, cone_3), (z1, z2, z3), strict=False):
            assert isinstance(cone, openmc.ZCone)
            assert cone.z0 == z0 * 100
        return cone_1, cone_2, cone_3

    @check_returned_surface_centered_at_z_axis
    def test_z_cylinder(self):
        """https://docs.openmc.org/en/latest/pythonapi/generated/openmc.ZCylinder.html"""
        r1, r2, r3 = 3.0, 4.0, 5.0
        cyl_1 = self._surface_from_2points([r1, 0, 1.0], [r1, 0, 0])
        cyl_2 = self._surface_from_straight_line([r2, 0, 1.0], [r2, 0, 0])
        cyl_3 = self._surfaces_from_single_straight_line_info_list(
            [r3, 0, 1.0], [r3, 0, 0]
        )

        for cyl in (cyl_1, cyl_2, cyl_3):
            assert isinstance(cyl, openmc.ZCylinder)

        return cyl_1, cyl_2, cyl_3

    def _surface_from_2points(
        self, start_point: Iterable[float], end_point: Iterable[float]
    ):
        return self.env.surface_from_2points(start_point[::2], end_point[::2])

    def _surface_from_straight_line(
        self, start_point: Iterable[float], end_point: Iterable[float]
    ):
        straight_line_info = StraightLineInfo(
            start_point=start_point,
            end_point=end_point,
        )
        return self.env.surface_from_straight_line(straight_line_info)

    def _surfaces_from_single_straight_line_info_list(
        self, start_point: Iterable[float], end_point: Iterable[float]
    ):
        straight_line_info = StraightLineInfo(
            start_point=start_point,
            end_point=end_point,
        )
        return self.env.surfaces_from_info_list(
            WireInfoList([WireInfo(straight_line_info, [[None] * 3, [None] * 3])])
        )[0][0]
