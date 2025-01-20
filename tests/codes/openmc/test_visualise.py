# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from collections.abc import Iterable

import matplotlib.pyplot as plt
import openmc
import pytest

from bluemira.codes.openmc.visualise import plot_surfaces

plane_1 = openmc.ZPlane(z0=10)
plane_2 = openmc.ZPlane(z0=-10)
plane_3 = openmc.ZPlane(z0=-20)
cone_1 = openmc.ZCone(z0=10, r2=1)
cone_2 = openmc.ZCone(z0=-10, r2=4)
cone_3 = openmc.ZCone(z0=-20, r2=9)
cylinder_1 = openmc.ZCylinder(r=10)
cylinder_2 = openmc.ZCylinder(r=20)
cylinder_3 = openmc.ZCylinder(r=30)
major_radius = 100
minor_radius = 15
torus_1 = openmc.ZTorus(z0=10, a=major_radius, b=minor_radius, c=minor_radius)
torus_2 = openmc.ZTorus(z0=-10, a=major_radius, b=minor_radius, c=minor_radius)
torus_3 = openmc.ZTorus(z0=-20, a=major_radius, b=minor_radius, c=minor_radius)


@pytest.mark.parametrize(
    ("surfaces_list", "plot_both_sides"),
    [
        ([plane_1, plane_2, plane_3], False),
        ([plane_1, plane_2, plane_3], True),
        ([cone_1, cone_2, cone_3], False),
        ([cone_1, cone_2, cone_3], True),
        ([cylinder_1, cylinder_2, cylinder_3], False),
        ([cylinder_1, cylinder_2, cylinder_3], True),
        ([torus_1, torus_2, torus_3], False),
        ([torus_1, torus_2, torus_3], True),
        ([plane_1, cone_2, cylinder_1, torus_3], False),
        ([plane_1, cone_2, cylinder_1, torus_3], True),
        # (plane, cone, cylinder, torus, True),
        # (plane, cone, cylinder, torus, False),
    ],
)
def test_plot_surfaces(
    surfaces_list: Iterable[openmc.Surface], *, plot_both_sides: bool
):
    """Check that the correct number of items are plotted using plot_surface."""
    ax = plot_surfaces(surfaces_list, plot_both_sides=plot_both_sides)
    num_patches_expected, num_straight_lines_expected = 0, 0
    for surface_tuple in surfaces_list:
        if not isinstance(surface_tuple, tuple):
            surface_tuple = (surface_tuple,)
        for surface in surface_tuple:
            if isinstance(surface, openmc.ZTorus):
                num_patches_expected += 1 + int(plot_both_sides)
            elif isinstance(surface, openmc.ZCone):
                num_straight_lines_expected += 2
            elif isinstance(surface, openmc.ZCylinder):
                num_straight_lines_expected += 1 + int(plot_both_sides)
            elif isinstance(surface, openmc.ZPlane):
                num_straight_lines_expected += 1

    assert len(ax.patches) == num_patches_expected
    assert len(ax.lines) == num_straight_lines_expected
    ax.clear()
    plt.close()
