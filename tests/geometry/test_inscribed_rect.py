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

from bluemira.display.plotter import PlotOptions, plot_2d
from bluemira.geometry._private_tools import make_circle_arc
from bluemira.geometry.constants import MINIMUM_LENGTH
from bluemira.geometry.coordinates import Coordinates, get_area, in_polygon
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.inscribed_rect import _rect, inscribed_rect_in_poly
from bluemira.geometry.tools import boolean_cut, make_circle, make_polygon


class TestInscribedRectangle:
    square = Coordinates(np.array([[2, 4, 4, 2, 2], [0, 0, 0, 0, 0], [0, 0, 2, 2, 0]]))
    diamond = Coordinates(
        np.array([[6, 8, 10, 8, 6], [0, 0, 0, 0, 0], [8, 6, 8, 10, 8]])
    )
    circle_xz = make_circle_arc(2, 4, -4)
    circle = Coordinates(
        np.array([circle_xz[0], np.zeros_like(circle_xz[0]), circle_xz[1]])
    )

    complex_shape = BluemiraFace(make_circle(2, center=(4, 0, -4), axis=(0, 1, 0)))
    circle_sm = BluemiraFace(make_circle(0.6, center=(5, 0, -5), axis=(0, 1, 0)))

    for i in [(0, 0, 0), (-2, 0, 2), (-2, 0, 0), (0, 0, 2)]:
        c_s = circle_sm.deepcopy()
        c_s.translate(i)
        complex_shape = boolean_cut(complex_shape, c_s)[0]

    complex_shape = complex_shape.boundary[0].discretize(  # noqa: PIE794
        byedges=True, ndiscr=100
    )

    shapes = (square, diamond, circle, complex_shape)
    convex = (True, True, True, False)

    aspectratios = np.logspace(-1, 1, num=5)

    po = PlotOptions(face_options={})

    MIN_AREA = MINIMUM_LENGTH**2

    @pytest.mark.parametrize(("shape", "convex"), zip(shapes, convex))
    def test_inscribed_rectangle(self, shape, convex):
        x = y = 5
        self.r = False
        # Random points in a rectangular grid of the shape
        points = np.random.random((3, x, y))
        points[1] = 0
        points[0] *= np.ptp(shape[0])
        points[2] *= np.ptp(shape[2])
        points[0] += np.min(shape[0])
        points[2] += np.min(shape[2])

        fig, ax = plt.subplots()

        shape_face = BluemiraFace(make_polygon(shape, closed=True))
        plot_2d(
            shape_face,
            self.po,
            ax=ax,
            face_options={"zorder": -10},
            wire_options={"linewidth": 0.1},
            show=False,
        )
        for i in range(x):
            for j in range(y):
                point = points[:, i, j]
                if in_polygon(point[0], point[2], shape.xz.T):
                    for k in self.aspectratios:
                        dx, dz = inscribed_rect_in_poly(
                            shape.x,
                            shape.z,
                            point[0],
                            point[2],
                            aspectratio=k,
                            convex=convex,
                        )
                        sq = _rect(point[0], point[2], dx, dz)
                        assert len(sq.x) == 5
                        try:
                            tf = boolean_cut(
                                BluemiraFace(make_polygon(sq.xyz, closed=True)),
                                shape_face,
                            )
                            tf = [
                                Coordinates(seg.discretize(byedges=True, ndiscr=50))
                                for seg in tf
                            ]
                        except ValueError:
                            tf = None

                        ax.plot(point[0], point[2], marker="o")
                        ax.plot(sq.x, sq.z, linewidth=0.1, color="k")
                        # Some overlaps are points or lines of 0 area
                        if tf is not None and not all(
                            get_area(*t.xz) <= self.MIN_AREA for t in tf
                        ):
                            self.assertion_error_creator(
                                "Overlap", [dx, dz, point, k, convex]
                            )

                            for t in tf:
                                t.plot(
                                    ax,
                                    facecolor="r",
                                    edgecolor="r",
                                    linewidth=1,
                                    zorder=40,
                                )

                        if not np.allclose(dx / dz, k):
                            self.assertion_error_creator("Aspect", [dx, dz, dx / dz, k])
        plt.show()
        plt.close(fig)

        if self.r is not False:
            raise AssertionError(self.r)

    def assertion_error_creator(self, name, list_obj):
        if self.r is False:
            self.r = {name: [list_obj]}
        else:
            if name not in self.r:
                self.r = {name: []}
            self.r[name].append(list_obj)

    def test_zero(self):
        assert inscribed_rect_in_poly(self.diamond.x, self.diamond.z, 0, 0) == (0, 0)
