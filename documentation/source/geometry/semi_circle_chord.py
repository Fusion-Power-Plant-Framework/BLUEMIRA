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

"""
Generator for parameterisation semi_circle image
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc
from matplotlib.transforms import Bbox, IdentityTransform, TransformedBbox


class AngleAnnotation(Arc):
    """
    Draws an arc between two vectors which appears circular in display space.

    https://github.com/matplotlib/matplotlib/blob/main/examples/text_labels_and_annotations/angle_annotation.py
    """

    def __init__(
        self,
        xy,
        p1,
        p2,
        size=75,
        unit="points",
        ax=None,
        text="",
        textposition="inside",
        text_kw=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        xy, p1, p2 : tuple or array of two floats
            Center position and two points. Angle annotation is drawn between
            the two vectors connecting *p1* and *p2* with *xy*, respectively.
            Units are data coordinates.

        size : float
            Diameter of the angle annotation in units specified by *unit*.

        unit : str
            One of the following strings to specify the unit of *size*:

            * "pixels": pixels
            * "points": points, use points instead of pixels to not have a
              dependence on the DPI
            * "axes width", "axes height": relative units of Axes width, height
            * "axes min", "axes max": minimum or maximum of relative Axes
              width, height

        ax : `matplotlib.axes.Axes`
            The Axes to add the angle annotation to.

        text : str
            The text to mark the angle with.

        textposition : {"inside", "outside", "edge"}
            Whether to show the text in- or outside the arc. "edge" can be used
            for custom positions anchored at the arc's edge.

        text_kw : dict
            Dictionary of arguments passed to the Annotation.

        **kwargs
            Further parameters are passed to `matplotlib.patches.Arc`. Use this
            to specify, color, linewidth etc. of the arc.

        """
        self.ax = ax or plt.gca()
        self._xydata = xy  # in data coordinates
        self.vec1 = p1
        self.vec2 = p2
        self.size = size
        self.unit = unit
        self.textposition = textposition

        super().__init__(
            self._xydata,
            size,
            size,
            angle=0.0,
            theta1=self.theta1,
            theta2=self.theta2,
            **kwargs,
        )

        self.set_transform(IdentityTransform())
        self.ax.add_patch(self)

        self.kw = dict(
            ha="center",
            va="center",
            xycoords=IdentityTransform(),
            xytext=(0, 0),
            textcoords="offset points",
            annotation_clip=True,
        )
        self.kw.update(text_kw or {})
        self.text = ax.annotate(text, xy=self._center, **self.kw)

    def get_size(self):  # noqa: D102
        factor = 1.0
        if self.unit == "points":
            factor = self.ax.figure.dpi / 72.0
        elif self.unit.startswith("axes"):
            b = TransformedBbox(Bbox.unit(), self.ax.transAxes)
            dic = {
                "max": max(b.width, b.height),
                "min": min(b.width, b.height),
                "width": b.width,
                "height": b.height,
            }
            factor = dic[self.unit[5:]]
        return self.size * factor

    def set_size(self, size):  # noqa: D102
        self.size = size

    def get_center_in_pixels(self):
        """Return center in pixels"""
        return self.ax.transData.transform(self._xydata)

    def set_center(self, xy):
        """Set center in data coordinates"""
        self._xydata = xy

    def get_theta(self, vec):  # noqa: D102
        vec_in_pixels = self.ax.transData.transform(vec) - self._center
        return np.rad2deg(np.arctan2(vec_in_pixels[1], vec_in_pixels[0]))

    def get_theta1(self):  # noqa: D102
        return self.get_theta(self.vec1)

    def get_theta2(self):  # noqa: D102
        return self.get_theta(self.vec2)

    def set_theta(self, angle):  # noqa: D102
        pass

    # Redefine attributes of the Arc to always give values in pixel space
    _center = property(get_center_in_pixels, set_center)
    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)
    width = property(get_size, set_size)
    height = property(get_size, set_size)

    # The following two methods are needed to update the text position.
    def draw(self, renderer):  # noqa: D102
        self.update_text()
        super().draw(renderer)

    def update_text(self):  # noqa: D102
        c = self._center
        s = self.get_size()
        angle_span = (self.theta2 - self.theta1) % 360
        angle = np.deg2rad(self.theta1 + angle_span / 2)
        r = s / 2
        if self.textposition == "inside":
            r = s / np.interp(angle_span, [60, 90, 135, 180], [3.3, 3.5, 3.8, 4])
        self.text.xy = c + r * np.array([np.cos(angle), np.sin(angle)])
        if self.textposition == "outside":

            def R90(a, r, w, h):  # noqa: N802
                if a < np.arctan(h / 2 / (r + w / 2)):
                    return np.sqrt((r + w / 2) ** 2 + (np.tan(a) * (r + w / 2)) ** 2)
                else:
                    c = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
                    T = np.arcsin(  # noqa: N806
                        c * np.cos(np.pi / 2 - a + np.arcsin(h / 2 / c)) / r
                    )
                    xy = r * np.array([np.cos(a + T), np.sin(a + T)])
                    xy += np.array([w / 2, h / 2])
                    return np.sqrt(np.sum(xy**2))

            def R(a, r, w, h):  # noqa: N802
                aa = (a % (np.pi / 4)) * ((a % (np.pi / 2)) <= np.pi / 4) + (
                    np.pi / 4 - (a % (np.pi / 4))
                ) * ((a % (np.pi / 2)) >= np.pi / 4)
                return R90(aa, r, *[w, h][:: int(np.sign(np.cos(2 * a)))])

            bbox = self.text.get_window_extent()
            X = R(angle, r, bbox.width, bbox.height)  # noqa: N806
            trans = self.ax.figure.dpi_scale_trans.inverted()
            offs = trans.transform(((X - s / 2), 0))[0] * 72
            self.text.set_position([offs * np.cos(angle), offs * np.sin(angle)])


def draw_brace(span, axis_offset, text, ax=None, vertical=False, sideswitch=False):
    """
    Draws an annotated brace on the axes.

    https://stackoverflow.com/questions/18386210/annotating-ranges-of-data-in-matplotlib/68180887#68180887
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect("equal")

    min_, max_ = span
    span = max_ - min_

    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin

    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(span / xax_span * 100) * 2 + 1  # guaranteed uneven
    beta = 300.0 / xax_span  # the higher this is, the smaller the radius

    x = np.linspace(min_, max_, resolution)
    x_half = x[: int(resolution / 2) + 1]
    y_half_brace = 1 / (1.0 + np.exp(-beta * (x_half - x_half[0]))) + 1 / (
        1.0 + np.exp(-beta * (x_half - x_half[-1]))
    )
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = axis_offset + (0.05 * y - 0.01) * yspan  # adjust vertical position

    if vertical:
        px = y
        py = x

        textx = axis_offset + 0.05 * xax_span
        texty = (max_ + min_) / 2.0
        ha = "left"
        va = "center"
    else:
        px = x
        py = y

        textx = (max_ + min_) / 2.0
        texty = axis_offset + 0.07 * yspan

        va = "bottom"
        ha = "center"

    if sideswitch:
        py = -py
        if vertical:
            textx = -(axis_offset + 0.10 * xax_span)
        else:
            texty = -(axis_offset + 0.17 * yspan)

    ax.plot(px, py, color="black", lw=1)

    ax.text(textx, texty, text, ha=ha, va=va, math_fontfamily="cm")

    return ax


def semi_circle_angle_fig(height=20, width=20, ax=None, x0=50, y0=0):
    """
    Create semi circle chord figure
    """
    x = np.linspace(-width + x0, width + x0, num=1000)
    y = height * np.sqrt(1 - ((x - x0) / width) ** 2) + y0

    tr_1 = np.array([[x0 - width, x0], [y0, y0 + width]])
    tr_2 = np.array([[x0, x0 + width], [y0 + width, y0]])
    tr_3 = np.array([[x0 - width, x0 + width], [y0, y0]])

    if ax is None:
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect("equal")

    ax.plot(x, y)
    ax.plot(*tr_1, color="k")
    ax.plot(*tr_2, color="k")
    ax.plot(*tr_3, color="k")
    AngleAnnotation(
        (x0, height),
        tr_1[:, 0],
        tr_2[:, 1],
        ax=ax,
        size=75,
        text=r"$2\alpha$",
        edgecolor="k",
    )

    return ax


def annotated_figure():
    """
    Create full figure
    """
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect("equal")
    radius = 20
    x0 = 50
    y0 = 0
    ax = semi_circle_angle_fig(ax=ax, width=radius, height=radius, x0=x0, y0=y0)
    ax = draw_brace(
        [x0 - radius, x0 + radius],
        0,
        r"$\mathrm{Chord\ Width}\ (w)$",
        ax=ax,
        sideswitch=True,
    )
    ax = draw_brace(
        [0, radius], x0 + radius, r"$\mathrm{Chord\ Heigh}t\ (h)$", ax=ax, vertical=True
    )

    return fig, ax


def main():
    """Main"""
    fig, ax = annotated_figure()
    ax.set_xlim(27, 80)
    ax.set_ylim(-5, 21)
    ax.axis("off")
