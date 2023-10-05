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
Tests for palettes module
"""
from copy import copy

import pytest
from matplotlib import colors

from bluemira.display.palettes import (
    ColorPalette,
    background_colour_string,
    make_alpha_palette,
    make_rgb_alpha,
)


class TestColorPalette:
    rect_str = (
        '<rect x="{}" y="{}" width="55" height="55"'
        ' style="fill:{};stroke-width:2;stroke:rgb(255,255,255)"/>'
    )

    def setup_method(self):
        self.pal = ColorPalette({"C1": "#000000", "C2": "#ffffff"})

    def test_keys(self):
        assert self.pal.keys() == {"C1", "C2"}

    def test_next(self):
        assert next(self.pal) == "#000000"
        assert next(self.pal) == "#ffffff"
        assert next(self.pal) == "#000000"

    def test_set_get_item(self):
        self.pal[0] = "#aaaaaa"
        assert colors.to_hex(self.pal["C1"][0]) == "#aaaaaa"

        self.pal["C1"] = "#000000"
        assert colors.to_hex(self.pal["C1"][0]) == "#000000"

    def test_repr_term(self):
        assert (
            self.pal._repr_colour_str(self.pal._hex_horizontal())
            == "\x1b[48:2::0:0:0m  \x1b[49m\x1b[48:2::255:255:255m  \x1b[49m\n"
        )

    def test_repr_html(self):
        assert self.pal._repr_html() == (
            '<svg  width="110" height="55">'
            + self.rect_str.format(0, 0, "#000000")
            + self.rect_str.format(55, 0, "#ffffff")
            + "</svg>"
        )

    def test_repr_html_with_alpha(self):
        pal = copy(self.pal)
        pal["C1"] = make_alpha_palette(pal["C1"], 8)
        pal["C2"] = make_alpha_palette(pal["C2"], 3)

        assert pal._repr_html() == (
            '<svg  width="110" height="440">'
            + self.rect_str.format(0, 0, "#000000")
            + self.rect_str.format(55, 0, "#ffffff")
            + self.rect_str.format(0, 55, "#202020")
            + self.rect_str.format(55, 55, "#ffffff")
            + self.rect_str.format(0, 110, "#404040")
            + self.rect_str.format(55, 110, "#ffffff")
            + self.rect_str.format(0, 165, "#606060")
            + self.rect_str.format(0, 220, "#808080")
            + self.rect_str.format(0, 275, "#9f9f9f")
            + self.rect_str.format(0, 330, "#bfbfbf")
            + self.rect_str.format(0, 385, "#dfdfdf")
            + "</svg>"
        )

    def test_repr_term_with_alpha(self):
        pal = copy(self.pal)
        pal["C1"] = make_alpha_palette(pal["C1"], 8)
        pal["C2"] = make_alpha_palette(pal["C2"], 3)

        assert (
            pal._repr_colour_str(pal._hex_horizontal())
            == "\x1b[48:2::0:0:0m  \x1b[49m\x1b[48:2::255:255:255m  \x1b[49m\n"
            "\x1b[48:2::32:32:32m  \x1b[49m\x1b[48:2::255:255:255m  \x1b[49m\n"
            "\x1b[48:2::64:64:64m  \x1b[49m\x1b[48:2::255:255:255m  \x1b[49m\n"
            "\x1b[48:2::96:96:96m  \x1b[49m  \n"
            "\x1b[48:2::128:128:128m  \x1b[49m  \n"
            "\x1b[48:2::159:159:159m  \x1b[49m  \n"
            "\x1b[48:2::191:191:191m  \x1b[49m  \n"
            "\x1b[48:2::223:223:223m  \x1b[49m  \n\n"
        )


def test_background_colour_string():
    assert background_colour_string("#123456") == "\x1b[48:2::18:52:86m  \x1b[49m"
    assert background_colour_string("#123") == "\x1b[48:2::1:2:3m  \x1b[49m"


def test_make_rgb_alpha():
    assert make_rgb_alpha((1, 2, 3), 0.5) == (1.0, 1.5, 2.0)
    assert make_rgb_alpha((1, 2, 3), 0.5, (0.5, 0.5, 0.5)) == (0.75, 1.25, 1.75)


@pytest.mark.parametrize("pal", [ColorPalette({"C1": "#000000"}), "#000000"])
def test_make_alpha_palette(pal):
    new_pal = make_alpha_palette(pal, 3)
    assert [colors.to_hex(p) for p in new_pal._palette] == [
        "#000000",
        "#555555",
        "#aaaaaa",
    ]
    new_pal = make_alpha_palette(pal, 3, "red")
    assert [colors.to_hex(p) for p in new_pal._palette] == [
        "#000000",
        "#550000",
        "#aa0000",
    ]
