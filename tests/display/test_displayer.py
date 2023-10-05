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
Tests for the displayer module.
"""
import logging
from dataclasses import asdict
from unittest.mock import Mock, patch

import numpy as np
import pytest

import bluemira.codes._freecadapi as cadapi
from bluemira.base.components import Component, PhysicalComponent
from bluemira.display import displayer
from bluemira.display.error import DisplayError
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import extrude_shape, make_circle, make_polygon

_FREECAD_REF = "bluemira.codes._freecadapi"


def _skip_polyscope():
    try:
        import polyscope  # noqa: F401
    except ImportError:
        return True
    else:
        return False


class TestDisplayCADOptions:
    def test_default_options(self):
        """
        Check the values of the default options correspond to the global defaults.
        """
        the_options = displayer.DisplayCADOptions()
        d_dict = asdict(displayer.get_default_options())
        assert the_options.as_dict() == d_dict
        assert the_options.as_dict() is not displayer.get_default_options()

    def test_options(self):
        """
        Check the values can be set by passing kwargs.
        """
        the_options = displayer.DisplayCADOptions(color=(1.0, 0.0, 0.0))
        options_dict = the_options.as_dict()
        for key, val in options_dict.items():
            assert val == getattr(the_options, key)

    def test_modify_options(self):
        """
        Check the display options can be modified by passing in keyword args.
        """
        the_options = displayer.DisplayCADOptions()
        the_options.modify(color=(1.0, 0.0, 0.0))
        options_dict = the_options.as_dict()
        for key, val in options_dict.items():
            assert val == getattr(the_options, key)

    def test_properties(self):
        """
        Check the display option properties can be accessed
        """
        the_options = displayer.DisplayCADOptions()
        for key, val in asdict(displayer.get_default_options()).items():
            assert getattr(the_options, key) == val

        the_options.colour = (1.0, 0.0, 0.0)
        assert the_options.colour != displayer.get_default_options().colour
        assert the_options.transparency == displayer.get_default_options().transparency

        the_options.transparency = 0.1
        assert the_options.colour != displayer.get_default_options().colour
        assert the_options.transparency != displayer.get_default_options().transparency


class TestComponentDisplayer:
    @pytest.mark.parametrize("colour", [(1.0, 0.0, 0.0), BLUE_PALETTE["VV"]])
    @pytest.mark.parametrize(
        "viewer",
        [
            "freecad",
            pytest.param(
                "polyscope",
                marks=pytest.mark.skipif(_skip_polyscope(), reason="Not installed"),
            ),
        ],
    )
    def test_display(self, colour, viewer):
        square_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ]
        )

        wire1 = make_polygon(square_points, closed=True)
        wire2 = make_polygon(square_points + 1.0, closed=True)

        group = Component("Parent")
        child1 = PhysicalComponent("Child1", shape=wire1, parent=group)
        child1.display_cad_options.colour = (0.0, 1.0, 0.0)
        child2 = PhysicalComponent("Child2", shape=wire2, parent=group)

        child1.show_cad(backend=viewer)
        group.show_cad(backend=viewer)
        child2.display_cad_options = displayer.DisplayCADOptions(colour=colour)
        group.show_cad(backend=viewer)
        group.show_cad(colour=(0.0, 0.0, 1.0), backend=viewer)
        displayer.ComponentDisplayer().show_cad(group, color=colour, backend=viewer)

        with pytest.raises(DisplayError):
            child2.display_cad_options = (0.0, 0.0, 1.0)


class TestGeometryDisplayer:
    square_points = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    )

    def setup_method(self):
        displayer.ViewerBackend.get_module.cache_clear()

    @pytest.mark.parametrize(
        "viewer",
        [
            "freecad",
            pytest.param(
                "polyscope",
                marks=pytest.mark.skipif(_skip_polyscope(), reason="Not installed"),
            ),
        ],
    )
    def test_display(self, viewer):
        wire1 = make_polygon(self.square_points, label="wire1", closed=False)
        box1 = extrude_shape(wire1, vec=(0.0, 0.0, 1.0), label="box1")

        displayer.show_cad(wire1, backend=viewer)
        displayer.show_cad(
            box1,
            displayer.DisplayCADOptions(colour=(1.0, 0.0, 1.0)),
            backend=viewer,
        )
        displayer.show_cad([wire1, box1], backend=viewer)
        displayer.show_cad(
            [wire1, box1],
            displayer.DisplayCADOptions(colour=(1.0, 0.0, 1.0)),
            ["name", "name2"],
            backend=viewer,
        )
        displayer.show_cad(
            [wire1, box1],
            [
                displayer.DisplayCADOptions(colour=(1.0, 0.0, 0.0)),
                displayer.DisplayCADOptions(colour=(0.0, 1.0, 0.0), transparency=0.2),
            ],
            backend=viewer,
        )
        displayer.show_cad(
            [wire1, box1],
            color=(1.0, 0.0, 0.0),
            transparency=0.2,
            backend=viewer,
        )

        with pytest.raises(DisplayError):
            displayer.show_cad(
                wire1,
                [
                    displayer.DisplayCADOptions(colour=(1.0, 0.0, 0.0)),
                    displayer.DisplayCADOptions(colour=(0.0, 1.0, 0.0)),
                ],
                backend=viewer,
            )

    def _make_shape(self):
        circle_wire = make_circle(radius=5, axis=(0, 0, 1), label="my_wire")
        circle_face = BluemiraFace(circle_wire, label="my_face")
        return extrude_shape(circle_face, vec=(0, 0, 10), label="my_solid")

    @pytest.mark.parametrize(
        ("labels", "result"),
        [
            ("name", ["name", "name"]),
            (["name", "name"], ["name", "name"]),
            ("", ["", ""]),
            (None, ["", ""]),
        ],
    )
    def test_labels_passed_in_correctly(self, labels, result):
        with patch(f"{_FREECAD_REF}.show_cad") as show_cad_mock:
            displayer.show_cad([self._make_shape(), self._make_shape()], labels=labels)

        assert show_cad_mock.call_args_list[0][0][2] == result

    @pytest.mark.parametrize(
        "viewer",
        [
            "freecad",
            pytest.param(
                "polyscope",
                marks=pytest.mark.skipif(_skip_polyscope(), reason="Not installed"),
            ),
        ],
    )
    def test_3d_cad_displays_shape(self, viewer):
        displayer.show_cad(self._make_shape(), backend=viewer)

    @pytest.mark.parametrize(
        "mock",
        [
            Mock(side_effect=[FileNotFoundError(), cadapi]),
            Mock(side_effect=[ModuleNotFoundError(), cadapi]),
        ],
    )
    def test_no_displayer(self, mock, caplog):
        with patch("bluemira.display.displayer.get_module", mock):
            displayer.show_cad(self._make_shape(), backend="polyscope")
        assert len(caplog.messages) == 1
        with patch("bluemira.display.displayer.get_module", mock):
            displayer.show_cad(self._make_shape(), backend="polyscope")
        assert len(caplog.messages) == 1

    def test_unknown_displayer(self, caplog):
        caplog.set_level(logging.WARNING)
        displayer.show_cad(self._make_shape(), backend="mybackend")

        assert len(caplog.messages) == 1
