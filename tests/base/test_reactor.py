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

import time
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest

from bluemira.base.builder import ComponentManager
from bluemira.base.error import ComponentError
from bluemira.base.parameter_frame._parameter import Parameter
from bluemira.base.reactor import Reactor
from bluemira.builders.plasma import Plasma, PlasmaBuilder, PlasmaBuilderParams
from bluemira.geometry.tools import make_polygon
from bluemira.materials.material import Void

REACTOR_NAME = "My Reactor"


class TFCoil(ComponentManager):
    """
    This component manager is purely for testing the reactor is still
    valid if this is not set, so we don't need to implement it.
    """


class MyReactor(Reactor):
    SOME_CONSTANT: str = "not a component"

    plasma: Plasma
    tf_coil: TFCoil


class TestReactor:
    @classmethod
    def setup_class(cls):
        cls.reactor = cls._make_reactor()

    @classmethod
    def teardown_method(cls):
        plt.close("all")

    def test_time_since_init(self):
        a = self.reactor.time_since_init()
        time.sleep(0.1)
        b = self.reactor.time_since_init()
        assert a > 0
        assert b >= a + 0.1

    def test_name_set_on_root_component(self):
        assert self.reactor.component().name == REACTOR_NAME

    def test_unset_components_are_not_part_of_component_tree(self):
        # Only the plasma component is in the tree, as we haven't set the
        # TF coil
        assert len(self.reactor.component().children) == 1

    def test_component_tree_built_from_class_properties(self):
        assert self.reactor.plasma.component().name == "Plasma"

    @pytest.mark.parametrize("dim", ["xz", "xy", "xyz", ("xy", "xz")])
    def test_show_cad_displays_all_components(self, dim):
        with patch("bluemira.display.displayer.show_cad") as mock_show:
            if isinstance(dim, tuple):
                self.reactor.show_cad(*dim)
            else:
                self.reactor.show_cad(dim)

        assert (
            len(mock_show.call_args[0][0]) == len(dim) if isinstance(dim, tuple) else 1
        )

    @pytest.mark.parametrize("bad_dim", ["not_a_dim", 1, ["x"]])
    def test_ComponentError_given_invalid_plotting_dimension(self, bad_dim):
        with pytest.raises(ComponentError):
            self.reactor.show_cad(dim=bad_dim)

    @pytest.mark.parametrize("dim", ["xz", "xy", ("xy", "xz")])
    def test_plot_displays_all_components(self, dim):
        if isinstance(dim, tuple):
            self.reactor.plot(*dim)
        else:
            self.reactor.plot(dim)

    @pytest.mark.parametrize("bad_dim", ["i", 1, ["x"]])
    def test_ComponentError_given_invalid_plot_dimension_plot(self, bad_dim):
        with pytest.raises(ComponentError):
            self.reactor.plot(bad_dim)

    @pytest.mark.parametrize("material_filter", [True, False])
    @pytest.mark.parametrize("dim", ["xz", "xy", "xyz", ("xy", "xz")])
    def test_reactor_doesnt_show_void_material_by_default(self, dim, material_filter):
        reactor = self._make_reactor()

        if isinstance(dim, tuple):
            for d in dim:
                reactor.plasma.component().get_component(d).get_component(
                    "LCFS"
                ).material = Void("test")
        else:
            reactor.plasma.component().get_component(dim).get_component(
                "LCFS"
            ).material = Void("test")

        with patch("bluemira.display.displayer.show_cad") as mock_show:
            if isinstance(dim, tuple):
                if material_filter:
                    reactor.show_cad(*dim)
                else:
                    reactor.show_cad(*dim, _filter=None)

            elif material_filter:
                reactor.show_cad(dim)
            else:
                reactor.show_cad(dim, _filter=None)

        assert (
            len(mock_show.call_args[0][0]) == 0
            if material_filter
            else len(dim)
            if isinstance(dim, tuple)
            else 1
        )

    def test_save_cad(self, tmp_path):
        self.reactor.save_cad("xyz", directory=tmp_path)
        assert Path(tmp_path, f"{REACTOR_NAME}.stp").is_file()

    @staticmethod
    def _make_reactor() -> MyReactor:
        reactor = MyReactor(REACTOR_NAME, n_sectors=1)
        # use a square plasma, as circle causes topological naming issue
        lcfs = make_polygon({"x": [1, 1, 5, 5], "z": [-2, 2, 2, -2]}, closed=True)
        reactor.plasma = Plasma(
            PlasmaBuilder(
                PlasmaBuilderParams(n_TF=Parameter(name="n_TF", value=1)),
                {},
                lcfs,
            ).build()
        )
        return reactor


class TestComponentMananger:
    def setup_method(self):
        lcfs = make_polygon({"x": [1, 1, 5, 5], "z": [-2, 2, 2, -2]}, closed=True)

        self.p_comp = PlasmaBuilder(
            PlasmaBuilderParams(n_TF=Parameter(name="n_TF", value=1)),
            {},
            lcfs,
        ).build()

        self.plasma = Plasma(self.p_comp)

    def test_tree_contains_components(self):
        plasmatree = self.plasma.tree()
        assert all(dim in plasmatree for dim in ("xz", "xy", "xyz"))

    def test_save_cad(self, tmp_path):
        self.plasma.save_cad("xyz", directory=tmp_path)
        assert Path(tmp_path, "Plasma.stp").is_file()

    @pytest.mark.parametrize("dim", ["xz", "xy", "xyz", ("xy", "xz")])
    def test_show_cad_contains_components(self, dim):
        with patch("bluemira.display.displayer.show_cad") as mock_show:
            if isinstance(dim, tuple):
                self.plasma.show_cad(*dim)
            else:
                self.plasma.show_cad(dim)

        assert (
            len(mock_show.call_args[0][0]) == len(dim) if isinstance(dim, tuple) else 1
        )

    @pytest.mark.parametrize("material_filter", [True, False])
    @pytest.mark.parametrize("dim", ["xz", "xy", "xyz", ("xy", "xz")])
    def test_show_cad_ignores_void_by_default(self, dim, material_filter):
        p_comp = deepcopy(self.p_comp)
        if isinstance(dim, tuple):
            for d in dim:
                p_comp.get_component(d).get_component("LCFS").material = Void("test")
        else:
            p_comp.get_component(dim).get_component("LCFS").material = Void("test")

        plasma = Plasma(p_comp)

        with patch("bluemira.display.displayer.show_cad") as mock_show:
            if isinstance(dim, tuple):
                plasma.show_cad(*dim)
            else:
                plasma.show_cad(dim)

            if isinstance(dim, tuple):
                if material_filter:
                    plasma.show_cad(*dim)
                else:
                    plasma.show_cad(*dim, _filter=None)

            elif material_filter:
                plasma.show_cad(dim)
            else:
                plasma.show_cad(dim, _filter=None)

        assert (
            len(mock_show.call_args[0][0]) == 0
            if material_filter
            else len(dim)
            if isinstance(dim, tuple)
            else 1
        )

    @pytest.mark.parametrize("dim", ["xz", "xy", ("xy", "xz")])
    def test_plot_displays_all_components(self, dim):
        if isinstance(dim, tuple):
            self.plasma.plot(*dim)
        else:
            self.plasma.plot(dim)
