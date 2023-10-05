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

import pytest
from anytree import PreOrderIter

from bluemira.base.components import (
    Component,
    MagneticComponent,
    PhysicalComponent,
    get_properties_from_components,
)
from bluemira.base.error import ComponentError


class TestComponentClass:
    """
    Tests for the base Component functionality.
    """

    def test_tree(self):
        target_tree = """Parent (Component)
└── Child (Component)
    └── Grandchild (Component)"""

        child = Component("Child")
        parent = Component("Parent", children=[child])
        grandchild = Component("Grandchild", parent=child)
        assert parent.tree() == target_tree

        root: Component = grandchild.root
        assert root.tree() == target_tree

    def test_get_component_full_tree(self):
        parent = Component("Parent")
        child1 = Component("Child1", parent=parent)
        child2 = Component("Child2", parent=parent)
        grandchild = Component("Grandchild", parent=child1)

        assert grandchild.get_component("Child2", full_tree=True) is child2

    def test_get_component_from_node(self):
        parent = Component("Parent")
        child1 = Component("Child1", parent=parent)
        child2 = Component("Child2", parent=parent)
        grandchild = Component("Grandchild", parent=child1)

        assert grandchild.get_component("Child2") is None

    def test_get_component_multiple_full_tree(self):
        parent = Component("Parent")
        child1 = Component("Child1", parent=parent)
        child2 = Component("Child2", parent=parent)
        relative = Component("relative", parent=child1)
        grandchild1 = Component("Grandchild", parent=child1)
        grandchild2 = Component("Grandchild", parent=child2)

        components = relative.get_component("Grandchild", first=False, full_tree=True)
        assert len(components) == 2
        assert components[0] is not components[1]
        assert components[0].parent.parent == components[1].parent.parent

    def test_get_component_multiple_from_node(self):
        parent = Component("Parent")
        child1 = Component("Child1", parent=parent)
        child2 = Component("Child2", parent=parent)
        grandchild1 = Component("Grandchild", parent=child1)
        grandchild2 = Component("Grandchild", parent=child2)

        components = child1.get_component("Grandchild", first=False)
        assert len(components) == 1
        assert components[0] is grandchild1

        components = child2.get_component("Grandchild", first=False)
        assert len(components) == 1
        assert components[0] is grandchild2

    def test_get_component_missing(self):
        parent = Component("Parent")
        child = Component("Child", parent=parent)
        grandchild = Component("Grandchild", parent=child)

        component = grandchild.get_component("Banana")
        assert component is None

    def test_add_child(self):
        parent = Component("Parent")
        child = Component("Child")

        parent.add_child(child)
        assert parent.children == (child,)

    def test_fail_add_duplicate_child(self):
        parent = Component("Parent")
        child = Component("Child", parent=parent)

        with pytest.raises(ComponentError):
            parent.add_child(child)

    def test_add_children(self):
        parent = Component("Parent")
        child1 = Component("Child1")
        child2 = Component("Child2")

        parent.add_children([child1, child2])
        assert parent.children == (child1, child2)

    def test_fail_add_duplicate_children(self):
        parent = Component("Parent")
        child1 = Component("Child1", parent=parent)
        child2 = Component("Child2")

        with pytest.raises(ComponentError):
            parent.add_children([child1, child2])

    def test_prune_child_removes_node_with_given_name(self):
        parent = Component("Parent")
        Component("Child1", parent=parent)
        Component("Child2", parent=parent)

        parent.prune_child("Child1")

        assert parent.get_component("Child1") is None
        assert parent.get_component("Child2") is not None

    def test_prune_child_does_nothing_if_node_does_not_exist(self):
        parent = Component("Parent")
        Component("Child1", parent=parent)

        parent.prune_child("not_a_child")

        assert parent.get_component("Child1") is not None

    def test_add_child_ComponentError_given_duplicated_component_name(self):
        parent = Component("TFCoils")
        parent.add_child(Component("Sector 1"))

        with pytest.raises(ComponentError):
            parent.add_child(Component("Sector 1"))

    def test_add_child_with_same_name_different_level(self):
        parent = Component("TFCoils")
        parent.add_child(Component("Sector 1"))

        parent.add_child(Component("Othersector", children=[Component("Sector 1")]))

        assert len(parent.children) == 2
        assert len(parent.descendants) == 3

    def test_init_with_same_name_different_level(self):
        parent = Component("TFCoils")
        Component("Sector 1", parent=parent)
        Component("Other Sector", parent=parent, children=[Component("Sector 1")])

        assert len(parent.children) == 2
        assert len(parent.descendants) == 3

    def test_init_ComponentError_given_duplicated_component_name(self):
        parent = Component("TFCoils")
        Component("Sector 1", parent=parent)

        with pytest.raises(ComponentError):
            Component("Sector 1", parent=parent)

    def test_copy_does_fully_copy(self):
        parent = Component("Parent")
        child1 = Component("Child1", parent=parent)
        child2 = PhysicalComponent(
            "Child2", parent=parent, shape="A shape", material="A material"
        )
        Component("GrandchildAA", parent=child1)
        Component("GrandchildAB", parent=child1)
        PhysicalComponent(
            "GrandchildBA", parent=child2, shape="B shape", material="B material"
        )
        Component("GrandchildBB", parent=child2)

        parent_copy = parent.copy()

        def get_comp_copy_and_compare(comp: Component):
            cpy = parent_copy.get_component(comp.name)

            # test instance
            assert cpy is not None
            assert cpy is not comp

            # test parent
            if cpy.parent:
                # assert cpy.parent is not comp.parent
                assert cpy.parent.name == comp.parent.name

            # test children
            if cpy.children:
                comp_children_names = [c.name for c in comp.children]
                for c in cpy.children:
                    assert c.name in comp_children_names

            # test properties
            assert cpy._plot_options is comp._plot_options
            assert cpy._display_cad_options is comp._display_cad_options

            if isinstance(comp, PhysicalComponent):
                assert isinstance(cpy, PhysicalComponent)
                # assert they are the same instance
                assert cpy.shape is comp.shape
                assert cpy.material is comp.material

        [get_comp_copy_and_compare(c) for c in PreOrderIter(parent)]

    def test_filter_components_does_filter_on_single(self):
        parent = Component("Parent")
        child1 = Component("Child1", parent=parent)
        child2 = Component("Child2", parent=parent)

        child1a = Component("child1A", parent=child1)
        child1b = Component("child1B", parent=child1)

        def attach_dims_and_physical_comps_to(comp: Component):
            xy = Component("xy", parent=comp)
            xz = Component("xz", parent=comp)
            xyz = Component("xyz", parent=comp)

            PhysicalComponent(
                "pc_xy",
                parent=xy,
                shape="pc_xy shape",
                material="pc_xy material",
            )
            PhysicalComponent(
                "pc_xz",
                parent=xz,
                shape="pc_xz shape",
                material="pc_xz material",
            )
            PhysicalComponent(
                "pc_xyz",
                parent=xyz,
                shape="pc_xyz shape",
                material="pc_xyz material",
            )

        attach_dims_and_physical_comps_to(child1a)
        attach_dims_and_physical_comps_to(child1b)
        attach_dims_and_physical_comps_to(child2)

        parent.filter_components(["xz"])

        xy = parent.get_component("xy")
        xz = parent.get_component("xz")
        xyz = parent.get_component("xyz")

        assert xy is None
        assert type(xz) is Component
        assert xyz is None

    def test_filter_components_does_filter_on_double(self):
        parent = Component("Parent")
        child1 = Component("Child1", parent=parent)
        child2 = Component("Child2", parent=parent)

        child1a = Component("child1A", parent=child1)
        child1b = Component("child1B", parent=child1)

        def attach_dims_and_physical_comps_to(comp: Component):
            xy = Component("xy", parent=comp)
            xz = Component("xz", parent=comp)
            xyz = Component("xyz", parent=comp)

            PhysicalComponent(
                "pc_xy",
                parent=xy,
                shape="pc_xy shape",
                material="pc_xy material",
            )
            PhysicalComponent(
                "pc_xz",
                parent=xz,
                shape="pc_xz shape",
                material="pc_xz material",
            )
            PhysicalComponent(
                "pc_xyz",
                parent=xyz,
                shape="pc_xyz shape",
                material="pc_xyz material",
            )

        attach_dims_and_physical_comps_to(child1a)
        attach_dims_and_physical_comps_to(child1b)
        attach_dims_and_physical_comps_to(child2)

        parent.filter_components(["xy", "xz"])

        xy = parent.get_component("xy")
        xz = parent.get_component("xz")
        xyz = parent.get_component("xyz")

        assert type(xy) is Component
        assert type(xz) is Component
        assert xyz is None


class TestPhysicalComponent:
    """
    Tests for the PhysicalComponent class.
    """

    def test_get_component_property(self):
        component = PhysicalComponent("Dummy", shape="A shape")
        shape = component.get_component_properties("shape")
        assert shape == "A shape"

    def test_get_component_properties(self):
        component = PhysicalComponent("Dummy", shape="A shape", material="A material")
        shape, material = component.get_component_properties(("shape", "material"))
        assert shape == "A shape"
        assert material == "A material"

    @pytest.mark.parametrize(
        ("first", "result"),
        [
            (True, ("A shape", "A material")),
            (False, (["A shape"] * 2, ["A material"] * 2)),
        ],
    )
    def test_get_components_properties(self, first, result):
        parent = Component("Parent")
        component = PhysicalComponent(
            "Dummy1", shape="A shape", material="A material", parent=parent
        )
        component2 = PhysicalComponent(
            "Dummy2", shape="A shape", material="A material", parent=parent
        )
        shape, material = parent.get_component_properties(
            ("shape", "material"), first=first
        )
        assert shape == result[0]
        assert material == result[1]

    def test_shape(self):
        component = PhysicalComponent("Dummy", shape="A shape")
        assert component.shape == "A shape"

    def test_material_default(self):
        component = PhysicalComponent("Dummy", shape="A shape")
        assert component.material is None

    def test_material(self):
        component = PhysicalComponent("Dummy", shape="A shape", material="A material")
        assert component.material == "A material"


class TestMagneticComponent:
    """
    Tests for the MagneticComponent class.
    """

    def test_shape(self):
        component = MagneticComponent("Dummy", shape="A shape")
        assert component.shape == "A shape"

    def test_conductor_default(self):
        component = MagneticComponent("Dummy", shape="A shape")
        assert component.material is None

    def test_conductor(self):
        component = MagneticComponent("Dummy", shape="A shape", conductor="A conductor")
        assert component.conductor == "A conductor"


class TestGetProperties:
    """
    Tests for the get_properties_from_components function
    """

    def test_get_properties_from_components_single_single(self):
        comps = MagneticComponent("Dummy", shape="A shape")

        shape = get_properties_from_components(comps, "shape")

        assert shape == "A shape"

    def test_get_properties_from_components_complex(self):
        parent = Component("Parent")
        component = PhysicalComponent(
            "Dummy", shape="A shape", material="A material", parent=parent
        )
        component2 = PhysicalComponent(
            "Dummy1", shape="A shape", material="A material", parent=parent
        )
        component3 = PhysicalComponent("Dummy", shape="A shape", material="A material")
        component4 = Component("Dummy")
        shape, material = get_properties_from_components(
            [parent, component3, component4], ("shape", "material")
        )
        assert shape == ["A shape"] * 3
        assert material == ["A material"] * 3
