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

from copy import deepcopy

import numpy as np
import pytest
from matplotlib import pyplot as plt

from bluemira.base.constants import ANSI_COLOR
from bluemira.structural.crosssection import IBeam, RectangularBeam
from bluemira.structural.error import StructuralError
from bluemira.structural.loads import LoadCase
from bluemira.structural.material import SS316
from bluemira.structural.model import FiniteElementModel, check_matrix_condition


def test_illconditioned():
    # http://www.ti3.tu-harburg.de/paper/rump/NiRuOi11.pdf
    k1 = np.array([[1, -6, 7, -9], [1, -5, 0, 0], [0, 1, -5, 0], [0, 0, 1, -5]])
    k2 = np.array(
        [[17, -864, 716, -799], [1, -50, 0, 0], [0, 1, -50, 0], [0, 0, 1, -50]]
    )
    orange = ANSI_COLOR["orange"]
    for k, digits in [
        [k1, 3],
        [k2, 9],
    ]:
        with pytest.raises(StructuralError):
            check_matrix_condition(k, digits)

    for k, digits in [
        [k1, 5],
        [k2, 10],
    ]:
        check_matrix_condition(k, digits)


class TestFEModel:
    @pytest.mark.parametrize(("sup", "dof"), zip([False, True], [5, 6]))
    def test_errors(self, sup, dof):
        """
        Checks rigid-body motion detection (insufficient constraints)
        """
        model = FiniteElementModel()
        i_beam = IBeam(0.5, 0.8, 0.2, 0.2)

        length = 10
        p3 = -110
        m4 = -40
        m6 = -60
        model.add_node(0, 0, 0)
        model.add_node(length, 0, 0)
        model.add_node(2 * length, 0, 0)

        model.add_element(0, 1, i_beam, SS316)
        model.add_element(1, 2, i_beam, SS316)
        model.add_support(0, True, True, True, False, False, False)
        model.add_support(2, sup, True, True, False, False, False)
        model.find_supports()
        assert model.n_fixed_dofs == dof
        assert model.geometry.n_nodes == 3
        assert model.geometry.n_elements == 2
        load_case = LoadCase()
        load_case.add_node_load(1, p3, "Fy")
        load_case.add_node_load(1, m4, "Mz")
        load_case.add_node_load(2, m6, "Mz")
        with pytest.raises(StructuralError):
            model.solve(load_case)

    def test_610(self):
        """
        Example from the Przemieniecki book, Chapter 6, section 10
        NOTE: without the temperature distribution for now
        """
        model = FiniteElementModel()
        # Need to assume some numbers to re-derive the analytical solution

        i_beam = IBeam(0.5, 0.8, 0.2, 0.2)
        length = 10
        p3 = -110
        m4 = -40
        m6 = -60
        model.add_node(0, 0, 0)
        model.add_node(length, 0, 0)
        model.add_node(2 * length, 0, 0)
        model.add_element(0, 1, i_beam, SS316)
        model.add_element(1, 2, i_beam, SS316)
        model.add_support(0, True, True, True, True, True, True)
        model.add_support(2, False, True, True, False, False, False)

        e_mat, i_xs = SS316.E, i_beam.i_zz
        # Check element lengths
        for element in model.geometry.elements:
            assert element.length == length

        d = model.geometry.nodes[0].distance_to_other(model.geometry.nodes[-1])
        # Check model length
        assert d == 2 * length

        model.find_supports()
        assert model.n_fixed_dofs == 8
        assert model.geometry.n_nodes == 3
        assert model.geometry.n_elements == 2
        load_case = LoadCase()
        load_case.add_node_load(1, p3, "Fy")
        load_case.add_node_load(1, m4, "Mz")
        load_case.add_node_load(2, m6, "Mz")
        # Check individual element stiffness matrices
        k = [element.k_matrix_glob for element in model.geometry.elements]

        def treat_matrix(k_matrix):
            # Get rid of all useless rows and columns (for this problem)
            for i in [0, 2, 3, 4, 6, 8, 9, 10][::-1]:
                k_matrix = np.delete(k_matrix, i, axis=0)
                k_matrix = np.delete(k_matrix, i, axis=1)
            return k_matrix

        k1 = treat_matrix(k[0])

        k1a = (e_mat * i_xs / length**3) * np.array(
            [
                [12, 6 * length, -12, 6 * length],
                [6 * length, 4 * length**2, -6 * length, 2 * length**2],
                [-12, -6 * length, 12, -6 * length],
                [6 * length, 2 * length**2, -6 * length, 4 * length**2],
            ]
        )
        assert np.allclose(k1, k1a)
        k2 = treat_matrix(k[1])
        assert np.allclose(k2, k1a)

        result = model.solve(load_case)
        deflections = result.deflections

        # Check that there are three displacements
        non_zero = np.nonzero(deflections)[0]
        assert len(non_zero) == 3

        # Check that the displacements occur in the correct places
        assert non_zero[0] == 7  # dy1
        assert non_zero[1] == 11  # mz1
        assert non_zero[2] == 17  # mz2

        # Solve the problem analytically

        k_a = (
            e_mat
            * i_xs
            / length**3
            * np.array(
                [
                    [24, 0, 6 * length],
                    [0, 8 * length**2, 2 * length**2],
                    [6 * length, 2 * length**2, 4 * length**2],
                ]
            )
        )
        p_a = np.array([p3, m4, m6])
        # Check that the displacements are of the right magnitude
        u346 = deflections[non_zero]
        u_analytical = np.linalg.solve(k_a, p_a)
        msg = "\n" + f"True: {u_analytical}" + "\n" + f"Model: {u346}"
        assert np.allclose(u346, u_analytical), msg


class TestCantilever:
    def setup_method(self):
        model = FiniteElementModel()
        length = 4

        rect_beam = RectangularBeam(0.05, 0.61867)

        model.add_node(0, 0, 0)
        model.add_node(-length, 0, 0)
        dummy_material = deepcopy(SS316)
        dummy_material.E = 10e9
        model.add_element(0, 1, rect_beam, dummy_material)
        model.add_support(0, True, True, True, True, True, True)
        self.model = model
        self.material = dummy_material

        self.cross_section = rect_beam
        self.length = length

    def teardown_method(self):
        self.model.clear_loads()

    def test_single_load(self):
        load = -1000
        length = 4
        b = 3
        e_mat = self.material.E
        i_xs = self.cross_section.i_yy
        load_case = LoadCase()
        load_case.add_element_load(0, load, b / length, "Fz")

        result = self.model.solve(load_case)
        deflections = result.deflections
        end_deflection = deflections[6 + 2]
        analytical = load * b**2 / (6 * e_mat * i_xs) * (3 * length - b)
        assert np.isclose(end_deflection, analytical)

    def test_end_load(self):
        load = 8000
        length = 34
        e_mat = self.material.E
        i_xs = self.cross_section.i_yy
        end_deflection = load * length**3 / (3 * e_mat * i_xs)

        for node_coords in [[length, 0, 0]]:
            model = FiniteElementModel()
            model.add_node(0, 0, 0)
            model.add_node(*node_coords)
            model.add_element(0, 1, self.cross_section, self.material)
            model.add_support(0, True, True, True, True, True, True)
            load_case = LoadCase()

            load_case.add_node_load(1, load, "Fz")

            result = model.solve(load_case)
            deflections = result.deflections

            end_deflection1 = model.geometry.nodes[1].displacements[2]
            end_deflection2 = deflections[6 + 2]
            # Check the displacement has correctly been mapped to the Node
            assert end_deflection1 == end_deflection2

            # Check the deflection is the same as expected
            assert np.isclose(end_deflection1, end_deflection)

    def test_end_moment(self):
        load = -41278
        length = 8
        e_mat = self.material.E
        i_xs = self.cross_section.i_yy
        end_deflection = -load * length**2 / (2 * e_mat * i_xs)

        for node_coords in [[length, 0, 0]]:
            model = FiniteElementModel()
            model.add_node(0, 0, 0)
            model.add_node(*node_coords)
            model.add_element(0, 1, self.cross_section, self.material)
            model.add_support(0, True, True, True, True, True, True)
            load_case = LoadCase()

            load_case.add_node_load(1, load, "My")

            result = model.solve(load_case)
            deflections = result.deflections

            end_deflection1 = model.geometry.nodes[1].displacements[2]
            end_deflection2 = deflections[6 + 2]
            # Check the displacement has correctly been mapped to the Node
            assert end_deflection1 == end_deflection2

            # Check the deflection is the same as expected
            assert np.isclose(end_deflection1, end_deflection)

    def test_dual_load(self):
        p1 = -2000
        p2 = -4000
        b = 2
        load_case = LoadCase()
        load_case.add_node_load(1, p2, "Fz")
        load_case.add_element_load(0, p1, b / self.length, "Fz")
        result = self.model.solve(load_case)
        deflections = result.deflections
        end_deflection = deflections[6 + 2]
        analytical = -0.01
        assert np.isclose(end_deflection, analytical)

    def test_axes(self):
        """
        Simple cantilever problem rotation about Z-axis with Z load
        """
        length = 4

        load = -1000

        b = 3
        rect_beam = RectangularBeam(0.1, 0.1)
        dummy_material = deepcopy(SS316)
        dummy_material.E = 10e9

        e_mat = dummy_material.E
        i_xs = rect_beam.i_yy

        for node_coords in [
            [length, 0, 0],
            [0, length, 0],
            [-length, 0, 0],
            [0, -length, 0],
        ]:
            model = FiniteElementModel()
            model.add_node(0, 0, 0)
            model.add_node(*node_coords)
            model.add_element(0, 1, rect_beam, dummy_material)
            model.add_support(0, True, True, True, True, True, True)
            load_case = LoadCase()

            load_case.add_element_load(0, load, b / length, "Fz")

            result = model.solve(load_case)
            deflections = result.deflections

            end_deflection1 = model.geometry.nodes[1].displacements[2]
            end_deflection2 = deflections[6 + 2]
            # Check the displacement has correctly been mapped to the Node
            assert end_deflection1 == end_deflection2

            analytical = load * b**2 / (6 * e_mat * i_xs) * (3 * length - b)

            # Check the deflection is the same as expected
            assert np.isclose(end_deflection1, analytical)

    def test_node_load(self):
        load = 40000
        e_mat = self.material.E
        i_xs = self.cross_section.i_yy
        load_case = LoadCase()
        load_case.add_node_load(1, load, "Fz")
        result = self.model.solve(load_case)
        deflections = result.deflections
        end_deflection = load * self.length**3 / (3 * e_mat * i_xs)
        assert np.isclose(deflections[6 + 2], end_deflection)

    def test_neutral_load(self):
        load = 40000

        load_case = LoadCase()
        load_case.add_node_load(1, load, "Fz")
        load_case.add_node_load(1, -load, "Fz")
        result = self.model.solve(load_case)
        deflections = result.deflections
        end_deflection = 0
        assert np.isclose(deflections[6 + 2], end_deflection)

    def test_mixed_neutral_load(self):
        load = 40000

        load_case = LoadCase()
        load_case.add_node_load(1, load, "Fz")
        load_case.add_element_load(0, -load, 1, "Fz")
        result = self.model.solve(load_case)
        deflections = result.deflections
        end_deflection = 0
        assert np.isclose(deflections[6 + 2], end_deflection)


class TestDistributedLoads:
    def test_fixed_fixed_load(self):
        # The middle node is just an easy way to get the deflection before
        # having implemented full-beam deflection mapping
        # It also shows that the constraints of a discretised beam are the
        # simplest to implement!

        length = 2
        dummy_material = deepcopy(SS316)
        rect_beam = RectangularBeam(0.05, 0.61867)
        w = 2000

        e_mat = dummy_material.E
        i_xs = rect_beam.i_yy

        for node_coords in [[length, 0, 0]]:
            model = FiniteElementModel()

            model.add_node(0, 0, 0)
            model.add_node(*node_coords)
            node2 = 2 * np.array(node_coords)
            model.add_node(*node2)

            model.add_element(0, 1, rect_beam, dummy_material)
            model.add_element(1, 2, rect_beam, dummy_material)
            model.add_support(0, True, True, True, True, True, True)
            model.add_support(2, True, True, True, True, True, True)

            load_case = LoadCase()
            load_case.add_distributed_load(0, w, "Fz")
            load_case.add_distributed_load(1, w, "Fz")

            result = model.solve(load_case)
            deflections = result.deflections
            mid_deflection = w * (2 * length) ** 4 / (384 * e_mat * i_xs)

            assert np.isclose(mid_deflection, deflections[6 + 2])
            fz_reaction = -w * (2 * length) / 2
            m_reaction = -w * (2 * length) ** 2 / 12

            fz1 = model.geometry.nodes[0].reactions[2]
            fz2 = model.geometry.nodes[2].reactions[2]
            my1 = model.geometry.nodes[0].reactions[4]
            my2 = model.geometry.nodes[2].reactions[4]

            assert np.isclose(fz1, fz_reaction)
            assert np.isclose(fz2, fz_reaction)
            assert np.isclose(my1, -m_reaction)
            assert np.isclose(my2, m_reaction)

    def test_cantilever_load(self):
        length = 4

        rect_beam = RectangularBeam(0.5, 0.7)
        w = -2000
        e_mat = SS316.E
        i_xs = rect_beam.i_yy

        end_deflection = w * length**4 / (8 * e_mat * i_xs)

        for node_coords in [
            [length, 0, 0],
            [0, length, 0],
            [-length, 0, 0],
            [0, -length, 0],
        ]:
            model = FiniteElementModel()

            model.add_node(0, 0, 0)
            model.add_node(*node_coords)

            model.add_element(0, 1, rect_beam, SS316)
            model.add_support(0, True, True, True, True, True, True)

            load_case = LoadCase()
            load_case.add_distributed_load(0, w, "Fz")

            result = model.solve(load_case)
            deflections = result.deflections
            assert np.isclose(end_deflection, deflections[6 + 2])


class TestLFrame:
    # https://structx.com/Frame_Formulas_017.html
    def setup_method(self):
        length = 4
        height = 15
        model = FiniteElementModel()
        model.add_node(0, 0, 0)
        model.add_node(0, 0, height)
        model.add_node(length, 0, height)
        model.add_support(0, *[True] * 6)

        rect_beam = RectangularBeam(0.5, 0.5)
        model.add_element(0, 1, rect_beam, SS316)
        model.add_element(1, 2, rect_beam, SS316)
        self.length = length
        self.height = height
        self.e_mat = SS316.E
        self.i_xs = rect_beam.i_yy

        self.model = model

    def teardown_method(self):
        self.model.clear_loads()

    def test_udl(self):
        w = 1000
        load_case = LoadCase()
        load_case.add_distributed_load(1, w, "Fz")

        result = self.model.solve(load_case)
        deflections = result.deflections
        delta_cx = w * self.height**2 * self.length**2 / (4 * self.e_mat * self.i_xs)
        delta_cz = (
            w
            * self.length**3
            * (self.length + 4 * self.height)
            / (8 * self.e_mat * self.i_xs)
        )

        assert np.isclose(delta_cx, deflections[6 * 2], rtol=1e-2)
        assert np.isclose(delta_cz, deflections[6 * 2 + 2], rtol=1e-2)


@pytest.mark.longrun
class TestCompoundDeflection:
    def test_fixedfixed(self):
        length = 4

        rect_beam = RectangularBeam(0.7, 0.7)

        model = FiniteElementModel()
        model.add_node(0, 0, 0)
        model.add_node(length, 0, 0)
        model.add_node(2 * length, 0, 0)
        model.add_node(3 * length, 0, 0)
        model.add_node(4 * length, 0, 0)
        model.add_support(0, *[True] * 6)
        model.add_support(4, *[True] * 6)
        model.add_element(0, 1, rect_beam, SS316)
        model.add_element(1, 2, rect_beam, SS316)
        model.add_element(2, 3, rect_beam, SS316)
        model.add_element(3, 4, rect_beam, SS316)

        load = -10000
        load_case = LoadCase()
        load_case.add_distributed_load(0, load, "Fz")
        load_case.add_distributed_load(1, load, "Fz")
        load_case.add_distributed_load(2, load, "Fz")
        load_case.add_distributed_load(3, load, "Fz")

        load_case.add_distributed_load(0, load, "Fy")
        load_case.add_distributed_load(1, load, "Fy")
        load_case.add_distributed_load(2, load, "Fy")
        load_case.add_distributed_load(3, load, "Fy")

        result = model.solve(load_case)

        result.plot()

    def test_cantilever(self):
        length = 4

        rect_beam = RectangularBeam(0.7, 0.7)

        model = FiniteElementModel()
        model.add_node(0, 0, 0)
        model.add_node(length, 0, 0)
        model.add_node(2 * length, 0, 0)
        model.add_node(3 * length, 0, 0)
        model.add_node(4 * length, 0, 0)
        model.add_support(0, *[True] * 6)
        model.add_element(0, 1, rect_beam, SS316)
        model.add_element(1, 2, rect_beam, SS316)
        model.add_element(2, 3, rect_beam, SS316)
        model.add_element(3, 4, rect_beam, SS316)

        w_load = -10000
        load_case = LoadCase()
        load_case.add_distributed_load(0, w_load, "Fz")
        load_case.add_distributed_load(1, w_load, "Fz")
        load_case.add_distributed_load(2, w_load, "Fz")
        load_case.add_distributed_load(3, w_load, "Fz")

        load_case.add_distributed_load(0, w_load, "Fy")
        load_case.add_distributed_load(1, w_load, "Fy")
        load_case.add_distributed_load(2, w_load, "Fy")
        load_case.add_distributed_load(3, w_load, "Fy")

        result = model.solve(load_case)
        result.plot()
        plt.show()


@pytest.mark.longrun
class TestGravityLoads:
    def test_angled_cantilever(self):
        model = FiniteElementModel()
        model.add_node(0, 0, 0)
        model.add_support(0, *[True] * 6)

        rect_beam = RectangularBeam(1, 1)

        length = 10
        n = 50
        for i in range(1, n + 1):
            model.add_node(i * length / n, i * length / n, i * length / n)

            model.add_element(i - 1, i, rect_beam, SS316)

        model.add_gravity_loads()

        result = model.solve()
        result.plot()
        plt.show()

        # Check that tip displacements in the x and y directions are equal
        assert np.isclose(result.deflections[6 * n], result.deflections[6 * n + 1])


class TestFixedFixedStress:
    def test_stress(self):
        model = FiniteElementModel()

        w, h = 0.06, 0.2
        rect_beam = RectangularBeam(w, h)
        length = 4
        w = 5000
        model.add_node(0, 0, 0)
        model.add_node(length / 2, 0, 0)
        model.add_node(length, 0, 0)
        model.add_node(length + 1, 0, 0)
        model.add_element(0, 1, rect_beam, SS316)
        model.add_element(1, 2, rect_beam, SS316)
        model.add_element(2, 3, rect_beam, SS316)
        model.add_support(0, True, True, True, False, False, False)
        model.add_support(2, True, True, True, False, False, False)
        model.add_support(3, True, True, True, True, True, True)
        model.add_distributed_load(0, -w, "Fz")
        model.add_distributed_load(1, -w, "Fz")
        result = model.solve()

        result.plot()
        plt.show()


@pytest.mark.longrun
class TestMiniEiffelTower:
    @classmethod
    def setup_class(cls):
        model = FiniteElementModel()

        def make_platform(
            width, elevation, cross_section, elements=True, internodes=False
        ):
            if not internodes:
                model.add_node(-width / 2, -width / 2, elevation)
                i_id = model.geometry.nodes[-1].id_number
                model.add_node(width / 2, -width / 2, elevation)
                model.add_node(width / 2, width / 2, elevation)
                model.add_node(-width / 2, width / 2, elevation)

                if elements:
                    for j in range(3):
                        model.add_element(i_id + j, i_id + j + 1, cross_section, SS316)
                    model.add_element(i_id + j + 1, i_id, cross_section, SS316)
            if internodes:
                model.add_node(-width / 2, -width / 2, elevation)
                i_id = model.geometry.nodes[-1].id_number
                model.add_node(0, -width / 2, elevation)
                model.add_node(width / 2, -width / 2, elevation)
                model.add_node(width / 2, 0, elevation)
                model.add_node(width / 2, width / 2, elevation)
                model.add_node(0, width / 2, elevation)
                model.add_node(-width / 2, width / 2, elevation)
                model.add_node(-width / 2, 0, elevation)
                if elements:
                    for j in range(7):
                        model.add_element(i_id + j, i_id + j + 1, cross_section, SS316)
                    model.add_element(i_id + j + 1, i_id, cross_section, SS316)

        cs1 = RectangularBeam(3, 3)
        cs2 = RectangularBeam(1.5, 1.5)
        cs3 = RectangularBeam(1, 1)
        cs4 = RectangularBeam(0.5, 0.5)
        cs5 = RectangularBeam(0.25, 0.25)

        make_platform(124.9, 0, cs1, elements=False)
        for i in range(4):
            model.add_support(i, *6 * [True])

        make_platform(70.69, 57.64, cs2, internodes=True)
        cs1_array = np.array(
            [
                [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                [4, 5, 11, 5, 6, 7, 7, 8, 9, 9, 10, 11],
            ]
        ).T

        for x1, x2 in cs1_array:
            model.add_element(x1, x2, cs1, SS316)

        make_platform(40.96, 115.73, cs3)

        cs2_array = np.array(
            [
                [4, 5, 11, 5, 6, 7, 7, 8, 9, 9, 10, 11],
                [12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15],
            ]
        ).T
        for x1, x2 in cs2_array:
            model.add_element(x1, x2, cs2, SS316)

        make_platform(18.65, 276.13, cs4)

        cs3_array = np.array([[12, 13, 14, 15], [16, 17, 18, 19]]).T

        for x1, x2 in cs3_array:
            model.add_element(x1, x2, cs3, SS316)

        model.add_node(0, 0, 316)

        for i in range(4):
            model.add_element(16 + i, 20, cs3, SS316)

        model.add_node(0, 0, 324)
        model.add_element(20, 21, cs5, SS316)

        model.plot()
        plt.show()

        cls.model = model

    def test_something(self):
        self.model.add_gravity_loads()
        result = self.model.solve()
        result.plot(stress=True)
        self.model.clear_loads()
        plt.show()


@pytest.mark.longrun
class TestInterpolation:
    def test_model(self):
        xsection = RectangularBeam(0.2, 0.3)

        model = FiniteElementModel()

        model.add_node(0, 0, 0)
        model.add_node(0, 0, 2)
        model.add_node(2, 0, 2)
        model.add_node(4, 0, 2)
        model.add_node(4, 0, 0)
        model.add_node(4, 2, 2)
        model.add_node(4, 2, 0)
        model.add_element(0, 1, xsection, SS316)
        model.add_element(1, 2, xsection, SS316)
        model.add_element(2, 3, xsection, SS316)
        model.add_element(3, 4, xsection, SS316)

        model.add_element(2, 5, xsection, SS316)
        model.add_element(5, 6, xsection, SS316)

        model.add_support(0, *[True] * 6)
        model.add_support(4, *[True] * 6)
        model.add_support(6, *[True] * 6)

        model.add_distributed_load(1, -1000000, "Fz")
        model.add_gravity_loads()

        result = model.solve()

        result.plot(stress=True)
        plt.show()
