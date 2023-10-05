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
import copy

import numpy as np
import pytest
from matplotlib import pyplot as plt

from bluemira.base.constants import MU_0
from bluemira.equilibria.coils import (
    Coil,
    CoilGroup,
    CoilSet,
    CoilType,
    SymmetricCircuit,
    check_coilset_symmetric,
    make_mutual_inductance_matrix,
    symmetrise_coilset,
)
from bluemira.equilibria.constants import NBTI_J_MAX
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.grid import Grid


def callable_tester(f_callable, coils=1):
    """
    Checks that all different field calls (with different inputs,
    float, arrays of varying length etc) all return the same result
    """
    # This should go without a hitch...
    value = f_callable(8, 0)
    v2 = f_callable(np.array(8), np.array(0))
    v3 = f_callable(np.array([8]), np.array([0]))
    assert np.allclose(v2, value)
    assert np.allclose(v3, value)

    # Now let's check iterables (X = 4 or 20 is off-grid)
    # (Z = -10 or 10 off-grid)
    x = np.array([4, 8, 20, 4, 8, 20, 4, 8, 20])
    z = np.array([0, 0, 0, 10, 10, 10, -10, -10, 10])

    b = np.zeros((len(z), coils))

    b1 = f_callable(x, z)

    for i, (xi, zi) in enumerate(zip(x, z)):
        b[i] = f_callable(xi, zi)

    assert np.allclose(b.flat, b1.flat)


class TestCoil:
    @classmethod
    def setup_class(cls):
        # make a default coil
        cls.coil = Coil(x=4, z=4, current=10e6, ctype="PF", j_max=NBTI_J_MAX)
        cls.cs_coil = Coil(x=4, z=4, current=10e6, ctype="CS", j_max=NBTI_J_MAX)
        cls.no_coil = Coil(x=4, z=4, current=10e6, ctype="NONE", j_max=NBTI_J_MAX)

    def teardown_method(self):
        plt.show()
        plt.close("all")

    def test_name(self):
        assert self.coil.ctype == CoilType.PF
        assert self.cs_coil.ctype == CoilType.CS
        assert self.no_coil.ctype == CoilType.NONE
        coil = Coil(x=4, z=4, current=10e6, ctype="PF", j_max=NBTI_J_MAX)
        cs_coil = Coil(x=4, z=4, current=10e6, ctype="CS", j_max=NBTI_J_MAX)
        no_coil = Coil(x=4, z=4, current=10e6, ctype="NONE", j_max=NBTI_J_MAX)

        assert coil._number == self.coil._number + 1
        assert cs_coil._number == self.cs_coil._number + 1
        assert no_coil._number == self.no_coil._number + 1

    def test_field(self):
        c = Coil(x=1, z=0, current=1591550, dx=0, dz=0)  # Should produce 5 T on axis
        Bx, Bz = 0, MU_0 * c.current / (2 * c.x)

        assert c.Bx(0.001, 0) == Bx
        assert np.round(abs(c.Bz(0.001, 0) - Bz), 5) == 0
        z = 4
        Bx, Bz = (
            0,
            MU_0
            * 2
            * np.pi
            * c.x**2
            * c.current
            / (4 * np.pi * (z**2 + c.x**2) ** (3 / 2)),
        )
        assert np.round(abs(c.Bx(0.001, z) - Bx), 4) == 0
        assert np.round(abs(c.Bz(0.001, z) - Bz), 5) == 0
        psi_single = c.psi(15, 15)
        c.discretisation = 0.1
        assert np.round(abs(c.Bx(0.001, z) - Bx), 4) == 0
        assert np.round(abs(c.Bz(0.001, z) - Bz), 3) == 0
        psi_multi = c.psi(15, 15)
        assert np.round(abs(psi_single - psi_multi), 2) == 0

    def test_mesh(self):
        xmin, xmax = 0.1, 20
        nx, nz = 100, 100

        zmin, zmax = -12, 12
        x_1_d = np.linspace(xmin, xmax, nx)
        z_1_d = np.linspace(zmin, zmax, nz)
        x, z = np.meshgrid(x_1_d, z_1_d, indexing="ij")
        c = Coil(x=4, z=0, current=1591550, dx=0.3, dz=1)

        gbx = c.Bx_response(x, z)
        gbz = c.Bz_response(x, z)
        _ = np.sqrt(gbx**2 + gbz**2)
        _ = c.psi_response(x, z)

        _, ax = plt.subplots()
        cc = ax.contourf(x, z, gbx)

        plt.colorbar(cc)
        ax.set_aspect("equal")
        ax.set_xlim([2, 6])
        ax.set_ylim([-3, 3])

        c.discretisation = 0.1

        gbxn = c.Bx_response(x, z)
        _ = c.Bz_response(x, z)
        _ = np.sqrt(gbx**2 + gbz**2)
        _ = c.psi_response(x, z)

        _, ax = plt.subplots()
        c = ax.contourf(x, z, gbxn)
        plt.colorbar(c)
        ax.set_aspect("equal")
        ax.set_xlim([2, 6])
        ax.set_ylim([-3, 3])

    def test_bx(self):
        callable_tester(self.coil.Bx)

    def test_bz(self):
        callable_tester(self.coil.Bz)

    def test_bp(self):
        callable_tester(self.coil.Bp)

    def test_psi(self):
        callable_tester(self.coil.psi)

    def test_point_in_coil(self):
        coil = Coil(x=4, z=4, current=10, dx=1, dz=2)
        inside_x = [3, 4, 5, 3, 4, 5, 3, 4, 5]
        inside_z = [2, 2, 2, 3, 3, 3, 6, 6, 6]
        inside = coil._points_inside_coil(inside_x, inside_z)

        assert np.all(inside)

        outside_x = [0, 0, 0, 1, 1, 1, 10, 10, 10, 3, 3, 3]
        outside_z = [0, 4, 6, 0, 4, 6, 0, 4, 6, 1.9, 6.1, 10]
        outside = coil._points_inside_coil(outside_x, outside_z)

        assert np.all(~outside)
        assert np.all(coil._points_inside_coil(coil.x_boundary, coil.z_boundary))

    def test_position(self):
        coil = Coil(x=4, z=4, current=10, dx=1, dz=2)
        pos = np.array([coil.x, coil.z])
        pos += 1
        coil.position = pos
        assert np.allclose(coil.position, np.array([5, 5]))
        coil.x = 6
        coil.z = 6
        assert np.allclose(coil.position, np.array([6, 6]))


class TestSemiAnalytic:
    """
    Compare all three control response methods, and that the combination of the
    Greens and semi-analytical methods makes sense (graphically..)
    """

    @classmethod
    def setup_class(cls):
        cls.coil = Coil(x=4, z=4, ctype="PF", current=10e6, dx=1, dz=2)
        cls.cg1 = CoilGroup(cls.coil)
        cls.cg2 = CoilGroup(
            cls.coil, Coil(x=8, z=8, ctype="PF", current=10e6, dx=1, dz=2)
        )
        cls.coil.discretisation = 0.2
        cls.cg1.discretisation = 0.2
        cls.cg2.discretisation = 0.2
        cls.grid = Grid(0.1, 8, 0, 8, 100, 100)
        cls.grid2 = Grid(0.1, 12, 0, 12, 100, 100)
        cls.x_boundary = [np.append(cls.coil.x_boundary, cls.coil.x_boundary[0])]
        cls.z_boundary = [np.append(cls.coil.z_boundary, cls.coil.z_boundary[0])]
        cls.x_boundary.append(
            np.append(cls.cg2.x_boundary, cls.cg2.x_boundary[:, 0][:, None], axis=-1)
        )
        cls.z_boundary.append(
            np.append(cls.cg2.z_boundary, cls.cg2.z_boundary[:, 0][:, None], axis=-1)
        )

    def teardown_method(self):
        plt.show()
        plt.close("all")

    def _plotter(self, gp, gp_greens, gp_analytic):
        _, ax = plt.subplots(3, 3)

        for axis in ax[:2].flat:
            axis.plot(self.x_boundary[0], self.z_boundary[0], color="r")
            axis.set_aspect("equal")

        for axis in ax[2].flat:
            axis.plot(self.x_boundary[1][0], self.z_boundary[1][0], color="r")
            axis.plot(self.x_boundary[1][1], self.z_boundary[1][1], color="r")
            axis.set_aspect("equal")

        levels = np.linspace(np.amin(gp[0]), np.amax(gp[0]), 20)

        ax[0, 0].set_title("Green's functions")
        ax[0, 1].set_title("Combined Green's and semi-analytic")
        ax[0, 2].set_title("Semi-analytic method")

        for i in range(2):
            ax[i, 0].contourf(self.grid.x, self.grid.z, gp_greens[i], levels=levels)
            ax[i, 1].contourf(self.grid.x, self.grid.z, gp[i], levels=levels)
            ax[i, 2].contourf(self.grid.x, self.grid.z, gp_analytic[i], levels=levels)

        levels = np.linspace(np.amin(gp[2]), np.amax(gp[2]), 20)
        ax[2, 0].contourf(
            self.grid2.x, self.grid2.z, np.sum(gp_greens[2], axis=-1), levels=levels
        )
        ax[2, 1].contourf(
            self.grid2.x, self.grid2.z, np.sum(gp[2], axis=-1), levels=levels
        )
        ax[2, 2].contourf(
            self.grid2.x, self.grid2.z, np.sum(gp_analytic[2], axis=-1), levels=levels
        )

    @pytest.mark.parametrize("fd", ["Bx", "Bz"])
    def test_bfield(self, fd):
        gp_greens = []
        gp_analytic = []
        gp = []
        for cl, grid in [
            [self.coil, self.grid],
            [self.cg1, self.grid],
            [self.cg2, self.grid2],
        ]:
            gp_greens.append(getattr(cl, f"_{fd}_response_greens")(grid.x, grid.z))
            gp_analytic.append(getattr(cl, f"_{fd}_response_analytical")(grid.x, grid.z))
            gp.append(getattr(cl, f"{fd}_response")(grid.x, grid.z))

        self._plotter(gp, gp_greens, gp_analytic)


class TestCoilGroup:
    def setup_method(self):
        x = [6, 7, 4, 4]
        z = [6, 7, 4, 5]
        ctype = ["CS", "CS", "PF", "PF"]
        name = ["CS_8", "CS_0", "PF_1", "PF_0"]
        j_max = NBTI_J_MAX

        self.group = CoilGroup(
            *(
                Coil(x=_x, z=_z, name=_n, ctype=_ct, j_max=j_max)
                for _x, _z, _ct, _n in zip(x, z, ctype, name)
            )
        )

    def test_init_sort(self):
        assert self.group.n_coils() == 4
        assert self.group.name == ["CS_8", "CS_0", "PF_1", "PF_0"]

    def test_add(self):
        self.group.add_coil(Coil(3, 3, ctype="PF", name="PF_3", j_max=NBTI_J_MAX))
        self.group.add_coil(Coil(9, 9, ctype="CS", name="CS_9", j_max=NBTI_J_MAX))

        assert self.group.n_coils() == 6
        assert self.group.name == ["CS_8", "CS_0", "PF_1", "PF_0", "PF_3", "CS_9"]

    def test_remove(self):
        self.group.remove_coil("PF_0")
        self.group.remove_coil("PF_1")

        assert self.group.n_coils() == 2
        assert self.group.name == ["CS_8", "CS_0"]

        with pytest.raises(EquilibriaError):
            self.group.remove_coil("PF_1")

        # TODO test nested removal

    def test_resize(self):
        initdx = self.group.dx
        initdz = self.group.dz

        self.group.fix_sizes()
        self.group.resize(10)

        np.testing.assert_allclose(self.group.dx, initdx)
        np.testing.assert_allclose(self.group.dz, initdz)

        self.group._resize(10)

        assert not np.allclose(self.group.dx, initdx)
        assert not np.allclose(self.group.dz, initdz)

        with pytest.raises(ValueError):  # noqa: PT011
            self.group.resize([10, 10])

    def test_psi(self):
        callable_tester(self.group.psi, self.group.n_coils())

    def test_bx(self):
        callable_tester(self.group.Bx, self.group.n_coils())

    def test_bz(self):
        callable_tester(self.group.Bz, self.group.n_coils())

    def test_bp(self):
        callable_tester(self.group.Bp, self.group.n_coils())


class TestSymmetricCircuit:
    @classmethod
    def setup_class(cls):
        coil = Coil(x=1.5, z=6, current=1e6, dx=0.25, dz=0.5, ctype="PF", name="TEST")
        mirror_coil = Coil(
            x=1.5, z=-6, current=1e6, dx=0.25, dz=0.5, ctype="PF", name="TEST_MIRROR"
        )
        cls.circuit = SymmetricCircuit(coil, mirror_coil)
        cls.coils = [copy.deepcopy(coil), copy.deepcopy(mirror_coil)]

    @pytest.mark.parametrize("fieldtype", ["_response", ""])
    def test_fields(self, fieldtype):
        points = [
            [1, 1],
            [2, 2],
            [1.5, 6],
            [1.5, -6],
        ]

        for point in points:
            coil_psi = sum(
                [getattr(coil, f"psi{fieldtype}")(*point) for coil in self.coils]
            )
            coil_Bx = sum(
                [getattr(coil, f"Bx{fieldtype}")(*point) for coil in self.coils]
            )
            coil_Bz = sum(
                [getattr(coil, f"Bz{fieldtype}")(*point) for coil in self.coils]
            )

            circuit_psi = getattr(self.circuit, f"psi{fieldtype}")(*point)
            circuit_Bx = getattr(self.circuit, f"Bx{fieldtype}")(*point)
            circuit_Bz = getattr(self.circuit, f"Bz{fieldtype}")(*point)

            assert np.allclose(coil_psi, np.sum(circuit_psi))
            assert np.allclose(coil_Bx, np.sum(circuit_Bx))
            assert np.allclose(coil_Bz, np.sum(circuit_Bz))

    def test_current(self):
        self.circuit.current = 2e6
        for coil in self.coils:
            coil.current = 2e6
        self.test_fields("")

    def test_attributes(self):
        circ = copy.deepcopy(self.circuit)
        circ.x = 4
        assert circ.x[0] == 4
        assert circ.x[1] == 4

        circ.z = 6
        assert circ.z[0] == 6
        assert circ.z[1] == -6

        assert np.allclose(self.coils[0].volume, self.circuit.volume)

    def test_position(self):
        circ = copy.deepcopy(self.circuit)
        before = circ.x.copy()
        circ.x = circ.x + 1
        assert np.allclose(before + 1, circ.x)


class TestCoilSet:
    @classmethod
    def setup_class(cls):
        coil = Coil(
            x=4, z=10, current=2e6, dx=1, dz=0.5, j_max=5.0, b_max=50, name="PF_1"
        )
        circuit = SymmetricCircuit(
            Coil(
                x=1.5,
                z=6,
                current=1e6,
                dx=0.25,
                dz=0.5,
                j_max=10.0,
                b_max=100,
                ctype="PF",
                name="PF_2",
            ),
            Coil(
                x=1.5,
                z=-6,
                current=1e6,
                dx=0.25,
                dz=0.5,
                j_max=10.0,
                b_max=100,
                ctype="PF",
                name="PF_3",
            ),
        )

        cls.coilset = CoilSet(coil, circuit)

    def test_padding_of_quads(self):
        cs = copy.deepcopy(self.coilset)
        cs._coils[0].discretisation = 0.5

        # if padding is done on multiple axes this shape will be (10, 8)
        assert cs._quad_x.shape == (3, 8)

    def test_group_vecs(self):
        x, z, dx, dz, currents = self.coilset.to_group_vecs()

        assert np.allclose(currents, np.array([2e6, 1e6, 1e6]))
        assert np.allclose(x, np.array([4, 1.5, 1.5]))
        assert np.allclose(z, np.array([10, 6, -6]))
        assert np.allclose(dz, np.array([0.5, 0.5, 0.5]))
        assert np.allclose(dx, np.array([1, 0.25, 0.25]))

    def test_member_attributes(self):
        coilset = copy.deepcopy(self.coilset)

        assert np.isclose(self.coilset["PF_1"].x, 4)
        assert np.isclose(self.coilset["PF_2"].x, 1.5)

        assert np.isclose(self.coilset["PF_1"].z, 10)
        assert np.isclose(self.coilset["PF_2"].z, 6)

        assert np.isclose(self.coilset["PF_1"].dx, 1)
        assert np.isclose(self.coilset["PF_2"].dx, 0.25)

        assert np.isclose(self.coilset["PF_1"].dz, 0.5)
        assert np.isclose(self.coilset["PF_2"].dz, 0.5)

        assert np.isclose(self.coilset["PF_1"].j_max, 5.0)
        assert np.isclose(self.coilset["PF_2"].j_max, 10.0)

        assert np.isclose(self.coilset["PF_1"].b_max, 50)
        assert np.isclose(self.coilset["PF_2"].b_max, 100)

        assert np.isclose(self.coilset["PF_1"].current, 2e6)
        assert np.isclose(self.coilset["PF_2"].current, 1e6)

        coil1 = coilset["PF_1"]
        coil1.dz = 0.77
        symm_circuit = coilset._coils[1]
        symm_circuit.dz = 0.6
        assert np.isclose(coil1.dz, 0.77)
        assert np.allclose(symm_circuit.dz, 0.6)
        assert np.isclose(coilset["PF_3"].dz, 0.6)

    def test_numbers(self):
        assert self.coilset.n_coils("PF") == 2
        assert self.coilset.n_coils("CS") == 0
        assert self.coilset.n_coils("NONE") == 1
        assert self.coilset.n_coils() == 3

    def test_currents(self):
        coilset = copy.deepcopy(self.coilset)

        set_currents = np.array([3e6, 4e6])
        coilset.get_control_coils().current = set_currents
        currents = coilset.get_control_coils().current
        assert np.allclose(set_currents, currents[:-1])
        assert np.isclose(set_currents[-1], currents[-1])

    def test_material_assignment(self):
        coilset = copy.deepcopy(self.coilset)
        test_j_max = 7.0
        test_b_max = 24.0
        coilset.assign_material("PF", j_max=test_j_max, b_max=test_b_max)
        n_indep_coils = coilset.n_coils("PF")
        assert len(coilset.get_max_current()) == n_indep_coils + 1
        assert len(coilset.b_max) == n_indep_coils + 1
        assert np.allclose(coilset.j_max[1:], test_j_max)
        assert np.allclose(coilset.b_max[1:], test_b_max)

    def test_get_max_current(self):
        coilset = copy.deepcopy(self.coilset)
        coilset["PF_1"]._flag_sizefix = False

        # isnan(j_max) = False False False
        # not flagsizefix = True False False
        np.testing.assert_allclose(coilset.get_max_current(10), [10, 5, 5])
        np.testing.assert_allclose(coilset.get_max_current(), [np.infty, 5, 5])

        # isnan(j_max) = True True True
        # not flagsizefix = True False False
        coilset.assign_material("PF", j_max=np.nan, b_max=5)

        np.testing.assert_allclose(coilset.get_max_current(10), [10, 10, 10])
        np.testing.assert_allclose(
            coilset.get_max_current(), [np.infty, np.infty, np.infty]
        )


class TestCoilSetSymmetry:
    def test_symmetry_check(self):
        coilset = CoilSet(Coil(5, 5, dx=1.0, dz=1.0), Coil(5, -5, dx=1.0, dz=1.0))

        assert check_coilset_symmetric(coilset)

        coilset = CoilSet(
            Coil(5, 5, dx=1.0, dz=1.0), Coil(5, -5, dx=1.0, dz=1.0, current=1e6)
        )

        assert not check_coilset_symmetric(coilset)

        coilset = CoilSet(
            Coil(5, 5, dx=1.0, dz=1.0),
            Coil(5, 0, dx=1.0, dz=1.0),
            Coil(5, -5, dx=1.0, dz=1.0),
        )
        assert check_coilset_symmetric(coilset)

        coilset = CoilSet(
            Coil(5, 5, dx=1.0, dz=1.0),
            Coil(5, 1, dx=1.0, dz=1.0),
            Coil(5, -5, dx=1.0, dz=1.0),
        )

        assert not check_coilset_symmetric(coilset)

        coilset = CoilSet(
            Coil(5, 5, dx=1.0, dz=1.0),
            Coil(5, 0, dx=1.0, dz=1.0),
            Coil(5, 1, dx=1.0, dz=1.0),
            Coil(5, -5, dx=1.0, dz=1.0),
        )

        assert not check_coilset_symmetric(coilset)

    def test_symmetrise(self):
        coilset = CoilSet(
            Coil(5, 5, current=1e6, dx=1, dz=1),
            Coil(5, -5, current=1e6, dx=1, dz=1),
        )
        new = symmetrise_coilset(coilset)
        assert len(new._coils) == 1
        assert new.n_coils() == 2
        assert isinstance(next(iter(new._coils)), SymmetricCircuit)

        coilset = CoilSet(
            SymmetricCircuit(
                Coil(5, 5, current=1e6, dx=1, dz=1), Coil(5, -5, current=1e6, dx=1, dz=1)
            ),
            SymmetricCircuit(
                Coil(12, 7, current=1e6, dx=1, dz=1),
                Coil(12, -7, current=1e6, dx=1, dz=1),
            ),
            SymmetricCircuit(Coil(4, 9, current=1e6, dx=1, dz=1)),
            Coil(5, 0, current=1e6, dx=1, dz=1),
        )
        new = symmetrise_coilset(coilset)
        assert len(new._coils) == len(coilset._coils)
        assert new.n_coils() == coilset.n_coils()


class TestCoilSizing:
    def test_initialisation(self):
        c = Coil(4, 4, current=0, j_max=np.nan, dx=1, dz=1)
        assert c.dx == 1
        assert c.dz == 1

        c = Coil(4, 4, current=0, j_max=np.nan, dx=1, dz=2)
        assert c.dx == 1
        assert c.dz == 2

        c = Coil(4, 4, current=0, j_max=np.nan, dx=2, dz=1)
        assert c.dx == 2
        assert c.dz == 1

        c = Coil(4, 4, current=0, j_max=np.nan, dx=1, dz=1)
        assert c.dx == 1
        assert c.dz == 1
        assert c._flag_sizefix

        c = Coil(4, 4, current=0, j_max=10)
        assert c.dx == 0
        assert c.dz == 0
        assert not c._flag_sizefix

    def test_bad_initialisation(self):
        with pytest.raises(EquilibriaError):
            Coil(4, 4, current=1)
        with pytest.raises(EquilibriaError):
            Coil(4, 4, dx=1)
        with pytest.raises(EquilibriaError):
            Coil(4, 4, dz=1)
        with pytest.raises(EquilibriaError):
            Coil(4, 4, j_max=np.nan)


class TestMutualInductances:
    @classmethod
    def setup_class(cls):
        coil1 = Coil(4, 4, j_max=1)
        coil2 = Coil(5, 5, j_max=1)
        coil3 = Coil(6, 6, j_max=1)
        cls.coilset1 = CoilGroup(coil1, coil2, coil3)

    def test_normal(self):
        """
        Just check the symmetry for now
        """
        m = make_mutual_inductance_matrix(self.coilset1)

        assert m.shape == (3, 3)
        idxs = np.triu_indices(3, k=1)
        tri_upper = m[idxs]

        test_m = np.zeros((3, 3))
        test_m[idxs] = tri_upper
        test_m = test_m + test_m.T
        diag = np.diag_indices(3)
        m[diag] = 0.0
        assert np.allclose(m, test_m)
