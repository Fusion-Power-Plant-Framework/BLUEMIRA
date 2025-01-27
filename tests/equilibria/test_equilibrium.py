# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from eqdsk import EQDSKInterface
from matplotlib import pyplot as plt

from bluemira.base.file import get_bluemira_path, try_get_bluemira_private_data_root
from bluemira.equilibria.coils import CoilGroup, CoilSet
from bluemira.equilibria.equilibrium import Equilibrium, FixedPlasmaEquilibrium
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.optimisation.constraints import (
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
)
from bluemira.equilibria.optimisation.problem import (
    UnconstrainedTikhonovCurrentGradientCOP,
)
from bluemira.equilibria.physics import calc_li3
from bluemira.equilibria.profiles import (
    BetaIpProfile,
    BetaLiIpProfile,
    CustomProfile,
    DoublePowerFunc,
    LaoPolynomialFunc,
)
from bluemira.equilibria.shapes import flux_surface_kuiroukidis
from bluemira.equilibria.solve import DudsonConvergence, PicardIterator
from bluemira.utilities.tools import abs_rel_difference, compare_dicts
from tests.equilibria.setup_methods import _coilset_setup


class TestFields:
    @classmethod
    def setup_class(cls):
        # Let's make a complete **** equilibrium..
        _coilset_setup(cls)
        grid = Grid(4.5, 14, -9, 9, 65, 65)

        profiles = CustomProfile(
            np.linspace(1, 0), -np.linspace(1, 0), R_0=9, B_0=6, I_p=10e6
        )

        eq = Equilibrium(cls.coilset, grid, profiles)

        isoflux = IsofluxConstraint(
            np.array([6, 8, 12, 6]), np.array([0, 7, 0, -8]), 6, 0, 0
        )
        x_point = FieldNullConstraint(8, -8)

        targets = MagneticConstraintSet([isoflux, x_point])

        opt_problem = UnconstrainedTikhonovCurrentGradientCOP(
            eq.coilset, eq, targets, gamma=1e-8
        )

        program = PicardIterator(eq, opt_problem, relaxation=0.1, plot=False)
        program()
        cls.eq = eq

    def callable_tester(self, f_callable):
        """
        Checks that all different field calls (with different inputs) behave
        as expected
        """
        # This should go without a hitch...
        value = f_callable(8, 0)
        v2 = f_callable(np.array(8), np.array(0))
        v3 = f_callable(np.array([8]), np.array([0]))[0]
        assert np.isclose(v2, value)
        assert np.isclose(v3, value)

        # Now let's check the full field calls
        b1 = f_callable(x=None, z=None)
        b2 = f_callable(self.eq.grid.x, self.eq.grid.z)
        assert np.allclose(b1, b2)

        # Now let's check iterables (X = 4 or 20 is off-grid)
        # (Z = -10 or 10 off-grid)
        x_array = np.array([4, 8, 20, 4, 8, 20, 4, 8, 20])
        z_array = np.array([0, 0, 0, 10, 10, 10, -10, -10, 10])

        b_values = np.zeros(len(z_array))

        for i, (x, z) in enumerate(zip(x_array, z_array, strict=False)):
            b_values[i] = f_callable(x, z)

        b1 = f_callable(x_array, z_array)

        assert np.allclose(b_values, b1)

    def test_Bx(self):
        self.callable_tester(self.eq.Bx)

    def test_Bz(self):
        self.callable_tester(self.eq.Bz)

    def test_Bp(self):
        self.callable_tester(self.eq.Bp)

    def test_psi(self):
        self.callable_tester(self.eq.psi)

    def test_out_of_bounds(self):
        eq = self.eq
        f, (ax, ax2, ax3, ax4) = plt.subplots(1, 4)
        psi = eq.psi()
        psi = eq.plasma.psi()

        newgrid = Grid(1, 15, -15, 15, 100, 100)  # bigger grid

        newpsi = np.zeros((newgrid.nx, newgrid.nz))

        # NOTE: _plasmacoil should not be accessed directly, this is just to
        # check. also why we have to do the following useless call to init
        # the _plasmacoil
        eq.plasma.Bz(0.1, 0)
        for i, x in enumerate(newgrid.x_1d):
            for j, z in enumerate(newgrid.z_1d):
                newpsi[i, j] = eq.plasma.psi(x, z)

        levels = np.linspace(np.amin(newpsi), np.amax(newpsi), 20)
        ax.plot(*eq.grid.bounds, color="r")
        ax.contour(eq.grid.x, eq.grid.z, psi, levels=levels)
        ax.contour(
            newgrid.x,
            newgrid.z,
            newpsi,
            levels=levels,
            cmap="plasma",
            linestyles="dashed",
        )
        ax.set_aspect("equal")
        ax.set_title("plasma_psi")

        Bx = eq.plasma.Bx()

        new_bx = np.zeros((newgrid.nx, newgrid.nz))

        for i, x in enumerate(newgrid.x_1d):
            for j, z in enumerate(newgrid.z_1d):
                new_bx[i, j] = eq.plasma.Bx(x, z)

        levels = np.linspace(np.amin(new_bx), np.amax(new_bx), 20)

        ax2.plot(*eq.grid.bounds, color="r")
        ax2.contour(eq.grid.x, eq.grid.z, Bx, levels=levels)
        ax2.contour(
            newgrid.x,
            newgrid.z,
            new_bx,
            levels=levels,
            cmap="plasma",
            linestyles="dashed",
        )
        ax2.set_aspect("equal")
        ax2.set_title("plasma_Bx")

        Bz = eq.plasma.Bz()

        new_bz = np.zeros((newgrid.nx, newgrid.nz))

        for i, x in enumerate(newgrid.x_1d):
            for j, z in enumerate(newgrid.z_1d):
                new_bz[i, j] = eq.plasma.Bz(x, z)

        levels = np.linspace(np.amin(new_bz), np.amax(new_bz), 20)

        ax3.plot(*eq.grid.bounds, color="r")
        ax3.contour(eq.grid.x, eq.grid.z, Bz, levels=levels)
        ax3.contour(
            newgrid.x,
            newgrid.z,
            new_bz,
            levels=levels,
            cmap="plasma",
            linestyles="dashed",
        )
        ax3.set_aspect("equal")
        ax3.set_title("plasma_Bz")

        Bp = eq.plasma.Bp()

        new_bp = np.sqrt(new_bx**2 + new_bz**2)

        levels = np.linspace(np.amin(new_bp), np.amax(new_bp), 20)

        ax4.plot(*eq.grid.bounds, color="r")
        ax4.contour(eq.grid.x, eq.grid.z, Bp, levels=levels)
        ax4.contour(
            newgrid.x,
            newgrid.z,
            new_bp,
            levels=levels,
            cmap="plasma",
            linestyles="dashed",
        )
        ax4.set_aspect("equal")
        ax4.set_title("plasma_Bp")


class TestSolveEquilibrium:
    R_0 = 9
    A = 3
    I_p = 15e6
    beta_p = 1.2
    l_i = 0.8
    B_0 = 5

    grid = Grid(5, 15, -8, 8, 50, 50)
    lcfs = flux_surface_kuiroukidis(R_0, 0, R_0 / A, 1.5, 1.7, 0.33, 0.33)
    x_arg = np.argmin(lcfs.z)
    targets = MagneticConstraintSet([
        IsofluxConstraint(lcfs.x, lcfs.z, lcfs.x[0], lcfs.z[0]),
        FieldNullConstraint(lcfs.x[x_arg], lcfs.z[x_arg]),
    ])
    shape_funcs = (DoublePowerFunc([2, 1]), LaoPolynomialFunc([3, 1, 0.5]))

    @classmethod
    def setup_class(cls):
        _coilset_setup(cls, materials=True)

    def test_custom_profile(self):
        profiles = CustomProfile(
            np.array([
                86856,
                86506,
                84731,
                80784,
                74159,
                64576,
                52030,
                36918,
                20314,
                4807,
                0.0,
            ]),
            -np.array([
                0.125,
                0.124,
                0.122,
                0.116,
                0.106,
                0.093,
                0.074,
                0.053,
                0.029,
                0.007,
                0.0,
            ]),
            R_0=self.R_0,
            B_0=self.B_0,
            I_p=self.I_p,
        )
        eq = Equilibrium(deepcopy(self.coilset), self.grid, profiles)
        opt_problem = UnconstrainedTikhonovCurrentGradientCOP(
            eq.coilset, eq, self.targets, gamma=1e-8
        )
        program = PicardIterator(
            eq,
            opt_problem,
            convergence=DudsonConvergence(1e-1),
            fixed_coils=True,
            relaxation=0.2,
            plot=False,
            gif=False,
        )
        program()
        assert program.check_converged()

    @pytest.mark.parametrize("shape", shape_funcs)
    def test_betaip_profile(self, shape):
        profiles = BetaIpProfile(self.beta_p, self.I_p, self.R_0, self.B_0, shape=shape)
        eq = Equilibrium(deepcopy(self.coilset), self.grid, profiles)
        opt_problem = UnconstrainedTikhonovCurrentGradientCOP(
            eq.coilset, eq, self.targets, gamma=1e-8
        )
        program = PicardIterator(
            eq,
            opt_problem,
            convergence=DudsonConvergence(1e-1),
            fixed_coils=True,
            relaxation=0.2,
            plot=False,
            gif=False,
        )
        program()
        assert program.check_converged()

    @pytest.mark.parametrize("shape", shape_funcs)
    def test_betapliip_profile(self, shape):
        rel_tol = 0.015
        profiles = BetaLiIpProfile(
            self.beta_p,
            self.l_i,
            self.I_p,
            self.R_0,
            self.B_0,
            shape=shape,
            li_min_iter=0,
            li_rel_tol=rel_tol,
        )
        eq = Equilibrium(deepcopy(self.coilset), self.grid, profiles)
        opt_problem = UnconstrainedTikhonovCurrentGradientCOP(
            eq.coilset, eq, self.targets, gamma=1e-8
        )
        program = PicardIterator(
            eq,
            opt_problem,
            convergence=DudsonConvergence(1e-1),
            fixed_coils=True,
            relaxation=0.2,
            plot=False,
            gif=False,
        )
        program()
        assert abs_rel_difference(calc_li3(eq), self.l_i) <= rel_tol
        assert program.check_converged()


class TestEquilibrium:
    @classmethod
    def setup_class(cls):
        path = get_bluemira_path("equilibria/test_data", subfolder="tests")
        cls.dn = Equilibrium.from_eqdsk(
            Path(path, "DN-DEMO_eqref.json"), from_cocos=3, qpsi_positive=False
        )
        cls.sn = Equilibrium.from_eqdsk(Path(path, "eqref_OOB.json"), from_cocos=7)

    def test_double_null(self):
        assert self.dn.is_double_null
        assert not self.sn.is_double_null

    def test_qpsi_calculation_modes(self):
        with patch.object(self.dn, "q") as eq_q:
            res = self.dn.to_dict(qpsi_calcmode=0)
            assert eq_q.call_count == 0
            assert "qpsi" not in res

            res = self.dn.to_dict(qpsi_calcmode=1)
            assert eq_q.call_count == 1
            assert "qpsi" in res

            res = self.dn.to_dict(qpsi_calcmode=2)
            assert eq_q.call_count == 1
            assert "qpsi" in res
            assert np.all(res["qpsi"] == 0)  # array is all zeros

    @pytest.mark.parametrize("grouping", [CoilSet, CoilGroup])
    def test_woops_no_coils(self, grouping):
        testfile = Path(
            get_bluemira_path("equilibria/test_data", subfolder="tests"),
            "jetto.eqdsk_out",
        )
        e = EQDSKInterface.from_file(testfile, from_cocos=11)
        coil = grouping.from_group_vecs(e)
        assert isinstance(coil, grouping), "Check classmethod is making the right class"
        assert coil.current.any() == 0
        assert coil.j_max.any() == 0
        assert coil.b_max.any() == 0
        assert coil.n_coils(ctype="DUM") == 4
        if grouping is CoilSet:
            assert len(coil.control) == 0

    def test_eq_coilnames(self):
        testfile = Path(
            get_bluemira_path("equilibria/test_data", subfolder="tests"),
            "DN-DEMO_eqref_withCoilNames.json",
        )
        e = Equilibrium.from_eqdsk(testfile, from_cocos=3, qpsi_positive=False)
        assert e.coilset.name == [
            *("PF_1", "PF_2", "PF_3", "PF_4", "PF_5", "PF_6"),
            *("CS_1", "CS_2", "CS_3", "CS_4", "CS_5"),
        ]
        assert e.coilset.n_coils(ctype="PF") == 6
        assert e.coilset.n_coils(ctype="CS") == 5

    def test_plotting_field(self):
        self.dn.plot_field()

    @pytest.mark.parametrize("plasma", [False, True])
    def test_plotting_plasma(self, plasma):
        self.dn.plot(plasma=plasma)


class TestEqReadWrite:
    @pytest.mark.parametrize("qpsi_calcmode", [0, 1])
    @pytest.mark.parametrize("file_format", ["json", "eqdsk"])
    def test_read_write(self, qpsi_calcmode, file_format):
        data_path = get_bluemira_path("equilibria/test_data", subfolder="tests")
        file_name = "eqref_OOB.json"
        new_file_name = f"eqref_OOB_temp1.{file_format}"
        new_file_path = Path(data_path, new_file_name)

        eq = Equilibrium.from_eqdsk(
            Path(data_path, file_name), from_cocos=7, to_cocos=None
        )
        # Note we have recalculated the qpsi data here
        eq.to_eqdsk(
            directory=data_path,
            filename=new_file_name,
            qpsi_calcmode=qpsi_calcmode,
            filetype=file_format,
        )
        d1 = eq.to_dict(qpsi_calcmode=qpsi_calcmode)

        eq2 = Equilibrium.from_eqdsk(
            new_file_path,
            from_cocos=7 if qpsi_calcmode else 3,
            to_cocos=None,
            qpsi_positive=None if qpsi_calcmode else False,
        )
        d2 = eq2.to_dict(qpsi_calcmode=qpsi_calcmode)
        new_file_path.unlink()
        if file_format == "eqdsk":
            d1.pop("coil_names")
            d2.pop("coil_names")
        assert compare_dicts(d1, d2, almost_equal=True)


@pytest.mark.private
class TestQBenchmark:
    @classmethod
    def setup_class(cls):
        root = try_get_bluemira_private_data_root()
        path = Path(root, "equilibria", "STEP_SPR_08")
        jetto_file = "SPR-008_3_Inputs_jetto.eqdsk_out"
        jetto = EQDSKInterface.from_file(Path(path, jetto_file), from_cocos=11)
        cls.q_ref = jetto.qpsi
        eq_file = "SPR-008_3_Outputs_STEP_eqref.eqdsk"
        cls.eq = Equilibrium.from_eqdsk(Path(path, eq_file), from_cocos=7)

    def test_q_benchmark(self):
        n = len(self.q_ref)
        psi_norm = np.linspace(0, 1, n)
        q = self.eq.q(psi_norm)

        assert 100 * np.median(abs_rel_difference(q, self.q_ref)) < 3.5
        assert 100 * np.mean(abs_rel_difference(q, self.q_ref)) < 4.0


@pytest.mark.private
class TestFixedPlasmaEquilibrium:
    @classmethod
    def setup_class(cls):
        root = try_get_bluemira_private_data_root()
        path = Path(root, "equilibria", "STEP_SPR_08", "jetto.eqdsk_out")
        cls.eq = FixedPlasmaEquilibrium.from_eqdsk(path)

    @pytest.mark.parametrize("field", [False, True])
    def test_plotting(self, field):
        self.eq.plot(field=field)
