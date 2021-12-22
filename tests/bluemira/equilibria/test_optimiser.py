# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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
import pytest
import tests
from unittest.mock import patch, MagicMock
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from bluemira.base.file import get_bluemira_path
from bluemira.geometry._deprecated_loop import Loop
from bluemira.geometry._deprecated_tools import make_circle_arc
from bluemira.equilibria.optimiser import (
    PositionOptimiser,
    BreakdownOptimiser,
    CoilsetOptimiser,
)
from bluemira.utilities.opt_tools import process_scipy_result
from bluemira.equilibria.equilibrium import Breakdown
from bluemira.equilibria.grid import Grid
from tests.bluemira.equilibria.setup_methods import _coilset_setup, _make_square
from bluemira.equilibria.coils import (
    Coil,
    CoilSet,
    SymmetricCircuit,
    PF_COIL_NAME,
)


class TestPositionOptimiser:
    @classmethod
    def setup_class(cls):
        cls._coilset_setup = _coilset_setup

    def _track_setup(self):
        # Stripped from
        # BLUEPRINT.examples.equilibria.single_null
        fp = get_bluemira_path("geometry", subfolder="data")
        TF = Loop.from_file(os.sep.join([fp, "TFreference.json"]))
        TF = TF.offset(2.4)
        clip = np.where(TF.x >= 3.5)
        self.TF_track = Loop(TF.x[clip], z=TF.z[clip])
        self.TF_track.interpolate(200)

    def _coil_positioning_b4_update(self, pos_opt, positions):
        s, f = 0, 2
        for c_name, coil in self.coilset.coils.items():
            if c_name in ["PF_5", "PF_6"]:
                num = int(c_name.split("_")[1])
                assert self.pos_dict[c_name] == pos_opt.Rmap.L_to_xz(
                    num, positions[-4:][s:f]
                )
                s += 2
                f += 2

    def _coil_positioning(self, pos_opt, positions, coilset):

        for name, pos in zip(self.coil_names, positions[:4][::-1]):
            cpos = pos_opt.XLmap.L_to_xz(pos)
            coil = coilset.coils[name]
            np.testing.assert_equal(cpos, pos_opt._positions[name])
            np.testing.assert_equal(cpos, np.array((coil.x, coil.z)))

        CS_pos = pos_opt.XLmap.L_to_zdz(positions[4:9])
        for name, cpos in zip(self.coil_names[4:9], zip(*CS_pos)):
            coil = coilset.coils[name]

            np.testing.assert_equal(cpos, pos_opt._positions[name])
            np.testing.assert_equal(cpos, np.array((coil.x, coil.z, coil.dz)))

        for name, pos in zip(self.coil_names[9:], positions[9:].reshape(-1, 2)):
            num = int(name.split("_")[1])
            cpos = pos_opt.Rmap.L_to_xz(num, pos)
            coil = coilset.coils[name]
            np.testing.assert_equal(cpos, pos_opt._positions[name])
            np.testing.assert_equal(cpos, np.array((coil.x, coil.z)))

    def PFcoil_region_ordering(self, patch_loc, nlo_retv, pos_opt, eq):
        with patch(
            patch_loc + "nlopt.opt", name="nlopt.opt", return_value=nlo_retv
        ) as nlo:
            with patch(patch_loc + "process_NLOPT_result"):
                with patch.object(pos_opt, "update_positions"):

                    pos_opt.I_star = None
                    pos_opt(eq, constraints=None)
                    np.testing.assert_equal(nlo.call_args[0][1], self.no_coils)

                positions = nlo_retv.optimize.call_args[0][0]

                self._coil_positioning_b4_update(pos_opt, positions)

                with patch.object(pos_opt, "_optimise_currents"):
                    pos_opt.update_positions(positions)

                self._coil_positioning(pos_opt, positions, eq.coilset)

        np.testing.assert_equal(pos_opt.n_L, self.no_coils)

    def get_i_max(self, patch_loc, nlo_retv, pos_opt, eq):
        with patch(patch_loc + "nlopt.opt", name="nlopt.opt", return_value=nlo_retv):
            with patch(patch_loc + "process_NLOPT_result"):
                with patch.object(pos_opt, "update_positions"):
                    pos_opt.I_star = None
                    pos_opt(eq, None)

                    i_max_b4 = pos_opt._get_i_max()
                    pos_opt.flag_PFR = not pos_opt.flag_PFR
                    i_max_after = pos_opt._get_i_max()
        assert not np.array_equal(i_max_after, i_max_b4)

    @pytest.mark.parametrize("function_name", ["get_i_max", "PFcoil_region_ordering"])
    def test_optimiser(self, function_name):

        function = getattr(self, function_name)

        self._track_setup()
        self._coilset_setup(materials=True)

        self.coil_names = (
            self.coil_names[:4] + self.coil_names[6:] + self.coil_names[4:6]
        )
        regions = {}

        for i in range(2):
            coil = self.coilset.coils[f"PF_{i+5}"]
            mn = {"x": coil.x - 2 * coil.dx, "z": coil.z - 2 * coil.dz}
            mx = {"x": coil.x + 2 * coil.dx, "z": coil.z + 2 * coil.dz}
            regions[PF_COIL_NAME.format(i + 5)] = Loop(**_make_square(mn, mx))

        self.no_coils += len(regions)

        self.pos_dict = {}
        for name, coil in self.coilset.coils.items():
            self.pos_dict[name] = (coil.x, coil.z)

        # Stripped from
        # BLUEPRINT.equilibria.run.EquilibriumProblem.optimise_positions
        solenoid = self.coilset.get_solenoid()
        CS_x = solenoid.radius
        CS_zmin = solenoid.z_min
        CS_zmax = solenoid.z_max
        CS_gap = solenoid.gap

        # Stripped from
        # BLUEPRINT.examples.equilibria.single_null
        max_PF_current = 25e6  # [A]
        PF_Fz_max = 400e6  # [N]
        CS_Fz_sum = 300e6  # [N]
        CS_Fz_sep = 250e6  # [N]

        patch_loc = "bluemira.equilibria.optimiser."

        nlo_retv = MagicMock(name="opt.optimize()")

        eq = MagicMock(name="Equilibrium()")
        eq.coilset = self.coilset

        with patch(patch_loc + "FBIOptimiser"):

            pos_opt = PositionOptimiser(
                CS_x=CS_x,
                CS_zmin=CS_zmin,
                CS_zmax=CS_zmax,
                CS_gap=CS_gap,
                max_PF_current=max_PF_current,
                max_fields=self.coilset.get_max_fields(),
                PF_Fz_max=PF_Fz_max,
                CS_Fz_sum=CS_Fz_sum,
                CS_Fz_sep=CS_Fz_sep,
                psi_values=None,
                pfcoiltrack=self.TF_track,
                pf_coilregions=regions,
                pf_exclusions=None,
                CS=True,
                plot=False,
                gif=False,
            )

            function(patch_loc, nlo_retv, pos_opt, eq)


class TestCoilsetOptimiser:
    @classmethod
    def setup_class(cls):
        coil = Coil(
            x=1.5,
            z=6.0,
            current=1e6,
            dx=0.25,
            dz=0.5,
            j_max=1e-5,
            b_max=100,
            ctype="PF",
            name="PF_2",
        )
        circuit = SymmetricCircuit(coil)

        coil2 = Coil(
            x=4.0,
            z=10.0,
            current=2e6,
            dx=0.5,
            dz=0.33,
            j_max=5.0e-6,
            b_max=50.0,
            name="PF_1",
        )

        coil3 = Coil(
            x=4.0,
            z=20.0,
            current=7e6,
            dx=0.5,
            dz=0.33,
            j_max=None,
            b_max=50.0,
            name="PF_3",
        )
        cls.coilset = CoilSet([circuit, coil2, coil3])

        max_coil_shifts = {
            "x_shifts_lower": -2.0,
            "x_shifts_upper": 1.0,
            "z_shifts_lower": -1.0,
            "z_shifts_upper": 5.0,
        }

        cls.pfregions = {}
        for coil in cls.coilset._ccoils:
            xu = coil.x + max_coil_shifts["x_shifts_upper"]
            xl = coil.x + max_coil_shifts["x_shifts_lower"]
            zu = coil.z + max_coil_shifts["z_shifts_upper"]
            zl = coil.z + max_coil_shifts["z_shifts_lower"]

            rect = Loop(x=[xl, xu, xu, xl, xl], z=[zl, zl, zu, zu, zl])

            cls.pfregions[coil.name] = rect

        cls.optimiser = CoilsetOptimiser(cls.coilset, cls.pfregions)

    def test_modify_coilset(self):
        # Read
        coilset_state, substates = self.optimiser.read_coilset_state(self.coilset)
        # Modify vectors
        x, z, currents = np.array_split(coilset_state, substates)
        x += 1.1
        z += 0.6
        currents += 0.99
        updated_coilset_state = np.concatenate((x, z, currents))
        self.optimiser.set_coilset_state(updated_coilset_state)

        coilset_state, substates = self.optimiser.read_coilset_state(self.coilset)
        state_x, state_z, state_i = np.array_split(coilset_state, substates)
        assert np.allclose(state_x, x)
        assert np.allclose(state_z, z)
        assert np.allclose(state_i, currents)

    def test_current_bounds(self):
        n_control_currents = len(self.coilset.get_control_currents())
        user_max_current = 2.0e9
        user_current_limits = (
            user_max_current * np.ones(n_control_currents) / self.optimiser.scale
        )
        coilset_current_limits = self.optimiser.coilset.get_max_currents(0.0)

        control_current_limits = np.minimum(user_current_limits, coilset_current_limits)
        bounds = (-control_current_limits, control_current_limits)

        assert n_control_currents == len(user_current_limits)
        assert n_control_currents == len(coilset_current_limits)

        optimiser_current_bounds = self.optimiser.get_current_bounds(user_max_current)
        assert np.allclose(bounds[0], optimiser_current_bounds[0])
        assert np.allclose(bounds[1], optimiser_current_bounds[1])

        # print(self.optimiser.coilset.get_max_currents(0.0))
        # print(self.optimiser.get_current_bounds(10.0) / self.optimiser.scale)

        # self.optimiser.get_current_bounds()

        # optimiser_maxima = 0.9
        # i_max = self.coilset.get_max_currents(max_currents)


# Recursion test and comparision between scipy and NLopt implementation of
# Breakdown optimisation:
class BreakdownOptimiserOLD:
    """
    Optimiser for the premagnetisation phase of the plasma. The sum of the
    PF coil currents is minimised (operating at maximum CS module voltages).
    Constraints are applied directly within the optimiser:
    - the maximum absolute current value per PF coil
    - the maxmimum poloidal magnetic field inside the breakdown zone
    - peak field inside the conductors

    Parameters
    ----------
    x_zone: float
        The X coordinate of the centre of the circular breakdown zone [m]
    z_zone: float
        The Z coordinate of the centre of the circular breakdown zone [m]
    r_zone: float
        The radius of the circular breakdown zone [m]
    b_zone_max: float
        The maximum field constraint inside the breakdown zone [T]
    max_currents: np.array(coils)
        The array of maximum coil currents [A]
    max_fields: np.array(n_coils)
        The array of maximum poloidal field [T]
    PF_Fz_max: float
        The maximum absolute vertical on a PF coil [N]
    CS_Fz_sum: float
        The maximum absolute vertical on all CS coils [N]
    CS_Fz_sep: float
        The maximum Central Solenoid vertical separation force [N]
    """

    def __init__(
        self,
        x_zone,
        z_zone,
        r_zone,
        b_zone_max,
        max_currents,
        max_fields,
        PF_Fz_max,
        CS_Fz_sum,
        CS_Fz_sep,
        **kwargs,
    ):
        self.scale = 1e6
        self.R_zone = x_zone
        self.Z_zone = z_zone
        self.r_zone = r_zone
        self.B_zone_max = b_zone_max
        self._I_max = max_currents
        self.B_max = max_fields
        self.PF_Fz_max = PF_Fz_max / self.scale
        self.CS_Fz_sum = CS_Fz_sum / self.scale
        self.CS_Fz_sep = CS_Fz_sep / self.scale
        self.meshed = False
        self.I_max = None
        self.constraint_tol = kwargs.get("constraint_tol", 1e-3)

    def __call__(self, eq):
        """
        Optimise the coil currents in an breakdown.

        Parameters
        ----------
        eq: Breakdown
            The breakdown to optimise the positions for

        Returns
        -------
        opt_currents: np.array(n_coils)
            The optimal currents for the controlled coils.
        """
        self.n_PF, self.n_CS = eq.coilset.n_PF, eq.coilset.n_CS
        self.n_C = eq.coilset.n_coils
        self.eq = eq
        if not self.meshed:
            self.zone = np.array(
                make_circle_arc(self.r_zone, self.R_zone, self.Z_zone, n_points=20)
            )
            self.meshed = True

        self.eq.set_forcefield()

        self._I = eq.coilset.get_control_currents() / self.scale

        if self.I_max is None:
            self.I_max = eq.coilset.get_max_currents(self._I_max) / self.scale

        return self.optimise()

    def constrain_breakdown(self, x):
        """
        Constraint on the maximum field value insize the breakdown zone

        \t:math:`\\text{max}(B_p(\\mathbf{p})) \\forall \\mathbf{p}`
        \t:math:`\\in \\delta\\Omega \\leq B_{max}`
        """
        self.eq.coilset.set_control_currents(x * self.scale)
        B = self.eq.Bp(*self.zone)
        return self.B_zone_max - B

    def constrain_field(self, x):
        """
        Constraint on the maximum field within a coil

        \t:math:`B_{p_{i}} \\leq B_{p_{i_{max}}}`
        """
        B, _ = self.eq.force_field.calc_field(x * self.scale)
        return self.B_max - B

    def constrain_force(self, x):
        """
        Constraint of the forces in the coilset.
        """
        F, _ = self.eq.force_field.calc_force(x * self.scale)
        F /= self.scale
        constraint = np.zeros(self.n_PF + self.n_CS + 1)
        pf_fz = F[: self.n_PF, 1]  # vertical force on PF coils
        constraint[: self.n_PF] = self.PF_Fz_max - np.abs(pf_fz)
        cs_fz = F[self.n_PF :, 1]  # vertical force on CS coils
        cs_fz_sum = np.sum(cs_fz)  # vertical force on CS stack
        constraint[self.n_PF] = self.CS_Fz_sum - np.abs(cs_fz_sum)
        for j in range(self.n_CS - 1):  # evaluate each gap
            f_sep = np.sum(cs_fz[: j + 1]) - np.sum(cs_fz[j + 1 :])
            # CS seperation
            constraint[self.n_PF + 1 + j] = self.CS_Fz_sep - f_sep
        return constraint

    def f_maxflux(self, x):
        """
        Objective function for total current sum minimisation

        \t:math:`\\sum_i^{n_C} \\lvert I_i\\rvert`
        """
        self.eq.coilset.set_control_currents(x * self.scale)
        return -self.eq.psi(self.R_zone, self.Z_zone)

    def f_max_current(self, x):
        """
        Current maximisation objective function
        """
        self.eq.coilset.set_control_currents(x * self.scale)
        return -np.sum(np.abs(x[:]))

    def optimise(self):
        """
        Optimiser handle for the BreakdownOptimiser object. Called on __call__
        """
        bounds = [[-self.I_max[i], self.I_max[i]] for i in range(self.n_C)]
        constraints = [
            {"type": "ineq", "fun": self.constrain_breakdown},
            {"type": "ineq", "fun": self.constrain_field},
            {"type": "ineq", "fun": self.constrain_force},
        ]

        x0 = 1e-6 * np.ones(self.n_C)
        res = minimize(
            self.f_maxflux,
            x0,
            constraints=constraints,
            bounds=bounds,
            method="SLSQP",
            options={"eps": 1e-6, "maxiter": 1000},
        )
        currents = process_scipy_result(res)
        self._I_star = currents * self.scale
        return currents * self.scale

    def update_current_constraint(self, max_currents):
        """
        Update the current vector bounds. Must be called prior to optimise
        """
        self.I_max = max_currents / self.scale


class TestScipyNLoptOptimiser:
    @classmethod
    def setup_class(cls):
        # Parameter set
        cls.R_0 = 8.938
        cls.B_t = 5.63
        cls.I_p = 19.6
        cls.l_i = 0.8
        cls.beta_p = 1.107
        cls.k_95 = 1.59
        cls.d_95 = 0.33

        # Constraints
        cls.PF_Fz_max = 450e6
        cls.CS_Fz_sum = 300e6
        cls.CS_Fz_sep = 350e6

        _coilset_setup(cls, materials=True)

    def runner(self, optimiser_class):
        self.coilset.mesh_coils(0.3)
        optimiser = optimiser_class(
            9.42,
            0.16,
            r_zone=1.5,  # Not sure about this but hey
            b_zone_max=0.003,
            max_currents=self.coilset.get_max_currents(0),
            max_fields=self.coilset.get_max_fields(),
            PF_Fz_max=self.PF_Fz_max,
            CS_Fz_sum=self.CS_Fz_sum,
            CS_Fz_sep=self.CS_Fz_sep,
        )

        grid = Grid(0.1, self.R_0 * 2, -1.5 * self.R_0, 1.5 * self.R_0, 100, 100)
        bd = Breakdown(deepcopy(self.coilset), grid, psi=None, R_0=self.R_0)

        currents = optimiser(bd)
        bd.coilset.set_control_currents(currents)
        return bd, optimiser, currents

    def test_scipy_nlopt(self):
        """
        Carry out the same optimisation in scipy and NLopt using the same
        algorithm.
        """
        scipy_bd, scipy_optimiser, scipy_currents = self.runner(BreakdownOptimiserOLD)
        nlopt_bd, nlopt_optimiser, nlopt_currents = self.runner(BreakdownOptimiser)

        scipy_psi_bd = scipy_bd.psi(scipy_optimiser.R_zone, scipy_optimiser.Z_zone)
        nlopt_psi_bd = nlopt_bd.psi(nlopt_optimiser.x_zone, nlopt_optimiser.z_zone)

        assert np.isclose(scipy_psi_bd, nlopt_psi_bd, rtol=1e-3)
        d_currents = np.round(nlopt_currents / scipy_currents, 1)
        assert (d_currents == np.ones(len(d_currents))).all()
        if tests.PLOTTING:
            f, ax = plt.subplots(1, 2)
            scipy_bd.plot(ax[0])
            scipy_bd.coilset.plot(ax[0])
            nlopt_bd.plot(ax[1])
            nlopt_bd.coilset.plot(ax[1])
            plt.show()


if __name__ == "__main__":
    pytest.main([__file__])
