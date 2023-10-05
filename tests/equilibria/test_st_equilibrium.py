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
bluemira ST equilibrium recursion test
"""

import copy
import os
from pathlib import Path

import numpy as np
import pytest

from bluemira.base.file import get_bluemira_root
from bluemira.equilibria import (
    Coil,
    CoilSet,
    CustomProfile,
    Equilibrium,
    Grid,
    IsofluxConstraint,
    MagneticConstraintSet,
    PicardIterator,
    SymmetricCircuit,
)
from bluemira.equilibria.file import EQDSKInterface
from bluemira.equilibria.opt_problems import UnconstrainedTikhonovCurrentGradientCOP
from bluemira.equilibria.solve import DudsonConvergence
from bluemira.geometry.coordinates import get_area_2d


@pytest.mark.private
class TestSTEquilibrium:
    @classmethod
    def setup_class(cls):
        # Load reference and input data
        root = get_bluemira_root()
        private = os.path.split(root)[0]
        private = Path(private, "bluemira-private-data/equilibria/STEP_SPR_08")
        eq_name = "STEP_SPR08_BLUEPRINT.json"
        cls.eq_blueprint = Equilibrium.from_eqdsk(Path(private, eq_name))
        jeq_name = "jetto.eqdsk_out"
        filename = Path(private, jeq_name)
        cls.profiles = CustomProfile.from_eqdsk(filename)
        cls.jeq_dict = EQDSKInterface.from_file(filename)

    def test_equilibrium(self):
        build_tweaks = {
            "plot_fbe_evol": True,
            "plot_fbe": True,
            "sol_isoflux": True,
            "process_midplane_iso": True,
            "tikhonov_gamma": 1e-8,
            "fbe_convergence": "Dudson",
            "fbe_convergence_crit": 1.0e-6,
            "nx_number_x": 7,
            "nz_number_z": 8,
        }

        R_0 = 3.639
        A = 1.667
        i_p = self.jeq_dict.cplasma

        xc = np.array(
            [1.5, 1.5, 8.259059936102478, 8.259059936102478, 10.635505223274231]
        )
        zc = np.array([8.78, 11.3, 11.8, 6.8, 1.7])
        dxc = np.array([0.175, 0.25, 0.25, 0.25, 0.35])
        dzc = np.array([0.5, 0.4, 0.4, 0.4, 0.5])

        coils = []
        for i, (x, z, dx, dz) in enumerate(zip(xc, zc, dxc, dzc)):
            coil = SymmetricCircuit(
                Coil(x=x, z=z, dx=dx, dz=dz, name=f"PF_{i+1}", ctype="PF")
            )
            coils.append(coil)
        coilset = CoilSet(*coils)

        grid = Grid(
            x_min=0.0,
            x_max=max(xc + dxc) + 0.5,
            z_min=-max(zc + dzc),
            z_max=max(zc + dzc),
            nx=2 ** build_tweaks["nx_number_x"] + 1,
            nz=2 ** build_tweaks["nz_number_z"] + 1,
        )

        inboard_iso = [R_0 * (1.0 - 1 / A), 0.0]
        outboard_iso = [R_0 * (1.0 + 1 / A), 0.0]

        x = self.jeq_dict.xbdry
        z = self.jeq_dict.zbdry
        upper_iso = [x[np.argmax(z)], np.max(z)]
        lower_iso = [x[np.argmin(z)], np.min(z)]

        x_core = np.array([inboard_iso[0], upper_iso[0], outboard_iso[0], lower_iso[0]])
        z_core = np.array([inboard_iso[1], upper_iso[1], outboard_iso[1], lower_iso[1]])

        # Points chosen to replicate divertor legs in AH's FIESTA demo
        x_hfs = np.array(
            [
                1.42031,
                1.057303,
                0.814844,
                0.669531,
                0.621094,
                0.621094,
                0.645312,
                0.596875,
            ]
        )
        z_hfs = np.array(
            [4.79844, 5.0875, 5.37656, 5.72344, 6.0125, 6.6484, 6.82188, 7.34219]
        )
        x_lfs = np.array(
            [1.85625, 2.24375, 2.53438, 2.89766, 3.43047, 4.27813, 5.80391, 6.7]
        )
        z_lfs = np.array(
            [4.79844, 5.37656, 5.83906, 6.24375, 6.59063, 6.76406, 6.70625, 6.70625]
        )

        x_div = np.concatenate([x_lfs, x_lfs, x_hfs, x_hfs])
        z_div = np.concatenate([z_lfs, -z_lfs, z_hfs, -z_hfs])

        # Scale up Agnieszka isoflux constraints
        size_scaling = R_0 / 2.5
        x_div = size_scaling * x_div
        z_div = size_scaling * z_div

        xx = np.concatenate([x_core, x_div])
        zz = np.concatenate([z_core, z_div])

        constraint_set = MagneticConstraintSet(
            [
                IsofluxConstraint(
                    xx,
                    zz,
                    ref_x=inboard_iso[0],
                    ref_z=inboard_iso[1],
                    constraint_value=0.0,
                )
            ]
        )

        initial_psi = self._make_initial_psi(
            coilset,
            grid,
            constraint_set,
            R_0 + 0.5,
            0,
            i_p,
            build_tweaks["tikhonov_gamma"],
        )

        eq = Equilibrium(
            coilset, grid, self.profiles, force_symmetry=True, psi=initial_psi
        )
        opt_problem = UnconstrainedTikhonovCurrentGradientCOP(
            eq.coilset, eq, constraint_set, gamma=build_tweaks["tikhonov_gamma"]
        )

        criterion = DudsonConvergence(build_tweaks["fbe_convergence_crit"])

        fbe_iterator = PicardIterator(
            eq,
            opt_problem,
            plot=False,
            gif=False,
            relaxation=0.3,
            maxiter=400,
            convergence=criterion,
        )
        fbe_iterator()
        self.eq = eq
        self._test_equilibrium_good(eq)
        self._test_profiles_good(eq)

        # Verify by removing symmetry constraint and checking convergence
        eq.force_symmetry = False
        eq.set_grid(grid)
        fbe_iterator()
        # I probably exported the eq before it was regridded without symmetry..
        self._test_equilibrium_good(eq)

        self._test_profiles_good(eq)

    def _test_equilibrium_good(self, eq):
        assert np.isclose(eq.profiles.I_p, abs(self.jeq_dict.cplasma))
        lcfs = eq.get_LCFS()
        assert np.isclose(
            get_area_2d(*self.eq_blueprint.get_LCFS().xz),
            get_area_2d(*lcfs.xz),
            rtol=1e-2,
        )
        assert np.isclose(lcfs.center_of_mass[-1], 0.0)

    def _test_profiles_good(self, eq):
        """
        Test the profiles are the same shape. Normalisation won't be one: JETTO is
        fixed boundary and has a different plasma volume.
        """

        def scale(profile):
            return np.abs(profile) / np.max(np.abs(profile))

        jetto_pprime = self.jeq_dict.pprime
        jetto_ffprime = self.jeq_dict.ffprime

        psi_n = self.jeq_dict.psinorm
        bm_pprime_p = self.profiles.pprime(psi_n)
        bm_ffprime_p = self.profiles.ffprime(psi_n)

        bm_pprime = eq.profiles.pprime(psi_n)
        bm_ffprime = eq.profiles.ffprime(psi_n)
        assert np.allclose(bm_pprime, bm_pprime_p)
        assert np.allclose(bm_ffprime, bm_ffprime_p)
        assert np.isclose(max(bm_ffprime) / max(jetto_ffprime), abs(eq.profiles.scale))
        assert np.isclose(max(bm_pprime) / max(jetto_pprime), abs(eq.profiles.scale))

        jetto_pprime = scale(jetto_pprime)
        jetto_ffprime = scale(jetto_ffprime)
        bm_pprime = scale(bm_pprime)
        bm_ffprime = scale(bm_ffprime)
        assert np.allclose(jetto_pprime, bm_pprime)
        assert np.allclose(jetto_ffprime, bm_ffprime)

    def _make_initial_psi(
        self,
        coilset,
        grid,
        constraint_set,
        x_current,
        z_current,
        plasma_current,
        tikhonov_gamma,
    ):
        coilset_temp = copy.deepcopy(coilset)

        dummy = Coil(
            x=x_current,
            z=z_current,
            dx=0,
            dz=0,
            current=plasma_current,
            name="plasma_dummy",
        )

        coilset_temp.add_coil(dummy)
        coilset_temp.control = coilset.name

        eq = Equilibrium(
            coilset_temp, grid, self.profiles, force_symmetry=True, psi=None
        )
        opt_problem = UnconstrainedTikhonovCurrentGradientCOP(
            coilset, eq, constraint_set, gamma=tikhonov_gamma
        )
        opt_problem.optimise()

        # Note that this for some reason (incorrectly) only includes the psi from the
        # controlled coils and the plasma dummy psi contribution is not included...
        # which for some reason works better than with it.
        # proper mindfuck this... no idea why it wasn't working properly before, and
        # no idea why it works better with what is blatantly a worse starting solution.
        # Really you could just avoid adding the dummy plasma coil in the first place..
        # Perhaps the current centre is poorly estimated by R_0 + 0.5
        return coilset_temp.psi(grid.x, grid.z).copy() - np.squeeze(
            dummy.psi(grid.x, grid.z)
        )
