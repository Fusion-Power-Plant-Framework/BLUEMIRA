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
from pathlib import Path

import numpy as np
import pytest

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.geometry.coordinates import Coordinates
from bluemira.radiation_transport.advective_transport import ChargedParticleSolver
from bluemira.radiation_transport.error import AdvectionTransportError

TEST_PATH = get_bluemira_path("radiation_transport/test_data", subfolder="tests")
EQ_PATH = get_bluemira_path("equilibria", subfolder="data")


class TestChargedParticleInputs:
    def test_bad_fractions(self):
        params = {
            "f_lfs_lower_target": 0.1,
            "f_hfs_lower_target": 0.1,
            "f_lfs_upper_target": 0.1,
            "f_hfs_upper_target": 0.1,
        }

        with pytest.raises(AdvectionTransportError):
            ChargedParticleSolver(params, None)

        params = {
            "f_lfs_lower_target": 0.9,
            "f_hfs_lower_target": 0.9,
            "f_lfs_upper_target": 0,
            "f_hfs_upper_target": 0.9,
        }
        with pytest.raises(AdvectionTransportError):
            ChargedParticleSolver(params, None)

    def test_TypeError_given_unknown_parameter(self):
        params = {"not_a_param": 0.1}

        with pytest.raises(TypeError) as type_error:
            ChargedParticleSolver(params, None)
        assert "not_a_param" in str(type_error)


class TestChargedParticleRecursionSN:
    @classmethod
    def setup_class(cls):
        eq_name = "EU-DEMO_EOF.json"
        filename = Path(EQ_PATH, eq_name)
        eq = Equilibrium.from_eqdsk(filename)
        fw_name = "first_wall.json"
        filename = Path(TEST_PATH, fw_name)
        fw = Coordinates.from_json(filename)

        cls.params = {
            "P_sep_particle": 100,
            "f_p_sol_near": 0.50,
            "fw_lambda_q_near_omp": 0.05,
            "fw_lambda_q_far_omp": 0.05,
            "f_lfs_lower_target": 0.75,
            "f_hfs_lower_target": 0.25,
            "f_lfs_upper_target": 0,
            "f_hfs_upper_target": 0,
        }

        solver = ChargedParticleSolver(cls.params, eq, dx_mp=0.001)
        x, z, hf = solver.analyse(fw)
        cls.x, cls.z, cls.hf = np.array(x), np.array(z), np.array(hf)
        cls.solver = solver

    def test_recursion(self):
        assert np.isclose(np.max(self.hf), 5.379, rtol=1e-2)
        assert np.argmax(self.hf) == 0
        assert np.isclose(np.sum(self.hf), 399, rtol=1e-2)

    def test_n_intersections(self):
        """
        Because it is a single null, we expect the same number of flux surfaces LFS and
        HFS.
        """
        len_ob_lfs = len(self.solver.flux_surfaces_ob_lfs)
        len_ob_hfs = len(self.solver.flux_surfaces_ob_hfs)

        assert len_ob_hfs == len_ob_lfs
        assert len_ob_lfs + len_ob_hfs == len(self.x)
        assert len(self.solver.flux_surfaces) == len(self.x)

    def test_integrals(self):
        n_fs = len(self.solver.flux_surfaces)
        x_lfs = self.x[:n_fs]
        x_hfs = self.x[n_fs:]
        z_lfs = self.z[:n_fs]
        z_hfs = self.z[n_fs:]
        hf_lfs = self.hf[:n_fs]
        hf_hfs = self.hf[n_fs:]

        dx_lfs = x_lfs[:-1] - x_lfs[1:]
        dz_lfs = z_lfs[:-1] - z_lfs[1:]
        d_lfs = np.hypot(dx_lfs, dz_lfs)
        q_lfs = sum(hf_lfs[:-1] * d_lfs * (x_lfs[:-1] + 0.5 * abs(dx_lfs)))

        dx_hfs = x_hfs[:-1] - x_hfs[1:]
        dz_hfs = z_hfs[:-1] - z_hfs[1:]
        d_hfs = np.hypot(dx_hfs, dz_hfs)
        q_hfs = sum(hf_hfs[:-1] * d_hfs * (x_hfs[:-1] + 0.5 * abs(dx_hfs)))
        true_total = self.params["P_sep_particle"]
        assert np.isclose(q_lfs + q_hfs, true_total, rtol=2e-2)

    def test_geometry_handling(self):
        """
        Trying screwing up the geometry and get the same results.
        """
        solver = ChargedParticleSolver(self.params, self.solver.eq, dx_mp=0.001)
        fw = deepcopy(self.solver.first_wall)
        fw.open()
        fw.reverse()
        x, z, hf = solver.analyse(fw)
        assert solver.first_wall.closed
        assert solver.first_wall.check_ccw()

        assert np.allclose(self.hf, hf)
        assert np.allclose(self.x, x)
        assert np.allclose(self.z, z)

    def test_plotting(self):
        ax = self.solver.plot(show=True)
        assert len(ax.lines) > 2


class TestChargedParticleRecursionDN:
    @classmethod
    def setup_class(cls):
        eq_name = "DN-DEMO_eqref.json"
        filename = Path(EQ_PATH, eq_name)
        eq = Equilibrium.from_eqdsk(filename)
        fw_name = "DN_fw_shape.json"
        filename = Path(TEST_PATH, fw_name)
        fw = Coordinates.from_json(filename)

        cls.params = {
            "P_sep_particle": 140,
            "f_p_sol_near": 0.65,
            "fw_lambda_q_near_omp": 0.003,
            "fw_lambda_q_far_omp": 0.1,
            "fw_lambda_q_near_imp": 0.003,
            "fw_lambda_q_far_imp": 0.1,
            "f_lfs_lower_target": 0.9 * 0.5,
            "f_hfs_lower_target": 0.1 * 0.5,
            "f_lfs_upper_target": 0.9 * 0.5,
            "f_hfs_upper_target": 0.1 * 0.5,
        }

        solver = ChargedParticleSolver(cls.params, eq, dx_mp=0.001)
        x, z, hf = solver.analyse(fw)
        cls.x, cls.z, cls.hf = np.array(x), np.array(z), np.array(hf)
        cls.solver = solver

    def test_recursion(self):
        assert np.isclose(np.max(self.hf), 86.194, rtol=1e-2)
        assert np.isclose(np.sum(self.hf), 830.6, rtol=1e-2)

    def test_analyse_DN(self, caplog):
        fw = deepcopy(self.solver.first_wall)
        self.solver.flux_surfaces_ob_hfs = []
        self.solver.flux_surfaces_ob_lfs = []
        x_sep_omp, x_wall_limit = self.solver._get_sep_out_intersection(outboard=True)

        x = x_sep_omp + 1e-3
        while x < x_wall_limit + 2e-3:
            lfs, hfs = self.solver._make_flux_surfaces(x, self.solver._o_point.z)
            self.solver.flux_surfaces_ob_lfs.append(lfs)
            self.solver.flux_surfaces_ob_hfs.append(hfs)
            x += 1e-3

        fs_before_pop = self.solver.flux_surfaces
        self.solver._clip_flux_surfaces(fw)
        fs_after_pop = self.solver.flux_surfaces
        assert len(fs_before_pop) > len(fs_after_pop)
        assert "No intersection detected" in caplog.text
