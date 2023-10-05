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

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bluemira.base.components import PhysicalComponent
from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.error import EquilibriaError
from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemGradShafranovFixedBoundary,
)
from bluemira.equilibria.fem_fixed_boundary.utilities import create_mesh
from bluemira.equilibria.profiles import DoublePowerFunc, LaoPolynomialFunc
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_polygon

DATA_DIR = get_bluemira_path(
    "equilibria/fem_fixed_boundary/test_generated_data", subfolder="tests"
)


def parameterisation_fixture_not_fully_init(
    test_no: int, solver_kwargs, p_prime=None, ff_prime=None, mesh=None
) -> FemGradShafranovFixedBoundary:
    """
    This function is to parameterise test_not_fully_init without creating the
    parametrize value which stop parallelised tests running
    """
    if test_no == 1:
        return FemGradShafranovFixedBoundary(**solver_kwargs)
    if test_no == 2:
        return FemGradShafranovFixedBoundary(p_prime, ff_prime, **solver_kwargs)
    return FemGradShafranovFixedBoundary(mesh=mesh, **solver_kwargs)


class TestFemGradShafranovFixedBoundary:
    def setup_method(self):
        lcfs_shape = make_polygon({"x": [7, 10, 7], "z": [-4, 0, 4]}, closed=True)
        lcfs_face = BluemiraFace(lcfs_shape)
        plasma = PhysicalComponent("plasma", lcfs_face)
        plasma.shape.mesh_options = {"lcar": 0.3, "physical_group": "plasma_face"}
        plasma.shape.boundary[0].mesh_options = {"lcar": 0.3, "physical_group": "lcfs"}
        self.mesh = create_mesh(
            plasma, DATA_DIR, "fixed_boundary_example", "fixed_boundary_example.msh"
        )

        self.p_prime = LaoPolynomialFunc([2, 3, 1])
        self.ff_prime = DoublePowerFunc([1.5, 2])
        self.solver_kwargs = {
            "R_0": 9,
            "I_p": 17e6,
            "B_0": 5,
            "p_order": 1,
            "max_iter": 2,
            "iter_err_max": 1.0,
            "relaxation": 0.0,
        }
        self.optional_init_solver = FemGradShafranovFixedBoundary(**self.solver_kwargs)
        self.flux_func_init_solver = FemGradShafranovFixedBoundary(
            self.p_prime, self.ff_prime, **self.solver_kwargs
        )
        self.mesh_init_solver = FemGradShafranovFixedBoundary(
            mesh=self.mesh, **self.solver_kwargs
        )
        self.full_init_solver = FemGradShafranovFixedBoundary(
            self.p_prime, self.ff_prime, self.mesh, **self.solver_kwargs
        )

    @classmethod
    def teardown_method(cls):
        plt.close()

    @pytest.mark.parametrize("plot", [False, True])
    def test_all_optional_init_12(self, plot):
        mod_current = 20e6
        self.optional_init_solver.set_profiles(
            self.p_prime, self.ff_prime, I_p=mod_current
        )
        self.optional_init_solver.set_mesh(self.mesh)
        self.optional_init_solver.solve(plot=plot)
        assert np.isclose(self.optional_init_solver._curr_target, mod_current)

    @pytest.mark.parametrize("plot", [False, True])
    def test_all_optional_init_21(self, plot):
        mod_current = 20e6
        self.optional_init_solver.set_mesh(self.mesh)
        self.optional_init_solver.set_profiles(
            self.p_prime, self.ff_prime, I_p=mod_current
        )
        self.optional_init_solver.solve(plot=plot)
        assert np.isclose(self.optional_init_solver._curr_target, mod_current)

    @pytest.mark.parametrize("test_no", [1, 2, 3])
    def test_not_fully_init(self, test_no):
        solver = parameterisation_fixture_not_fully_init(
            test_no, self.solver_kwargs, self.p_prime, self.ff_prime, self.mesh
        )
        with pytest.raises(EquilibriaError):
            solver.solve()
