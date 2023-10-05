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

import numpy as np

from bluemira.equilibria.coils import Coil, CoilSet, SymmetricCircuit
from bluemira.equilibria.optimisation.problem import CoilsetPositionCOP
from bluemira.geometry.tools import make_polygon
from bluemira.utilities.positioning import PositionMapper, RegionInterpolator


class TestCoilsetOptimiser:
    @classmethod
    def setup_class(cls):
        circuit = SymmetricCircuit(
            Coil(
                x=1.5,
                z=6.0,
                current=1e6,
                dx=0.25,
                dz=0.5,
                j_max=1e-5,
                b_max=100,
                ctype="PF",
                name="PF_2",
            ),
            Coil(
                x=1.5,
                z=-6.0,
                current=1e6,
                dx=0.25,
                dz=0.5,
                j_max=1e-5,
                b_max=100,
                ctype="PF",
                name="PF_4",
            ),
        )

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
            j_max=np.nan,
            b_max=50.0,
            name="PF_3",
        )
        cls.coilset = CoilSet(circuit, coil2, coil3)

        max_coil_shifts = {
            "x_shifts_lower": -2.0,
            "x_shifts_upper": 1.0,
            "z_shifts_lower": -1.0,
            "z_shifts_upper": 5.0,
        }

        cls.pfregions = {}

        xup = cls.coilset.x[cls.coilset._control_ind] + max_coil_shifts["x_shifts_upper"]
        xlo = cls.coilset.x[cls.coilset._control_ind] + max_coil_shifts["x_shifts_lower"]
        zup = cls.coilset.z[cls.coilset._control_ind] + max_coil_shifts["z_shifts_upper"]
        zlo = cls.coilset.z[cls.coilset._control_ind] + max_coil_shifts["z_shifts_lower"]

        for name, xl, xu, zl, zu in zip(cls.coilset.name, xup, xlo, zup, zlo):
            cls.pfregions[name] = RegionInterpolator(
                make_polygon({"x": [xl, xu, xu, xl, xl], "z": [zl, zl, zu, zu, zl]})
            )

        cls.optimiser = CoilsetPositionCOP(
            cls.coilset, None, None, PositionMapper(cls.pfregions)
        )

    def test_modify_coilset(self):
        # Read
        coilset_state, substates = self.optimiser.read_coilset_state(
            self.coilset, self.optimiser.scale
        )
        # Modify vectors
        x, z, currents = np.array_split(coilset_state, substates)
        x += 1.1
        z += 0.6
        currents += 0.99

        updated_coilset_state = np.concatenate((x, z, currents))
        self.optimiser.set_coilset_state(
            self.optimiser.coilset, updated_coilset_state, self.optimiser.scale
        )

        coilset_state, substates = self.optimiser.read_coilset_state(
            self.coilset, self.optimiser.scale
        )
        state_x, state_z, state_i = np.array_split(coilset_state, substates)
        assert np.allclose(state_x, x)
        assert np.allclose(state_z, z)
        assert np.allclose(state_i, currents)

    def test_current_bounds(self):
        n_control_currents = len(self.coilset.current[self.coilset._control_ind])
        user_max_current = 2.0e9
        user_current_limits = (
            user_max_current * np.ones(n_control_currents) / self.optimiser.scale
        )
        coilset_current_limits = self.optimiser.coilset.get_max_current()

        control_current_limits = np.minimum(user_current_limits, coilset_current_limits)
        bounds = (-control_current_limits, control_current_limits)

        assert n_control_currents == len(user_current_limits)
        assert n_control_currents == len(coilset_current_limits)

        optimiser_current_bounds = self.optimiser.get_current_bounds(
            self.optimiser.coilset, user_max_current, self.optimiser.scale
        )
        assert np.allclose(bounds[0], optimiser_current_bounds[0])
        assert np.allclose(bounds[1], optimiser_current_bounds[1])

        # print(self.optimiser.coilset.get_max_currents(0.0))
        # print(self.optimiser.get_current_bounds(10.0) / self.optimiser.scale)

        # self.optimiser.get_current_bounds()

        # optimiser_maxima = 0.9
        # i_max = self.coilset.get_max_currents(max_currents)
