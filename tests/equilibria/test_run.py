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

from dataclasses import fields
from unittest import mock

import numpy as np
import pytest

from bluemira.base.constants import EPS
from bluemira.equilibria.run import (
    BreakdownCOPSettings,
    EQSettings,
    PulsedCoilsetDesign,
    PulsedCoilsetDesignFrame,
    Snapshot,
)


class TestSnapshot:
    class Emptyclass:  # noqa: D106
        pass

    def test_Snapshot_copy(self):
        classes = [self.Emptyclass() for _ in range(7)]
        snapshot = Snapshot(*classes)

        for cl, sn in zip(classes, fields(type(snapshot))):
            if sn.name not in ("constraints", "tfcoil"):
                assert cl != getattr(snapshot, sn.name)
            else:
                assert cl == getattr(snapshot, sn.name)


class TestPulsedCoilSetDesign:
    def setup_method(self):
        self.params = PulsedCoilsetDesignFrame.from_dict(
            {
                "A": {"value": 1, "unit": ""},
                "B_premag_stray_max": {"value": 2, "unit": ""},
                "C_Ejima": {"value": 3, "unit": ""},
                "I_p": {"value": 4, "unit": ""},
                "l_i": {"value": 5, "unit": ""},
                "R_0": {"value": 6, "unit": ""},
                "tau_flattop": {"value": 7, "unit": ""},
                "tk_sol_ib": {"value": 8, "unit": ""},
                "v_burn": {"value": 9, "unit": ""},
            }
        )

    class MyPulsedCoilset(PulsedCoilsetDesign):  # noqa: D106
        def optimise(self):
            return self.coilset

    def test_breakdown_settings(self):
        mypcs = self.MyPulsedCoilset(self.params, *[None] * 4)
        assert isinstance(mypcs._bd_settings, BreakdownCOPSettings)
        mypcs._bd_settings.n_B_stray_points = 9
        assert mypcs._bd_settings.n_B_stray_points == 9

        mypcs = self.MyPulsedCoilset(
            self.params, *[None] * 4, breakdown_settings={"n_B_stray_points": 10}
        )
        assert isinstance(mypcs._bd_settings, BreakdownCOPSettings)
        assert mypcs._bd_settings.n_B_stray_points == 10

        with pytest.raises(TypeError):
            mypcs = self.MyPulsedCoilset(
                self.params, *[None] * 4, breakdown_settings={"gamma": 1e-5}
            )

    def test_equilibrium_settings(self):
        mypcs = self.MyPulsedCoilset(self.params, *[None] * 4)
        assert isinstance(mypcs._eq_settings, EQSettings)
        mypcs._eq_settings.gamma = 9
        assert mypcs._eq_settings.gamma == 9

        mypcs = self.MyPulsedCoilset(
            self.params, *[None] * 4, equilibrium_settings={"gamma": 1e-5}
        )
        assert isinstance(mypcs._eq_settings, EQSettings)
        assert mypcs._eq_settings.gamma == pytest.approx(1e-5, rel=0, abs=EPS)

        with pytest.raises(TypeError):
            mypcs = self.MyPulsedCoilset(
                self.params, *[None] * 4, equilibrium_settings={"n_B_stray_points": 10}
            )

    @mock.patch("bluemira.equilibria.run.Snapshot", return_value="SNAP")
    def test_take_snapshot(self, snapshot, caplog):
        mypcs = self.MyPulsedCoilset(self.params, *[None] * 4)
        mypcs.take_snapshot("test", "eq", "coilset", "problem", "profiles")
        snapshot.assert_called_with("eq", "coilset", "problem", "profiles", limiter=None)

        assert mypcs.snapshots["test"] == "SNAP"
        # Overwrite snapshot
        mypcs.take_snapshot("test", "eq", "coilset", "problem", "profiles")
        assert len(caplog.messages) == 1

    # def test_run_premagnetisation(self):
    # def test_run_reference_equilibrium(self):

    @mock.patch("bluemira.equilibria.run.calc_psib", return_value=0)
    def test_calculate_sof_eof_fluxes(self, calc_psib):
        mypcs = self.MyPulsedCoilset(self.params, *[None] * 4)

        def r_premag(self, val=1):
            self.snapshots[self.BREAKDOWN] = mock.MagicMock()
            self.snapshots[self.BREAKDOWN].eq.breakdown_psi = val

        with mock.patch.object(
            self.MyPulsedCoilset, "run_premagnetisation", new=r_premag
        ):
            # all the same because calc_psib is mocked
            out_val = mypcs.calculate_sof_eof_fluxes()
            np.testing.assert_allclose(out_val, (0, -63))
            r_premag(mypcs, 2)
            mypcs.calculate_sof_eof_fluxes()
            np.testing.assert_allclose(out_val, (0, -63))
            mypcs.calculate_sof_eof_fluxes(3)
            np.testing.assert_allclose(out_val, (0, -63))

        # calc_psib called with different values
        psi_premag2pi = np.array([call[0][0] for call in calc_psib.call_args_list])
        np.testing.assert_allclose(psi_premag2pi[0] * 2, psi_premag2pi[1])
        np.testing.assert_allclose(psi_premag2pi[0] * 3, psi_premag2pi[2])

    # def test_get_sof_eof_opt_problems(self):

    # def test_converge_and_snapshot(self):
