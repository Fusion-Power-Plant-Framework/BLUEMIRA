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

import json
from pathlib import Path

import numpy as np
import pytest

from bluemira.base.constants import EPS, raw_uc
from bluemira.base.file import get_bluemira_path
from bluemira.plasma_physics.reactions import E_DD_fusion, E_DT_fusion, reactivity


class TestReactionEnergies:
    def _msg(self, e, v):
        delta = e - v
        relate = "higher" if delta > 0 else "lower"
        return f"E=mc^2 value {delta * 1e-6:.2f} MeV {relate} than Kikuchi reference."

    def test_DT(self):
        e_dt_kikuchi = (3.5 + 14.1) * 1e6
        e, v = E_DT_fusion(), e_dt_kikuchi
        assert np.isclose(e, v, rtol=1e-3), self._msg(e, v)

    def test_DD(self):
        e_dd_kikuchi = np.array([1.01 + 3.02, 0.82 + 2.45]) * 1e6
        e, v = E_DD_fusion(), np.average(e_dd_kikuchi)
        assert np.isclose(e, v, rtol=1e-3), self._msg(e, v)


@pytest.fixture()
def _xfail_DD_He3p_erratum_erratum(request):
    """
    As far as I can tell, there is either something wrong with the parameterisation,
    or more likely with the data presented in:

    H.-S. Bosch and G.M. Hale 1993 Nucl. Fusion 33 1919
    """
    t = request.getfixturevalue("temp_kev")
    if t == pytest.approx(1.3, rel=0, abs=EPS):
        request.node.add_marker(pytest.mark.xfail(reason="Error in erratum data?"))


class TestReactivity:
    """
    H.-S. Bosch and G.M. Hale 1993 Nucl. Fusion 33 1919
    """

    path = get_bluemira_path("plasma_physics/test_data", subfolder="tests")
    filename = "reactivity_Bosch_Hale_1993.json"
    with open(Path(path, filename)) as file:
        data = json.load(file)

    temp = np.array(data["temperature_kev"])
    sv_DT = np.array(data["sv_DT_m3s"])
    sv_DHe3 = np.array(data["sv_DHe3_m3s"])  # noqa: N815
    sv_DD_He3p = np.array(data["sv_DD_He3p_m3s"])
    sv_DD_Tp = np.array(data["sv_DD_Tp_m3s"])

    @pytest.mark.parametrize(
        ("method", "rtol"), [("Bosch-Hale", 0.0025), ("PLASMOD", 0.1)]
    )
    @pytest.mark.parametrize(("temp_kev", "sigmav"), np.c_[temp, sv_DT])
    def test_Bosch_Hale_DT(self, temp_kev, sigmav, method, rtol):
        temp_k = raw_uc(temp_kev, "keV", "K")
        result = reactivity(temp_k, reaction="D-T", method=method)
        np.testing.assert_allclose(result, sigmav, rtol=rtol, atol=0)

    @pytest.mark.parametrize(("temp_kev", "sigmav"), np.c_[temp, sv_DHe3])
    def test_Bosch_Hale_DHe(self, temp_kev, sigmav):
        temp_k = raw_uc(temp_kev, "keV", "K")
        result = reactivity(temp_k, reaction="D-He3", method="Bosch-Hale")
        np.testing.assert_allclose(result, sigmav, rtol=0.003, atol=0)

    @pytest.mark.parametrize(("temp_kev", "sigmav"), np.c_[temp, sv_DD_He3p])
    @pytest.mark.usefixtures("_xfail_DD_He3p_erratum_erratum")
    def test_Bosch_Hale_DD_He3p(self, temp_kev, sigmav):
        temp_k = raw_uc(temp_kev, "keV", "K")
        result = reactivity(temp_k, reaction="D-D1", method="Bosch-Hale")
        np.testing.assert_allclose(result, sigmav, rtol=0.003, atol=0)

    @pytest.mark.parametrize(("temp_kev", "sigmav"), np.c_[temp, sv_DD_Tp])
    def test_Bosch_Hale_DD_Tp(self, temp_kev, sigmav):
        temp_k = raw_uc(temp_kev, "keV", "K")
        result = reactivity(temp_k, reaction="D-D2", method="Bosch-Hale")
        np.testing.assert_allclose(result, sigmav, rtol=0.0035, atol=0)
