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
import pytest

from bluemira.base.error import BuilderError
from bluemira.equilibria.coils import Coil
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import PictureFrame, PrincetonD, TripleArc
from bluemira.geometry.tools import boolean_cut, make_polygon
from eudemo.pf_coils.tools import (
    make_coil_mapper,
    make_coilset,
    make_pf_coil_path,
    make_solenoid,
)


class TestMakeCoilMapper:
    tracks = (
        PrincetonD(
            {"x1": {"value": 4}, "x2": {"value": 14}, "dz": {"value": 0}}
        ).create_shape(label="PrincetonD"),
        PictureFrame(
            {
                "x1": {"value": 4, "upper_bound": 5, "lower_bound": 0.3},
                "x2": {"value": 11.5, "upper_bound": 12, "lower_bound": 6},
                "ri": {"value": 0},
                "ro": {"value": 1},
                "z1": {"value": 8},
                "z2": {"value": -9},
            },
        ).create_shape(label="PFrame"),
        TripleArc().create_shape(label="TripleArc"),
    )

    @classmethod
    def setup_class(cls):
        exclusion1 = BluemiraFace(
            make_polygon([[6, 9, 9, 6], [0, 0, 0, 0], [0, 0, 20, 20]], closed=True)
        )
        exclusion2 = BluemiraFace(
            make_polygon([[9, 20, 20, 9], [0, 0, 0, 0], [-1, -1, 1, 1]], closed=True)
        )
        cls.exclusions = [exclusion1, exclusion2]

        cls.coils = [
            Coil(4, 9, current=1e6, j_max=1),
            Coil(9, -9, current=1e6, j_max=1),
            Coil(12, 0.1, current=1e6, j_max=1),
            Coil(6, -10, current=1e6, j_max=1),
        ]

    @pytest.mark.parametrize("track", tracks)
    def test_cuts(self, track):
        segments = boolean_cut(track, self.exclusions)
        actual_length = sum([seg.length for seg in segments])
        mapper = make_coil_mapper(track, self.exclusions, self.coils)
        interp_length = sum(
            [tool.geometry.length for tool in mapper.interpolators.values()]
        )
        assert np.isclose(actual_length, interp_length, rtol=1e-2)

    @pytest.mark.parametrize("track", tracks)
    def test_simple(self, track):
        mapper = make_coil_mapper(track, self.exclusions, self.coils)
        assert len(mapper.interpolators) == len(self.coils)


class TestMakeSolenoid:
    @pytest.mark.parametrize("n_CS", [1, 3, 5, 7])
    def test_odd_module_numbers(self, n_CS):
        coils = make_solenoid(4, 1, -1, 9, 0.1, 0, 0.1, n_CS)
        assert len(coils) == n_CS
        dzs = [c.dz for c in coils]
        if n_CS != 1:
            middle = (n_CS - 1) // 2
            dz_middle = dzs.pop(middle)
            assert np.allclose(dzs, dzs[0])
            assert np.isclose(dz_middle, dzs[0] * 2)
            assert np.isclose(coils[middle].z, -1 + 10 / 2)

    @pytest.mark.parametrize("n_CS", [2, 4, 6, 8])
    def test_even_modules_numbers(self, n_CS):
        coils = make_solenoid(4, 1, -1, 9, 0.1, 0, 0.1, n_CS)
        assert len(coils) == n_CS
        dzs = [c.dz for c in coils]
        assert np.allclose(dzs, dzs[0])

    def test_error_on_too_large_gaps(self):
        with pytest.raises(BuilderError):
            make_solenoid(4, 1, 0, 10, 1, 1, 1, 10)

    def test_error_on_equal_extrema(self):
        with pytest.raises(BuilderError):
            make_solenoid(4, 1, 1, 1, 0, 0, 0, 1)


class TestMakeCoilset:
    boundaries = (
        PrincetonD().create_shape(label="PrincetonD"),
        TripleArc().create_shape(label="TripleArc"),
    )

    @pytest.mark.parametrize("boundary", boundaries)
    def test_make_coilset(self, boundary):
        n_CS = 5
        n_PF = 6

        coilset = make_coilset(
            tf_boundary=boundary,
            R_0=9,
            kappa=1.6,
            delta=1.8,
            r_cs=3,
            tk_cs=0.4,
            g_cs=0.1,
            tk_cs_ins=0.1,
            tk_cs_cas=0.1,
            n_CS=n_CS,
            n_PF=n_PF,
            CS_jmax=10,
            CS_bmax=10,
            PF_jmax=10,
            PF_bmax=10,
        )

        assert coilset.n_coils() == n_PF + n_CS


class TestMakePfCoilPath:
    fixtures = (
        PrincetonD().create_shape(),
        PictureFrame().create_shape(),
    )

    @pytest.mark.parametrize("wire", fixtures)
    def test_make_pf_coil_path(self, wire):
        result = make_pf_coil_path(wire, offset_value=1.0)

        assert not result.is_closed()
