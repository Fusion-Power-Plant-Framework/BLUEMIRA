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

import pytest
import os
import numpy as np
import json
from bluemira.base.file import get_bluemira_path
from bluemira.utilities.tools import NumpyJSONEncoder, is_num, asciistr, dot, norm, cross


class TestNumpyJSONEncoder:
    def test_childclass(self):
        fp = get_bluemira_path("bluemira/utilities/test_data", subfolder="tests")
        fn = os.sep.join([fp, "testJSONEncoder.json"])
        d = {"x": np.array([1, 2, 3.4, 4]), "y": [1, 3], "z": 3, "a": "aryhfdhsdf"}
        with open(fn, "w") as file:
            json.dump(d, file, cls=NumpyJSONEncoder)
        with open(fn, "r") as file:
            dd = json.load(file)
        for k, v in d.items():
            for kk, vv in dd.items():
                if k == kk:
                    if isinstance(v, np.ndarray):
                        assert v.tolist() == vv
                    else:
                        assert v == vv


def test_is_num():
    vals = [0, 34.0, 0.0, -0.0, 34e183, 28e-182, np.pi, np.inf]
    for v in vals:
        assert is_num(v) is True

    vals = [True, False, np.nan, object()]
    for v in vals:
        assert is_num(v) is False


class TestAsciiStr:
    def test_asciistr(self):
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i in range(52):
            assert asciistr(i + 1) == alphabet[: i + 1]

        with pytest.raises(ValueError):
            asciistr(53)


class TestEinsumNorm:
    def test_norm(self):
        val = np.random.rand(999, 3)
        np.testing.assert_allclose(norm(val, axis=1), np.linalg.norm(val, axis=1))
        np.testing.assert_allclose(norm(val, axis=0), np.linalg.norm(val, axis=0))

    def test_raise(self):
        val = np.random.rand(999, 3)

        with pytest.raises(ValueError):
            norm(val, axis=3)


class TestEinsumDot:
    def test_dot(self):
        val3 = np.random.rand(999, 3, 3)
        val2 = np.random.rand(999, 3)
        val = np.random.rand(3)

        # ab, bc -> ac
        np.testing.assert_allclose(dot(val2, val2.T), np.dot(val2, val2.T))

        # abc, acd -> abd
        dv = dot(val3, val3)
        for no, i in enumerate(val3):
            np.testing.assert_allclose(dv[no], np.dot(i, i))

        # abc, c -> ab
        np.testing.assert_allclose(dot(val3, val), np.dot(val3, val))

        # a, abc -> ac | ab, abc -> ac | abc, bc -> ac -- undefined behaviour
        for (a, b) in [(val, val3.T), (val2, val3), (val3, val3[1:])]:
            with pytest.raises(ValueError):
                dot(a, b)

        # ab, b -> a
        np.testing.assert_allclose(dot(val2, val), np.dot(val2, val))

        # a, ab -> b
        np.testing.assert_allclose(dot(val, val2.T), np.dot(val, val2.T))

        # 'a, a -> ...'
        np.testing.assert_allclose(dot(val, val), np.dot(val, val))


class TestEinsumCross:
    def test_cross(self):
        val3 = np.random.rand(999, 3)
        val2 = np.random.rand(999, 2)
        val = np.random.rand(999)

        for i, v in enumerate([val2, val3], start=2):
            np.testing.assert_allclose(cross(v, v), np.cross(v, v))

        np.testing.assert_allclose(cross(val, val), val ** 2)

    def test_raises(self):
        val = np.random.rand(5, 4)

        with pytest.raises(ValueError):
            cross(val, val)


if __name__ == "__main__":
    pytest.main([__file__])
