# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later


from bluemira.geometry.tools import make_circle


class TestCircularBMWire:
    circle = make_circle(axis=[0, 1, 0])

    # @pytest.mark.parametrize()
    # def test_too_few_points():
    #     with pytest.raises(GeometryError):
    #         pass
