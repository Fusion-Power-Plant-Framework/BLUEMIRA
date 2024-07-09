# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import pytest

from bluemira.radiation_transport.neutronics.constants import (
    to_cm,
    to_cm3,
    to_m,
)


@pytest.mark.parametrize("length_in_cm", [1.0])
def test_cm_to_m(length_in_cm):
    assert (length_in_cm / 100) == to_m(length_in_cm)


@pytest.mark.parametrize("length_in_m", [1.0])
def test_m_to_cm(length_in_m):
    assert (length_in_m * 100) == to_cm(length_in_m)


@pytest.mark.parametrize("volume_in_m3", [1.0])
def test_m3_to_cm3(volume_in_m3):
    assert (volume_in_m3 * (100**3)) == to_cm3(volume_in_m3)
