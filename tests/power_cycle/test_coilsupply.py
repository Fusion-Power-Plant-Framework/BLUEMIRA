# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later


from bluemira.power_cycle.coilsupply import (
    CoilVariable,
)


def test_CoilVariable_from_str():
    assert CoilVariable.from_str("voltage") == CoilVariable.ACTIVE
    assert CoilVariable.from_str("current") == CoilVariable.REACTIVE
