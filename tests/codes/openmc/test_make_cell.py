# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later


def test_convexity_cells(): ...


def test_convexity_cell_stacks(): ...


def test_convexity_cell_arrays(): ...


def test_in_cell():
    """Checking the functionality of each cell"""


def test_cell_walls_multiply_accessed():
    """Each cell wall must've been accessed by the cells on either side, thus must've
    been used at least twice.
    """
