# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Functions for creating the openmc tallies."""

from itertools import chain

import openmc

from bluemira.codes.openmc.make_cell import BlanketCellArray, DivertorCellArray


def filter_cells(
    material_list,
    blanket_cell_array: BlanketCellArray,
    divertor_cell_array: DivertorCellArray,
):
    """
    Create scores and the filter for the scores. Give them names.

    Returns
    -------
    TBR
        Achieved by (n,Xt) reaction, which counts the number of tritium-producing
        nuclear reactions per neutron emitted at the source.

        We used the (n,Xt) score because the Lithium produces a maximum of 1 Tritium per
        reaction, so there won't be any concerns about uncer-counting the TBR.

    Powers
        Measures the nuclear heating in various locations and materials, and interpret
        this as power. "damage-energy" is given by eV per source neutron.
        Multiply by neutron source rate, and then divide by (number of atoms and
        threshold displacement energy) to get the DPA.

    Fluence
        Measures # of neutrons streaming through.
        "flux" is given in # per source particle, so multiply by # of source neutrons to
        get the total fluence over the simulation.
        Divide by area to get fluence in unit: cm^-2.

    """
    blanket_cells = [*chain.from_iterable(blanket_cell_array)]
    div_cells = [*chain.from_iterable(divertor_cell_array)]
    cells = blanket_cells + div_cells
    fw_surf_cells = [
        *(stack[0] for stack in blanket_cell_array),
        *(stack[1] for stack in blanket_cell_array),
    ]
    vv_cells = [
        *(stack[-1] for stack in blanket_cell_array),
        *(stack[-1] for stack in divertor_cell_array),
    ]
    bz_cells = [stack[2] for stack in blanket_cell_array]

    # Cell filters
    # blanket_cell_filter = openmc.CellFilter(blanket_cells)
    div_cell_filter = openmc.CellFilter(div_cells)
    cell_filter = openmc.CellFilter(cells)
    fw_surf_filter = openmc.CellFilter(fw_surf_cells)
    vv_filter = openmc.CellFilter(vv_cells)
    bz_filter = openmc.CellFilter(bz_cells)

    # material filters
    mat_filter = openmc.MaterialFilter(material_list[:-1])
    eurofer_filter = openmc.MaterialFilter([material_list[-1]])
    neutron_filter = openmc.ParticleFilter(["neutron"])
    photon_filter = openmc.ParticleFilter(["photon"])

    # name, scores, filters
    return (
        ("TBR", "(n,Xt)", []),  # theoretical maximum TBR only, obviously.
        # Powers
        ("Total power", "heating", [mat_filter]),
        ("divertor power", "heating", [div_cell_filter]),
        ("vacuum vessel power", "heating", [vv_filter]),
        ("breeding blanket power", "heating", [bz_filter]),
        # Fluence
        ("neutron flux in every cell", "flux", [cell_filter, neutron_filter]),
        ("photon heating", "heating", [fw_surf_filter, photon_filter]),
        # ("neutron flux in 2d mesh", "flux", [cyl_mesh_filter, neutron_filter]),
        # TF winding pack does not exits yet, so this will have to wait
        # DPA
        ("eurofer damage", "damage-energy", [cell_filter, eurofer_filter]),
        # used to get the EUROFER OBMP
        ("divertor damage", "damage-energy", [div_cell_filter, mat_filter]),
        ("vacuum vessel damage", "damage-energy", [vv_filter]),
    )
