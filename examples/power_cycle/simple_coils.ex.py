# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Coil Supply System example."""

from bluemira.power_cycle.coil_supply import CoilSupplySystem

if True:
    import pprint
    from dataclasses import asdict, is_dataclass

    def pp(obj):
        """PretyPrinter for dataclasses"""
        if is_dataclass(obj):
            return pprint.pp(asdict(obj), sort_dicts=False, indent=4)
        return pprint.pp(obj, indent=4)


coilsupply_config = {
    "description": "Coil Supply System",
    "correctors_tuple": ("PMS", "FDU", "SNU"),
    "converter_technology": "THY",
}

corrector_library = {
    "PMS": {
        "description": "Protective Make Switch",
        "correction_variable": "voltage",
        "correction_factor": 0,
    },
    "FDU": {
        "description": "Fast Discharging Unit",
        "correction_variable": "current",
        "correction_factor": 0,
    },
    "SNU": {
        "description": "Switiching Network Unit",
        "correction_variable": "voltage",
        "correction_factor": -0.4,
    },
}

converter_library = {
    "THY": {
        "class_name": "ThyristorBridges",
        "class_args": {
            "description": "Thyristor Bridges (ITER-like)",
            "max_bridge_voltage": 1.6e3,
            "power_loss_percentages": {
                "bridge losses": 1.5,
                "step-down transformer": 1,
                "output busbars": 0.5,
            },
        },
    },
    "NEW": {
        "class_name": None,
        "class_args": {
            "description": "New technology (?)",
            "...": "...",
        },
    },
}

coilsupply = CoilSupplySystem(
    coilsupply_config,
    corrector_library,
    converter_library,
)
pp(coilsupply.inputs)
pp(coilsupply.correctors_list)
pp(coilsupply.converter)
