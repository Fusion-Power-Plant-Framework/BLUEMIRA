# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Coil Supply System example using the equilibria module (EU-DEMO 2017)."""

# %%

from bluemira.power_cycle.coilsupply import CoilSupplySystem
from bluemira.power_cycle.tools import pp

# %% [markdown]
# # Coil Supply System set-up
#
# The Power Supply for DEMO coils is composed of a main converter device,
# potentially based on the technology of Thyristor Bridges, and some
# auxiliary components:
#   - Protective Make Switch: isolates the coils from the supply in case
#                             of component fault
#   - Fast Discharging Unit: quench protector for the superconducting
#                            coils
#   - Switiching Network Unit: devices connected in series to provide
#                              additional voltage during breakdown
#


# %%
coilsupply_config = {
    "description": "Coil Supply System",
    "correctors_tuple": ("FDU", "SNU", "PMS"),
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

# %% [markdown]
# # Coil Supply System simulation
#
# Import data of coil voltages and currents from `equilibria` module.
#


# %%
def run_equilibria_script():
    """Run the equilibria script."""
