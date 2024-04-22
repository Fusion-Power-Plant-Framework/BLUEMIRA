# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Example with EU-DEMO IDM data for Coil Supply System verification."""

# %%
import matplotlib.pyplot as plt
import numpy as np

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
            "max_bridge_voltage": 1.6e3,  # module
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
# Input data from IDM report 2Q6988.
# Voltages in V. Currents in A, varying linearly.
#

# %%
voltage_for_each_coil = 0
currents_start_of_flat = {
    "CS3U": 18.7e3,
    "CS2U": 5.6e3,
    "CS1": -1.9e3,
    "CS2L": -0.9e3,
    "CS3L": 25.6e3,
    "PF1": 33.6e3,
    "PF2": -18.9e3,
    "PF3": -24.8e3,
    "PF4": -5.3e3,
    "PF5": -24.3e3,
    "PF6": 34.1e3,
}
currents_end_of_flat = {
    "CS3U": -10.1e3,
    "CS2U": -33.5e3,
    "CS1": -34.7e3,
    "CS2L": -32.7e3,
    "CS3L": -21.4e3,
    "PF1": -3.0e3,
    "PF2": -30.5e3,
    "PF3": -23.6e3,
    "PF4": -9.0e3,
    "PF5": -26.8e3,
    "PF6": 19.2e3,
}

n_points = 10
voltages_during_flat = {}
currents_during_flat = {}
wallplug_info_during_flat = {}
for coil in currents_start_of_flat:
    coil_voltages = [voltage_for_each_coil] * n_points

    SOF_current = currents_start_of_flat[coil]
    EOF_current = currents_end_of_flat[coil]
    coil_currents = np.linspace(SOF_current, EOF_current, n_points)
    coil_currents = coil_currents.tolist()

    coil_wallplug_info = coilsupply.compute_wallplug_loads(
        coil_voltages,
        coil_currents,
    )

    voltages_during_flat[coil] = coil_voltages
    currents_during_flat[coil] = coil_currents
    wallplug_info_during_flat[coil] = coil_wallplug_info

    plt.plot(coil_currents, label=coil)
plt.legend()
plt.show()

pp(wallplug_info_during_flat)

key_variable = "power_reactive"
for coil in wallplug_info_during_flat:
    coil_info = wallplug_info_during_flat[coil]
    plot_variable = coil_info[key_variable]
    plt.plot(plot_variable, label=coil)
plt.legend()
plt.show()
