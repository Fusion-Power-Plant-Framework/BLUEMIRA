# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Set a Coil Supply System up with default EU-DEMO IDM data."""

# %%
from pathlib import Path

from bluemira.power_cycle.coilsupply import CoilSupplyInputs, CoilSupplySystem
from bluemira.power_cycle.tools import pp, read_json

script_dir = Path(__file__).resolve().parent

# %% [markdown]
# # Import default Coil Supply System data
#
# The Power Supply for DEMO coils is composed of a main converter device,
# potentially based on the technology of Thyristor Bridges (THY), and
# some auxiliary components:
#   - Protective Make Switch (PMS): isolates the coils from the supply
#                                   in case of component fault
#   - Fast Discharging Unit (FDU): quench protector for the superconducting
#                                  coils
#   - Switiching Network Unit (SNU): devices connected in series to reduce
#                                    voltage consumed during breakdown
#
# We can create the basic arguments to set a Coil Supply System up
# based on the EU-DEMO, using the example inputs stored in the file
# `config_coilsupply.json`.
#

# %%
config_path = script_dir / "config_coilsupply.json"
coilsupply_data = read_json(config_path)

coilsupply_config = coilsupply_data["coilsupply_config"]
corrector_library = coilsupply_data["corrector_library"]
converter_library = coilsupply_data["converter_library"]

# %% [markdown]
# # Import setup data
#
# Design data relevant for EU-DEMO Coil Supply System, taken from IDM,
# can be found in the `data_setup.json` file.

# For this simple example, we import the resistances that characterize
# the SNU for each coil circuit. This data is added to the appropriate
# field in the `corrector_library` variable.
#

# %%
setup_path = script_dir / "data_setup.json"
setup_data = read_json(setup_path)

snu_resistances = {}
coil_times = {}
for coil in setup_data:
    snu_resistances[coil] = setup_data[coil]["resistance_SNU"]
coil_names = list(setup_data.keys())

coilsupply_config["coil_names"] = coil_names
corrector_library["SNU"]["resistance_set"] = snu_resistances

# %% [markdown]
# # Initialize CoilSupplySystem
#
# A `CoilSupplySystem` instance can then be initialized.
#
# This is done by first initializing a `CoilSupplyInputs` instance.
# Notice that it pre-processes the argument variables into the required
# formats. For example, scalar inputs for correctors are transformed
# into dictionaries with the appropriate coil names, based on the
# `coil_names` list in the `coilsupply_config` variable.
#
# Then, it is used to initialize the desired `CoilSupplySystem` instance.
#

# %%
coilsupply_inputs = CoilSupplyInputs(
    config=coilsupply_config,
    corrector_library=corrector_library,
    converter_library=converter_library,
)

coilsupply = CoilSupplySystem(coilsupply_inputs)


# %%
if __name__ == "__main__":
    pp(coilsupply.inputs)
    pp(
        {
            c.name: {
                "class": type(c),
                "resistance": c.resistance_set,
            }
            for c in coilsupply.correctors
        }
    )
    pp(
        {
            c.name: {
                "class": type(c),
                "v_bridge_arg": c.max_bridge_voltage,
                "power_loss_arg": c.power_loss_percentages,
            }
            for c in [coilsupply.converter]
        }
    )
