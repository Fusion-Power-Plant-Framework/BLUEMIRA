# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Example with EU-DEMO IDM data for Coil Supply System verification."""

# %%
from pathlib import Path

from matplotlib import (
    colormaps as cmap,
)
from matplotlib import (
    pyplot as plt,
)

from bluemira.power_cycle.coilsupply import CoilSupplyInputs, CoilSupplySystem
from bluemira.power_cycle.tools import (
    read_json,
    symmetrical_subplot_distribution,
)

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
# We can set a Coil Supply System up based on the EU-DEMO, using the
# example inputs stored in the `data_coilsupply.json` file.
#

# %%
coilsupply_path = script_dir / "data_coilsupply.json"
coilsupply_data = read_json(coilsupply_path)

coilsupply_config = coilsupply_data["coilsupply_config"]
corrector_library = coilsupply_data["corrector_library"]
converter_library = coilsupply_data["converter_library"]

# %% [markdown]
# # Import breakdown data
#
# The data for verification of the coil supply model is stored in the
# `data_breakdown.json` file. We import it to use:
#   - SNU resistances for the config inputs
#   - coil voltages and currents as simulation inputs
#   - SNU and THY voltages as output verification
#
# So we start by adding the SNU resistances to the corrector library and
# reorganizing the voltages and currents for later passing as argument.
#

# %%
breakdown_path = script_dir / "data_breakdown.json"
breakdown_data = read_json(breakdown_path)

snu_resistances = {}
coil_voltages = {}
coil_currents = {}
snu_voltages = {}
thy_voltages = {}
coil_times = {}
for coil in breakdown_data:
    snu_resistances[coil] = breakdown_data[coil]["resistance_SNU"]
    snu_voltages[coil] = breakdown_data[coil]["voltage_SNU"]
    thy_voltages[coil] = breakdown_data[coil]["voltage_THY"]
    coil_voltages[coil] = breakdown_data[coil]["voltage_coil"]
    coil_currents[coil] = breakdown_data[coil]["current_coil"]
    coil_times[coil] = breakdown_data[coil]["time_coil"]
coil_names = list(breakdown_data.keys())

coilsupply_config["coil_names"] = coil_names
corrector_library["SNU"]["resistance_set"] = snu_resistances

coilsupply_inputs = CoilSupplyInputs(
    config=coilsupply_config,
    corrector_library=corrector_library,
    converter_library=converter_library,
)
# pp(coilsupply_inputs)

# %% [markdown]
# # Set-up the Coil Supply System
#
# Now we can build the Coil Supply System instance to compute.
#
# Notice how single values for `equivalent_resistance_set` values become
# tuples in the `inputs` dictionary with length equal to `coilset_size`.
#
#

# %%

coilsupply = CoilSupplySystem(
    coilsupply_config,
    corrector_library,
    converter_library,
)
# pp(coilsupply.inputs)
# pp(coilsupply.correctors)
# pp(coilsupply.converter)

for corrector in coilsupply.correctors:
    print(corrector.resistance_set)

wallplug_parameter = coilsupply.compute_wallplug_loads(
    coil_voltages,
    coil_currents,
)
# pp(wallplug_parameter)


# %% [markdown]
# # Simulate the Coil Supply System
#
#
#

# %%
n_plots = wallplug_parameter.len()
n_rows, n_cols = symmetrical_subplot_distribution(n_plots, direction="col")
colormap_choice = "cool"
ax_left_color = "b"
ax_right_color = "k"


def color_yaxis(ax, side, color):
    """
    Color all characteristics of a y-axis.

    The 'side' parameter chould be 'left' or 'right'.
    """
    ax.yaxis.label.set_color(color)
    ax.spines[side].set_color(color)
    ax.tick_params(axis="y", labelcolor=color)


voltage_labels = ["coil_voltages", "SNU_voltages", "THY_voltages"]
voltage_styles = ["-", "-.", ":"]
voltage_colors = cmap[colormap_choice].resampled(n_plots)

fig, axs = plt.subplots(
    nrows=n_rows,
    ncols=n_cols,
    layout="constrained",
    figsize=(3.5 * n_cols, 3.5 * n_rows),
)
plot_index = 0
for name in reversed(coil_names):
    wallplug_info = getattr(wallplug_parameter, name)

    ax_left = axs.flat[plot_index]
    for label, style in zip(voltage_labels, voltage_styles):
        ax_left.plot(
            coil_times[name],
            wallplug_info[label],
            f"{style}",
            color=voltage_colors(plot_index),
            label=label,
        )
    ax_left.set_ylabel("Voltage [V]")
    color_yaxis(ax_left, "left", ax_left_color)
    ax_left.set_xlabel("Time [s]")
    ax_left.grid()

    ax_right = ax_left.twinx()
    ax_right.set_ylabel("Current [A]")
    color_yaxis(ax_right, "right", ax_right_color)
    ax_right.plot(
        coil_times[name],
        wallplug_info["coil_currents"],
        "-",
        color="k",
        label="coil_currents",
    )

    ax_left.legend(loc="upper right")
    ax_left.title.set_text(name)

    plot_index += 1

# fig.tight_layout()
plt.show()
