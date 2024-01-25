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
    pp,
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


def display_inputs(coilsupply):
    """Print Coil Supply System inputs."""
    pp(coilsupply.inputs)


def display_subsystems(coilsupply):
    """Print summary of Coil Supply System subsystems."""
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


# %% [markdown]
# # Verification with breakdown data
#
# The IDM data for the behavior of correctors and converters during
# breakdown is stored in the `data_breakdown.json` file. We import it
# to use:
#   - coil voltages and currents as simulation inputs;
#   - SNU and THY voltages as output verification.
#
# Voltages and currents are first re-ordered and later used as arguments
# and data for comparison.
#
# The figure displays how computed voltages for the SNU and THY (in color)
# coincide with the same values in the original data (in black). The
# computation arguments (coil voltages and currents, in color and white,
# respectively) are also plotted against the original data (in black).
#

# %%
breakdown_path = script_dir / "data_breakdown.json"
breakdown_data = read_json(breakdown_path)

breakdown = {}
breakdown_reorder = {
    "SNU_voltages": "voltage_SNU",
    "THY_voltages": "voltage_THY",
    "coil_voltages": "voltage_coil",
    "coil_currents": "current_coil",
    "coil_times": "time_coil",
}
for new_key, old_key in breakdown_reorder.items():
    breakdown[new_key] = {}
    for coil in breakdown_data:
        breakdown[new_key][coil] = breakdown_data[coil][old_key]

breakdown["wallplug_parameter"] = coilsupply.compute_wallplug_loads(
    breakdown["coil_voltages"],
    breakdown["coil_currents"],
)


def _color_yaxis(ax, side, color):
    ax.yaxis.label.set_color(color)
    ax.spines[side].set_color(color)
    ax.tick_params(axis="y", labelcolor=color)


def plot_breakdown_verification(breakdown):
    """Plot Coil Supply System verification for breakdown data."""
    n_plots = len(breakdown["wallplug_parameter"])
    n_rows, n_cols = symmetrical_subplot_distribution(n_plots, direction="col")
    colormap_choice = "cool"
    ax_left_color = "b"
    ax_right_color = "k"
    line_width = 2

    v_labels_and_styles = {
        "coil_voltages": "-",
        "SNU_voltages": "-.",
        "THY_voltages": ":",
    }
    v_colors = cmap[colormap_choice].resampled(n_plots)

    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        layout="constrained",
        figsize=(3.5 * n_cols, 3.5 * n_rows),
    )
    plot_index = 0
    for name in reversed(coil_names):
        wallplug_info = getattr(breakdown["wallplug_parameter"], name)

        ax_left = axs.flat[plot_index]
        for label, style in v_labels_and_styles.items():
            ax_left.plot(
                breakdown["coil_times"][name],
                breakdown[label][name],
                f"{style}",
                linewidth=line_width + 1,
                color="k",
                label=f"_{label}_verification",
            )
            ax_left.plot(
                breakdown["coil_times"][name],
                wallplug_info[label],
                f"{style}",
                linewidth=line_width,
                color=v_colors(plot_index),
                label=label,
            )
        ax_left.set_ylabel("Voltage [V]")
        _color_yaxis(ax_left, "left", ax_left_color)
        ax_left.set_xlabel("Time [s]")
        ax_left.grid()

        ax_right = ax_left.twinx()
        ax_right.set_ylabel("Current [A]")
        _color_yaxis(ax_right, "right", ax_right_color)
        ax_right.plot(
            breakdown["coil_times"][name],
            breakdown["coil_currents"][name],
            "-",
            linewidth=line_width + 1,
            color="k",
            label="coil_currents",
        )
        ax_right.plot(
            breakdown["coil_times"][name],
            wallplug_info["coil_currents"],
            "-",
            linewidth=line_width,
            color="w",
            label="coil_currents",
        )

        ax_left.legend(loc="upper right")
        ax_left.title.set_text(name)

        plot_index += 1

    plt.show()
    return fig


# %% [markdown]
# # Verification with pulse data
#
# A downsampled version of the IDM data for the total active and reactive
# powers, consumed during a full pulse by the Coil Supply System, is stored
# in the `data_pulse.json` file. We import it to use:
#   - coil voltages and currents as simulation inputs;
#   - active and reactive powers consumed by the system as output verification.
#
# Voltages, currents are first re-ordered and later used as arguments.
# Returned power values are summed and compared against the original results.
#
# The figure displays...
#

# %%
pulse_path = script_dir / "data_pulse.json"
pulse_data = read_json(pulse_path)

power = pulse_data["power"]

pulse = {}
pulse_reorder = [
    "voltage_coil",
    "current_coil",
    "time_coil",
]
for key in pulse_reorder:
    pulse[key] = {}
    for coil in pulse_data["coils"]:
        pulse[key][coil] = pulse_data["coils"][coil][key]

pulse["wallplug_parameter"] = coilsupply.compute_wallplug_loads(
    pulse["voltage_coil"],
    pulse["current_coil"],
)


def plot_pulse_verification(pulse, power):
    """Plot Coil Supply System verification for pulse data."""
    return pulse, power


# %%
if __name__ == "__main__":
    display_inputs(coilsupply)
    display_subsystems(coilsupply)
    # fig_breakdown = plot_breakdown_verification(breakdown)
    pp(pulse["wallplug_parameter"])
