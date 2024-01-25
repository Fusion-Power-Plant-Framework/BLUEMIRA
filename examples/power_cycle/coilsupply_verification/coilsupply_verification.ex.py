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


class _PlotOptions:
    colormap_choice = "cool"
    line_width = 2
    ax_left_color = "b"
    ax_right_color = "r"
    subplots_size = 3.5
    color_shade_factor = 0.5

    @property
    def _line_thin(self):
        return self.line_width

    @property
    def _line_thick(self):
        return 1 + self.line_width

    def _make_colormap(self, n):
        return cmap[self.colormap_choice].resampled(n)

    def _color_yaxis(self, ax, side):
        if side == "left":
            color = self.ax_left_color
        elif side == "right":
            color = self.ax_right_color
        ax.yaxis.label.set_color(color)
        ax.spines[side].set_color(color)
        ax.tick_params(axis="y", labelcolor=color)

    def _constrained_fig_size(self, n_rows, n_cols):
        return (
            self.subplots_size * n_cols,
            self.subplots_size * n_rows,
        )

    def _darken_color(self, color):
        return tuple(c * self.color_shade_factor for c in color)


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
# computation arguments (coil currents, in red, and voltages, in shades
# of blue) are also plotted against the original data (in black).
#

# %%
breakdown_path = script_dir / "data_breakdown.json"
breakdown_data = read_json(breakdown_path)

breakdown = {}
breakdown_reorder_keys = {
    "SNU_voltages": "voltage_SNU",
    "THY_voltages": "voltage_THY",
    "coil_voltages": "voltage_coil",
    "coil_currents": "current_coil",
    "coil_times": "time_coil",
}
for new_key, old_key in breakdown_reorder_keys.items():
    breakdown[new_key] = {}
    for coil in breakdown_data:
        breakdown[new_key][coil] = breakdown_data[coil][old_key]

breakdown["wallplug_parameter"] = coilsupply.compute_wallplug_loads(
    breakdown["coil_voltages"],
    breakdown["coil_currents"],
)


def plot_breakdown_verification(breakdown):
    """Plot Coil Supply System verification for breakdown data."""
    n_plots = len(breakdown["wallplug_parameter"])
    n_rows, n_cols = symmetrical_subplot_distribution(
        n_plots,
        direction="col",
    )

    options = _PlotOptions()
    v_labels_and_styles = {
        "coil_voltages": "-",
        "SNU_voltages": "-.",
        "THY_voltages": ":",
    }
    v_colors = options._make_colormap(n_plots)

    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        layout="constrained",
        figsize=options._constrained_fig_size(n_rows, n_cols),
    )
    plot_index = 0
    for coil in reversed(coil_names):
        wallplug_info = getattr(breakdown["wallplug_parameter"], coil)

        ax_left = axs.flat[plot_index]
        for label, style in v_labels_and_styles.items():
            ax_left.plot(
                breakdown["coil_times"][coil],
                breakdown[label][coil],
                f"{style}",
                linewidth=options._line_thick,
                color="k",
                label=f"_{label}_verification",
            )
            ax_left.plot(
                breakdown["coil_times"][coil],
                wallplug_info[label],
                f"{style}",
                linewidth=options._line_thin,
                color=v_colors(plot_index),
                label=label,
            )
        ax_left.set_ylabel("Voltage [V]")
        options._color_yaxis(ax_left, "left")
        ax_left.set_xlabel("Time [s]")
        ax_left.grid()

        ax_right = ax_left.twinx()
        ax_right.set_ylabel("Current [A]")
        options._color_yaxis(ax_right, "right")
        ax_right.plot(
            breakdown["coil_times"][coil],
            breakdown["coil_currents"][coil],
            "-",
            linewidth=options._line_thick,
            color="k",
            label="coil_currents",
        )
        ax_right.plot(
            breakdown["coil_times"][coil],
            wallplug_info["coil_currents"],
            "-",
            linewidth=options._line_thin,
            color=options.ax_right_color,
            label="coil_currents",
        )

        ax_left.legend(loc="upper right")
        ax_left.title.set_text(coil)

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
pulse_reorder_keys = {
    "coil_voltages": "voltage_coil",
    "coil_currents": "current_coil",
    "coil_times": "time_coil",
}
for new_key, old_key in pulse_reorder_keys.items():
    pulse[new_key] = {}
    for coil in pulse_data["coils"]:
        pulse[new_key][coil] = pulse_data["coils"][coil][old_key]

pulse["wallplug_parameter"] = coilsupply.compute_wallplug_loads(
    pulse["coil_voltages"],
    pulse["coil_currents"],
)


def plot_pulse_verification(pulse, power):
    """Plot Coil Supply System verification for pulse data."""
    n_coils = len(pulse["wallplug_parameter"])

    options = _PlotOptions()
    coil_colors = options._make_colormap(n_coils)

    n_rows = 5
    n_cols = 1
    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        sharex=True,
    )
    subplots_axes = {
        "voltage": axs.flat[0],
        "current": axs.flat[1],
        "active": axs.flat[2],
        "reactive": axs.flat[3],
        "total": axs.flat[4],
    }

    idm_data = True
    coil_subplots_settings = {
        "voltage": ("coil_voltages", "Voltage [V]", idm_data),
        "current": ("coil_currents", "Current [A]", idm_data),
        "active": ("power_active", "Active Power [W]", not idm_data),
        "reactive": ("power_reactive", "Reactive Power [VAR]", not idm_data),
    }
    for key, settings in coil_subplots_settings.items():
        plot_index = 0
        ax = subplots_axes[key]
        variable = settings[0]
        ylabel = settings[1]
        verification_available = settings[2]

        for coil in reversed(coil_names):
            wallplug_info = getattr(pulse["wallplug_parameter"], coil)
            plot_index = plot_index + 1
            if verification_available:
                color_verification = coil_colors(plot_index)
                style_verification = "-"
                ax.plot(
                    pulse["coil_times"][coil],
                    pulse[variable][coil],
                    style_verification,
                    linewidth=options._line_thick,
                    color=color_verification,
                    label=f"_{coil}_verification",
                )
                color_computation = options._darken_color(color_verification)
                style_computation = "--"
            else:
                color_computation = coil_colors(plot_index)
                style_computation = "-"
            ax.plot(
                pulse["coil_times"][coil],
                wallplug_info[variable],
                style_computation,
                linewidth=options._line_thin,
                color=color_computation,
                label=coil,
            )
        ax.set_ylabel(ylabel)
        ax.grid()

    """
    total_subplot_settings = {
        "active": ("active_power"),
        "reactive": ("reactive_power"),
    }
    for coil in reversed(coil_names):
            wallplug_info = getattr(pulse["wallplug_parameter"], coil)
    """

    ax_left = subplots_axes["total"]
    ax_left.set_xlabel("Time [s]")

    """
    ax_right = ax_left.twinx()
    """

    plt.show()
    return fig, power


# %%
if __name__ == "__main__":
    display_inputs(coilsupply)
    display_subsystems(coilsupply)
    # fig_breakdown = plot_breakdown_verification(breakdown)
    fig_pulse = plot_pulse_verification(pulse, power)
    # print(pulse["wallplug_parameter"])
