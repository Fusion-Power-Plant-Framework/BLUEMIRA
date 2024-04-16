# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Verification of Coil Supply System classes with EU-DEMO IDM data."""

# %%
from pathlib import Path
from typing import ClassVar

import numpy as np
from matplotlib import (
    colormaps as cmap,
)
from matplotlib import (
    pyplot as plt,
)

from bluemira.power_cycle.coilsupply import CoilSupplyInputs, CoilSupplySystem
from bluemira.power_cycle.tools import (
    match_domains,
    pp,
    read_json,
    symmetrical_subplot_distribution,
)

script_dir = Path(__file__).resolve().parent
data_dir = script_dir / "test_data"


class _PlotOptions:
    title_time = "Time [s]"
    title_voltage = "Voltage [V]"
    title_current = "Current [A]"
    title_active = "Active Power [W]"
    title_reactive = "Reactive Power [VAR]"

    colormap_choice = "cool"
    line_width = 2
    ax_left_color = (0.0, 0.0, 1.0)
    ax_right_color = (1.0, 0.0, 0.0)
    default_subplot_size = 4.5
    color_shade_factor = 0.5
    fancy_legend: ClassVar = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, 1.2),
        "ncol": 6,
        "fancybox": True,
        "shadow": True,
    }
    default_format = "svg"

    @property
    def _line_thin(self):
        return self.line_width

    @property
    def _line_thick(self):
        return 1 + self.line_width

    def _make_colormap(self, n):
        return cmap[self.colormap_choice].resampled(n)

    def _side_color(self, side):
        if side == "left":
            return self.ax_left_color
        if side == "right":
            return self.ax_right_color
        return None

    def _color_yaxis(self, ax, side):
        color = self._side_color(side)
        ax.yaxis.label.set_color(color)
        ax.spines[side].set_color(color)
        ax.tick_params(axis="y", labelcolor=color)

    def _constrained_fig_size(
        self,
        n_rows,
        n_cols,
        row_size=None,
        col_size=None,
    ):
        row_size = self.default_subplot_size if row_size is None else row_size
        col_size = self.default_subplot_size if col_size is None else col_size
        return (col_size * n_cols, row_size * n_rows)

    def _darken_color(self, color, shade=None):
        shade = self.color_shade_factor if shade is None else shade
        return tuple(c * shade for c in color)

    def _save_fig(self, fig, fname, fig_format=None):
        if fig_format is not None:
            fig.savefig(
                fname=script_dir / f"{fname}.{fig_format}",
                format=fig_format,
                transparent=True,
            )
        fig.savefig(
            fname=script_dir / f"{fname}.{self.default_format}",
            format=self.default_format,
            transparent=True,
        )


options = _PlotOptions()

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
config_path = data_dir / "config_coilsupply.json"
coilsupply_data = read_json(config_path)

coilsupply_config = coilsupply_data["coilsupply_config"]
corrector_library = coilsupply_data["corrector_library"]
converter_library = coilsupply_data["converter_library"]

# %% [markdown]
# # Import setup data
#
# Design data relevant for EU-DEMO Coil Supply System design, taken from
# IDM, can be found in the `data_design.json` file.

# For this simple example, we import the resistances that characterize
# the SNU for each coil circuit. This data is added to the appropriate
# field in the `corrector_library` variable.
#

# %%
design_path = data_dir / "coilsupply_data_design.json"
design_data = read_json(design_path)

snu_resistances = {}
coil_times = {}
for coil in design_data:
    snu_resistances[coil] = design_data[coil]["SNU_resistance"]
coil_names = list(design_data.keys())

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


def display_inputs(coilsupply, summary):
    """Print Coil Supply System inputs."""
    pp(coilsupply.inputs, summary)


def display_subsystems(coilsupply, summary):
    """Print summary of Coil Supply System subsystems."""
    correctors_summary = {
        c.name: {
            "class": type(c),
            "resistance": c.resistance_set,
        }
        for c in coilsupply.correctors
    }
    converters_summary = {
        c.name: {
            "class": type(c),
            "v_bridge_arg": c.max_bridge_voltage,
            "power_loss_arg": c.power_loss_percentages,
        }
        for c in [coilsupply.converter]
    }
    pp(correctors_summary, summary)
    pp(converters_summary, summary)


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
breakdown_path = data_dir / "coilsupply_data_breakdown.json"
breakdown_data = read_json(breakdown_path)

breakdown_per_coil = {}
breakdown_reorder_keys = {
    "SNU_voltages": "SNU_voltage",
    "THY_voltages": "THY_voltage",
    "coil_voltages": "coil_voltage",
    "coil_currents": "coil_current",
    "coil_times": "coil_time",
}
duration_breakdown = []
for new_key, old_key in breakdown_reorder_keys.items():
    breakdown_per_coil[new_key] = {}
    for coil in breakdown_data:
        old_value = breakdown_data[coil][old_key]
        if new_key == "coil_times":
            duration_breakdown.append(max(old_value))
        breakdown_per_coil[new_key][coil] = old_value
duration_breakdown = max(duration_breakdown)

breakdown_wallplug = coilsupply.compute_wallplug_loads(
    breakdown_per_coil["coil_voltages"],
    breakdown_per_coil["coil_currents"],
)


def plot_breakdown_verification():
    """Plot Coil Supply System verification for breakdown data."""
    n_plots = len(breakdown_wallplug)
    n_rows, n_cols = symmetrical_subplot_distribution(
        n_plots,
        direction="col",
    )

    v_labels_and_styles = {
        "coil_voltages": "-",
        "SNU_voltages": "-.",
        "THY_voltages": ":",
    }
    v_colors = options._make_colormap(n_plots)

    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        layout="tight",
        figsize=options._constrained_fig_size(n_rows, n_cols, 3.5, 10.5),
    )
    plot_index = 0
    for coil in coil_names:
        wallplug_info = getattr(breakdown_wallplug, coil)

        ax_left = axs.flat[plot_index]
        for label, style in v_labels_and_styles.items():
            ax_left.plot(
                breakdown_per_coil["coil_times"][coil],
                breakdown_per_coil[label][coil],
                f"{style}",
                linewidth=options._line_thick,
                color="k",
                label=f"_{label}_verification",
            )
            ax_left.plot(
                breakdown_per_coil["coil_times"][coil],
                wallplug_info[label],
                f"{style}",
                linewidth=options._line_thin,
                color=v_colors(plot_index),
                label=label,
            )
        ax_left.set_ylabel(options.title_voltage)
        options._color_yaxis(ax_left, "left")
        ax_left.set_xlabel(options.title_time)
        ax_left.grid()

        ax_right = ax_left.twinx()
        ax_right.set_ylabel(options.title_current)
        options._color_yaxis(ax_right, "right")
        ax_right.plot(
            breakdown_per_coil["coil_times"][coil],
            breakdown_per_coil["coil_currents"][coil],
            "-",
            linewidth=options._line_thick,
            color="k",
            label="coil_currents",
        )
        ax_right.plot(
            breakdown_per_coil["coil_times"][coil],
            wallplug_info["coil_currents"],
            "-",
            linewidth=options._line_thin,
            color=options.ax_right_color,
            label="coil_currents",
        )

        ax_left.legend(loc="upper right")
        ax_left.title.set_text(coil)

        plot_index += 1

    fig.suptitle(
        "Coil Supply System Model, Breakdown Verification:\n"
        "original (black) X model (color)",
    )

    options._save_fig(fig, "figure_breakdown_BLUEMIRA", "png")
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
# pulse_path = data_dir / "coilsupply_data_pulse_partial.json"
pulse_path = data_dir / "coilsupply_data_pulse_full.json"
pulse_data = read_json(pulse_path)

pulse_totals = pulse_data["power"]

pulse_per_coil = {}
pulse_reorder_keys = {
    "coil_times": "coil_time",
    "coil_voltages": "coil_voltage",
    "coil_currents": "coil_current",
    "coil_active": "coil_active",
    "coil_reactive": "coil_reactive",
    "SNU_switches": "coil_time",
}

t_start_breakdown = 500
t_end_breakdown = t_start_breakdown + duration_breakdown

for new_key, old_key in pulse_reorder_keys.items():
    pulse_per_coil[new_key] = {}
    for coil in pulse_data["coils"]:
        old_value = pulse_data["coils"][coil].get(old_key, None)
        if new_key == "SNU_switches":
            t_after_start = [t >= t_start_breakdown for t in old_value]
            t_before_end = [t <= t_end_breakdown for t in old_value]
            new_value = [a and b for a, b in zip(t_after_start, t_before_end)]
        else:
            new_value = old_value

        if old_value is not None:
            pulse_per_coil[new_key][coil] = new_value

pulse_wallplug = coilsupply.compute_wallplug_loads(
    pulse_per_coil["coil_voltages"],
    pulse_per_coil["coil_currents"],
    {"SNU": pulse_per_coil["SNU_switches"]},
)

coil_times = pulse_per_coil.pop("coil_times")
snu_switches = pulse_per_coil.pop("SNU_switches")

coil_subplots_settings = {}
for key in pulse_per_coil:
    coil_subplots_settings[key] = {}
    if key in {"coil_voltages", "THY_voltages"}:
        coil_subplots_settings[key]["fig_ind"] = 1
        coil_subplots_settings[key]["side"] = "left"
        coil_subplots_settings[key]["y_title"] = options.title_voltage
        coil_subplots_settings[key]["plot_color"] = "map"
        coil_subplots_settings[key]["variable"] = key
    elif key == "coil_currents":
        coil_subplots_settings[key]["fig_ind"] = 1
        coil_subplots_settings[key]["side"] = "right"
        coil_subplots_settings[key]["y_title"] = options.title_current
        coil_subplots_settings[key]["plot_color"] = "side"
        coil_subplots_settings[key]["variable"] = key
    elif key == "coil_active":
        coil_subplots_settings[key]["fig_ind"] = 2
        coil_subplots_settings[key]["side"] = "left"
        coil_subplots_settings[key]["y_title"] = options.title_active
        coil_subplots_settings[key]["plot_color"] = "map"
        coil_subplots_settings[key]["variable"] = "power_active"
    elif key == "coil_reactive":
        coil_subplots_settings[key]["fig_ind"] = 2
        coil_subplots_settings[key]["side"] = "right"
        coil_subplots_settings[key]["y_title"] = options.title_reactive
        coil_subplots_settings[key]["plot_color"] = "side"
        coil_subplots_settings[key]["variable"] = "power_reactive"
    else:
        raise ValueError(f"Unknown subplot settings for: {key}")

total_subplots_settings = {}
for key in pulse_totals:
    total_subplots_settings[key] = {}
    if key == "total_active":
        total_subplots_settings[key]["side"] = "left"
        total_subplots_settings[key]["y_title"] = options.title_active
        total_subplots_settings[key]["plot_color"] = "side"
        total_subplots_settings[key]["variable"] = "power_active"
    elif key == "total_reactive":
        total_subplots_settings[key]["side"] = "right"
        total_subplots_settings[key]["y_title"] = options.title_reactive
        total_subplots_settings[key]["plot_color"] = "side"
        total_subplots_settings[key]["variable"] = "power_reactive"
    else:
        raise ValueError(f"Unknown subplot settings for: {key}")
    total_subplots_settings[key]["sum_time"] = []
    total_subplots_settings[key]["sum_power"] = []


def plot_pulse_verification():
    """Plot Coil Supply System verification for pulse data."""
    n_coils = len(pulse_wallplug)
    coil_colors = options._make_colormap(n_coils)
    n_plots = n_coils + 1  # last subplot for total powers
    n_rows, n_cols = symmetrical_subplot_distribution(
        n_plots,
        direction="col",
    )

    all_figs = {}
    all_axes = {}
    for key in pulse_per_coil:
        fig_ind = coil_subplots_settings[key]["fig_ind"]
        side = coil_subplots_settings[key]["side"]
        y_title = coil_subplots_settings[key]["y_title"]
        plot_color = coil_subplots_settings[key]["plot_color"]
        variable = coil_subplots_settings[key]["variable"]

        if fig_ind not in all_figs:
            fig, axs = plt.subplots(
                nrows=n_rows,
                ncols=n_cols,
                layout="tight",
                sharex=True,
                figsize=options._constrained_fig_size(
                    n_rows,
                    n_cols,
                    3.5,
                    10.5,
                ),
            )
            all_figs[fig_ind] = fig
            all_axes[fig_ind] = {}
            all_axes[fig_ind]["left"] = axs.flatten()
            all_axes[fig_ind]["right"] = []
            for ax in all_axes[fig_ind]["left"]:
                all_axes[fig_ind]["right"].append(ax.twinx())
        fig = all_figs[fig_ind]
        axs = all_axes[fig_ind][side]

        plot_index = 0
        for coil in coil_names:
            ax = axs[plot_index]
            ax.grid(True, axis="x")
            ax.set_xlabel(options.title_time)
            if y_title is not None:
                ax.set_ylabel(y_title)
            ax.title.set_text(coil)
            options._color_yaxis(ax, side)

            ax_color = options._side_color(side)
            color_verification = (
                ax_color if plot_color == "side" else coil_colors(plot_index)
            )
            color_computation = options._darken_color(color_verification)

            ax.plot(
                coil_times[coil],
                pulse_per_coil[key][coil],
                "-",
                linewidth=options._line_thick,
                color=color_verification,
                label=f"_{key}_verification",
            )
            wallplug_info = getattr(pulse_wallplug, coil)
            ax.plot(
                coil_times[coil],
                wallplug_info[variable],
                "--",
                linewidth=options._line_thin,
                color=color_computation,
                label=f"_{key}_computation",
            )
            if side == "right":
                # correct background color only for SNU switching on/off once
                switch = np.array(snu_switches[coil])
                first_true = np.argmax(switch)
                last_true = switch.size - np.argmax(switch[::-1]) - 1
                ax.axvspan(
                    coil_times[coil][first_true],
                    coil_times[coil][last_true],
                    facecolor=str(options.color_shade_factor),
                    alpha=0.2,
                    zorder=-100,
                )
            ax.grid(True, axis="y", linestyle=":", color=ax_color)

            plot_index += 1

    for key in pulse_totals:
        side = total_subplots_settings[key]["side"]
        y_title = total_subplots_settings[key]["y_title"]
        plot_color = total_subplots_settings[key]["plot_color"]
        variable = total_subplots_settings[key]["variable"]
        sum_time = total_subplots_settings[key]["sum_time"]
        sum_power = total_subplots_settings[key]["sum_power"]

        for coil in coil_names:
            wallplug_info = getattr(pulse_wallplug, coil)
            sum_time.append(coil_times[coil])
            sum_power.append(wallplug_info[variable])
        sum_time, sum_power = match_domains(sum_time, sum_power)
        sum_power = np.add.reduce(sum_power)
        total_subplots_settings[key]["sum_time"] = sum_time
        total_subplots_settings[key]["sum_power"] = sum_power

        ax_color = options._side_color(side)
        color_verification = (
            ax_color if plot_color == "side" else coil_colors(plot_index)
        )
        color_computation = options._darken_color(color_verification)

        for fig_ind in all_figs:
            axs = all_axes[fig_ind][side]
            ax = axs[-1]
            ax.grid(True, axis="x")
            ax.set_xlabel(options.title_time)
            ax.set_ylabel(y_title)
            ax.title.set_text("Totals")
            options._color_yaxis(ax, side)

            ax.plot(
                pulse_totals[key]["time"],
                pulse_totals[key]["power"],
                "-",
                linewidth=options._line_thick,
                color=color_verification,
                label=f"_{key}_verification",
            )
            ax.plot(
                sum_time,
                sum_power,
                "--",
                linewidth=options._line_thin,
                color=color_computation,
                label=f"_{key}_computation",
            )
            ax.grid(True, axis="y", linestyle=":", color=ax_color)

    for fig_ind, fig in all_figs.items():
        fig.suptitle("MATLAB Original (continuous) X BLUEMIRA Model (dashed)")
        data = "VI" if fig_ind == 1 else "PQ"
        options._save_fig(fig, f"figure_pulse_BLUEMIRA_{data}", "png")
    plt.show()
    return all_figs


# %%
if __name__ == "__main__":
    display_inputs(coilsupply, summary=False)
    display_subsystems(coilsupply, summary=True)
    fig_breakdown = plot_breakdown_verification()
    figs_pulse = plot_pulse_verification()
