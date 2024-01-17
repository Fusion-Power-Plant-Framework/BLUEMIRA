# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Power cycle example"""
# %%

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bluemira.base.reactor_config import ReactorConfig
from bluemira.power_cycle.net import (
    Efficiency,
    PowerCycleLibraryConfig,
    PowerCycleLoadConfig,
    interpolate_extra,
)

# %% [markdown]
# # Power Cycle example
#
# Firstly we read in the build config and extract the config for the PowerCycle
# %%

reactor_config = ReactorConfig(Path(__file__).parent / "scenario_config.json", None)
config = PowerCycleLibraryConfig.from_dict(reactor_config.config_for("Power Cycle"))

# %% [markdown]
#  We then import and subphase durations.
#  In principle these could come from other parts of the reactor design.
# %%


@dataclass
class PowerCycleDurationParameters:
    """Dummy power cycle duration parameters [s]"""

    cs_recharge_time: float = 300
    pumpdown_time: float = 600
    ramp_up_time: float = 157
    ramp_down_time: float = 157


config.import_subphase_duration(PowerCycleDurationParameters())

# %% [markdown]
# We can then dynamically add a new load to a specific subphase of the config.
# %%

config.add_load_config(
    PowerCycleLoadConfig(
        "cs_power",
        active_data=[1, 2],
        reactive_data=[10, 20],
        efficiencies=[Efficiency(0.1)],
        description="something made up",
    ),
    ["cru", "bri"],
    [Efficiency(0.2, reactive=True)],
)

# %% [markdown]
# Once the config is created we can now pull out the data for a specific phase.
#
# Below we have interpolated the timeseries and pulled out the active and reactive loads
# %%

phase = config.make_phase("dwl")

normalised_time = interpolate_extra(phase.loads.build_timeseries(), 3)
timeseries = normalised_time * phase.duration

active_loads = phase.loads.get_load_data_with_efficiencies(
    normalised_time, "active", "MW"
)
active_load_total = phase.loads.load_total(normalised_time, "active", "MW")

reactive_loads = phase.loads.get_load_data_with_efficiencies(
    normalised_time, "reactive", "MW"
)
reactive_load_total = phase.loads.load_total(normalised_time, "reactive", "MW")
