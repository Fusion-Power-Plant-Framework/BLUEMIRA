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
# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
A typical fuel cycle result for an EU-DEMO reference point
"""

# %%
from bluemira.display.auto_config import plot_defaults
from bluemira.fuel_cycle.analysis import FuelCycleAnalysis
from bluemira.fuel_cycle.cycle import EUDEMOFuelCycleModel
from bluemira.fuel_cycle.lifecycle import LifeCycle
from bluemira.fuel_cycle.timeline_tools import (
    GompertzLearningStrategy,
    LogNormalAvailabilityStrategy,
)
from bluemira.fuel_cycle.tools import convert_flux_to_flow
from bluemira.plasma_physics.reactions import n_DD_reactions, n_DT_reactions
from bluemira.utilities.tools import set_random_seed

plot_defaults()

# %% [markdown]
# # Fuel cycle example
#
# First let's set up a configuration with some values (a la EU-DEMO 2015)

# %%
p_fus_DT = 2037
p_fus_DD = 3

baseline_not_critical = True
HCPB_not_WCLL = True
high_not_low = False

if baseline_not_critical:
    A_global = 0.3
    TBR = 1.05
    f_dir = 0.8
else:
    A_global = 0.2
    TBR = 1.02
    f_dir = 0.6

lifecycle_config = {
    "A_global": A_global,
    "I_p": 19.2,
    "bmd": 150,
    "dmd": 90,
    "t_pulse": 7200,
    "t_cs_recharge": 600,
    "t_pumpdown": 599,
    "s_ramp_up": 0.1,
    "s_ramp_down": 0.1,
    "n_DT_reactions": n_DT_reactions(p_fus_DT),
    "n_DD_reactions": n_DD_reactions(p_fus_DD),
    "blk_1_dpa": 20,
    "blk_2_dpa": 50,
    "div_dpa": 5,
    "vv_dpa": 3.25,
    "tf_fluence": 3.2e21,
}

# %% [markdown]
#
# Now we set a LifeCycle object in order generate some pseudo-randomised timelines.
#
# We're going to define a LearningStrategy to determine how the operational availability
# of our reactor improves over time.
#
# We're going to define an AvailabilityStrategy to determine how the durations in
# between pulses are distributed.

# %%
lifecycle_inputs = {}

# We need to define some strategies to define the pseudo-random timelines

# Let's choose a LearningStrategy such that the operational availability grows over time
learning_strategy = GompertzLearningStrategy(
    learn_rate=1.0, min_op_availability=0.1, max_op_availability=0.5
)
# Let's choose an OperationalAvailabilityStrategy to determine how to distribute outages
availability_strategy = LogNormalAvailabilityStrategy(sigma=2.0)

lifecycle = LifeCycle(
    lifecycle_config, learning_strategy, availability_strategy, lifecycle_inputs
)

# %% [markdown]
#
# Now we use the LifeCycle to generate pseudo-randomised timelines. Let's set a
# random seed number first to get repeatable results

# %%
set_random_seed(2358203947)

# Let's do 50 runs Monte Carlo
# NOTE: Make sure you have enough memory..!
n = 50
time_dicts = [lifecycle.make_timeline().to_dict() for _ in range(n)]

# %% [markdown]
#
# Now let's set up a TFVSystem
#
# First we need to get some input parameters
#
# Some conversions from inventories to residence times
# (as discussed with Jonas Schwenzer, KIT)

# %%
m_exhaust_systems = 0.11
exhaust_influx = 3.28229059e-05
t_exh = m_exhaust_systems / exhaust_influx

vessel_outflux = 0.00032812

fw_retention_flux = 1e20 if high_not_low else 1e19

# Numbers taken from KDI-2 report (R. Arrendondo, F. Franza, C. Moro)
if HCPB_not_WCLL:
    if high_not_low:
        eta_ivc = 0.33013436
        max_ivc_inventory = 1.11571993
        max_bb_inventory = 0.055  # F. Franza
    else:
        eta_ivc = 0.0972729
        max_ivc_inventory = 0.37786173
        max_bb_inventory = 0.026
elif high_not_low:
    eta_ivc = 0.62037084
    max_ivc_inventory = 1.65440323
    max_bb_inventory = 0.18519
else:
    eta_ivc = 0.13090036
    max_ivc_inventory = 0.64900007
    max_bb_inventory = 0.06314

fw_flow = convert_flux_to_flow(fw_retention_flux, 1400)


# Normalise sqrt sink factors, because T flows are off...
eta_norm = fw_flow / vessel_outflux
eta_ivc /= eta_norm

m_dir_pump = 0.024
t_pump = m_dir_pump / vessel_outflux

# It was agreed that this 1.8 kg is needed steady-state, and is used as I_TFV_min
m_cryodistillation = 1.8

tfv_config = {
    "TBR": TBR,
    "f_b": 0.015,
    "m_gas": 50,
    "A_global": A_global,
    "r_learn": 1,
    "t_pump": t_pump,
    "t_exh": t_exh,
    "t_ters": 6750,
    "t_freeze": 1800.0,
    "f_dir": f_dir,
    "t_detrit": 0,
    "f_detrit_split": 0.9999,
    "f_exh_split": 0.99,
    "eta_fuel_pump": 0.9,
    "eta_f": 0.5,
    "I_miv": max_ivc_inventory,
    "I_tfv_min": m_cryodistillation,
    "I_tfv_max": m_cryodistillation + 0.2,
    "I_mbb": max_bb_inventory,
    "eta_iv": eta_ivc,
    "eta_bb": 0.995,
    "eta_tfv": 0.998,
    "f_terscwps": 0.9999,
}

# %% [markdown]
#
# Now we set up a fuel cycle model
#
# We can run a single model and look at a typical result

# %%
model = EUDEMOFuelCycleModel(tfv_config, {})
model.run(time_dicts[0])
model.plot()

# %% [markdown]
# Now, let's run the fuel cycle model for all the timelines we generated

# %%
tfv_analysis = FuelCycleAnalysis(model)
tfv_analysis.run_model(time_dicts)

# %% [markdown]
# And the distributions for the start-up inventory and doubling time:

# %%
tfv_analysis.plot()

# %% [markdown]
# And finally, you can get the desired statistical results:

# %%
m_T_start_95 = tfv_analysis.get_startup_inventory("95th")
t_d_95 = tfv_analysis.get_doubling_time("95th")

m_T_start_mean = tfv_analysis.get_startup_inventory("mean")
t_d_mean = tfv_analysis.get_doubling_time("mean")

m_T_start_max = tfv_analysis.get_startup_inventory("max")
t_d_max = tfv_analysis.get_doubling_time("max")


print(f"The mean start-up inventory is: {m_T_start_mean:.2f} kg.")
print(f"The mean doubling time is: {t_d_mean:.2f} years.")
print(f"The 95th percentile start-up inventory is: {m_T_start_95:.2f} kg.")
print(f"The 95th percentile doubling time is: {t_d_95:.2f} years.")
print(f"The maximum start-up inventory is: {m_T_start_max:.2f} kg.")
print(f"The maximum doubling time is: {t_d_max:.2f} years.")
