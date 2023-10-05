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
Test for plasmod run
"""

# %%
import shutil
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt

from bluemira.base.file import get_bluemira_path, get_bluemira_root
from bluemira.base.logs import set_log_level
from bluemira.codes import plasmod

# %% [markdown]
# # Configuring the PLASMOD solver
#
# PLASMOD is one of the codes bluemira can use to compliment a reactor design.
# As with any of the external codes bluemira uses, a solver object is created.
# The solver object abstracts away most of the complexities of running different
# programs within bluemira.
#
# ## Setting up
#
# ### Logging
# To enable debug logging run the below cell

# %%
set_log_level("DEBUG")

# %% [markdown]
# ### Binary Location
# Firstly if plasmod is not on your system path, we need to provide the
# binary location to the solver.

# %%
if plasmod_binary := shutil.which("plasmod"):
    PLASMOD_PATH = Path(plasmod_binary).parent
else:
    PLASMOD_PATH = Path(Path(get_bluemira_root()).parent, "plasmod/bin")
binary = Path(PLASMOD_PATH, "plasmod").as_posix()

# %% [markdown]
# ### Creating the solver object
# bluemira-plasmod parameter names have been mapped across where possible.
# Some example parameters have been set here in `new_params`
# before being converted into a bluemira configuration store.
#
# These parameters mirror running the plasmod input demoH.i reference configuration

# %%
new_params = {
    "A": 3.1,
    "R_0": 9.002,
    "I_p": 17.75e6,
    "B_0": 5.855,
    "V_p": -2500,
    "v_burn": -1.0e6,
    "kappa_95": 1.652,
    "delta_95": 0.333,
    "delta": 0.38491934960310104,
    "kappa": 1.6969830041844367,
}


# %% [markdown]
# Some values are not linked into bluemira. These plasmod parameters can be set
# directly in `problem_settings`.
# H-factor is set here as input therefore we will force plasmod to
# optimise to that H-factor.

# %%
problem_settings = {
    "pfus_req": 0.0,
    "pheat_max": 130.0,
    "q_control": 130.0,
    "Hfact": 1.1,
    "i_modeltype": "GYROBOHM_1",
    "i_equiltype": "Ip_sawtooth",
    "i_pedestal": "SAARELMA",
}

# %% [markdown]
# There are also some model choices that can be set in `problem_settings`.
# The available models with their options and explanations
# can be seen by running the below snippet.

# %%
for var_name in dir(plasmod.mapping):
    if "Model" in var_name and var_name != "Model":
        model = getattr(plasmod.mapping, var_name)
        model.info()

# %% [markdown]
# Finally the `build_config` dictionary collates the configuration settings for
# the solver

# %%
build_config = {
    "problem_settings": problem_settings,
    "binary": binary,
    "directory": get_bluemira_path("", subfolder="generated_data"),
}

# %% [markdown]
# Now we can create the solver object with the parameters and build configuration

# %%
solver = plasmod.Solver(params=new_params, build_config=build_config)


# %% [markdown]
# These few functions are helpers to simplify the remainder of the tutorial.
# The first shows a few of the output scalar values and the second plots a
# given profile.


# %%
def print_outputs(solver):
    """
    Print plasmod scalars
    """
    outputs = solver.plasmod_outputs()
    print(f"Fusion power [MW]: {solver.params.P_fus.value_as('MW')}")
    print(f"Additional heating power [MW]: {outputs.Paux / 1E6}")
    print(f"Radiation power [MW]: {solver.params.P_rad.value_as('MW')}")
    print(
        f"Transport power across separatrix [MW]: {solver.params.P_sep.value_as('MW')}"
    )
    print(f"{solver.params.q_95}")
    print(f"{solver.params.I_p}")
    print(f"{solver.params.l_i}")
    print(f"{solver.params.v_burn}")
    print(f"{solver.params.Z_eff}")
    print(f"H-factor [-]: {outputs.Hfact}")
    print(
        "Divertor challenging criterion (P_sep * Bt /(q95 * R0 * A)) [-]:"
        f" {outputs.psepb_q95AR}"
    )
    print(
        "H-mode operating regime f_LH = P_sep/P_LH [-]:"
        f" {solver.params.P_sep.value / solver.params.P_LH.value}"
    )
    print(f"{solver.params.tau_e}")
    print(f"Protium fraction [-]: {outputs.cprotium}")
    print(f"Helium fraction [-]: {outputs.che}")
    print(f"Xenon fraction [-]: {outputs.cxe}")
    print(f"Argon fraction [-]: {outputs.car}")


def plot_profile(solver, profile, var_unit):
    """
    Plot plasmod profile
    """
    prof = solver.get_profile(profile)
    x = solver.get_profile(plasmod.Profiles.x)
    _, ax = plt.subplots()
    ax.plot(x, prof)
    ax.set(xlabel="x (-)", ylabel=profile.name + " (" + var_unit + ")")
    ax.grid()
    plt.show()


# %% [markdown]
# ### Running the solver
# Very simply use the `run` method of the solver

# %%
solver.execute(plasmod.RunMode.RUN)

# %% [markdown]
# ### Using the results
# Outputs can be accessed through 3 ways depending on the
# linking mechanism.
# 1. Through the `params` attribute which contains
#    all the bluemira linked parameters
# 2. Profiles can be accessed through the `get_profile` function, using
#    a value form the `plasmod.Profile` enum
# 3. Unlinked plasmod parameters can be accessed through the
#    `plasmod_outputs` function
#
# The list of available profiles can be seen by running the below cell.
# A good exercise would be to try showing a different profile in the plot.

# %%
print("Profiles")
pprint(list(plasmod.mapping.Profiles))  # noqa: T203

# %%
plot_profile(solver, plasmod.Profiles.Te, "keV")
print_outputs(solver)

# %% [markdown]
# ### Plotting the results
# There is a default set of output profiles that can be plotted easily:

# %%
plasmod.plot_default_profiles(solver)


# %% [markdown]
# ### Rerunning with modified settings
# #### Changing the transport model

# %%
solver.problem_settings["i_modeltype"] = plasmod.TransportModel.GYROBOHM_2
solver.execute(plasmod.RunMode.RUN)
print_outputs(solver)


# %% [markdown]
# #### Fixing fusion power to 2000 MW and safety factor `q_95` to 3.5.
# Plasmod calculates the additional heating power and the plasma current

# %%
solver.params.q_95.set_value(3.5, "Input 1")

solver.problem_settings["pfus_req"] = 2000.0
solver.problem_settings["i_equiltype"] = plasmod.EquilibriumModel.q95_sawtooth
solver.problem_settings["isawt"] = plasmod.SafetyProfileModel.SAWTEETH
solver.problem_settings["q_control"] = 50.0

solver.execute(plasmod.RunMode.RUN)
print_outputs(solver)

# %% [markdown]
# #### Setting heat flux on divertor target to 10 MW/m²
# plasmod calculates the argon concentration to fulfill the constraint

# %%
solver.problem_settings["qdivt_sup"] = 10.0
solver.execute(plasmod.RunMode.RUN)
print_outputs(solver)

# %% [markdown]
# #### Changing the mapping sending or receiving
# The mapping can be changed on a given parameter or set of parameters.
# Notice how the value of `q_95` doesn't change in the output,
# even though its value has in the parameter (the previous value of 3.5 is used).

# %%
solver.modify_mappings({"q_95": {"send": False}})
solver.params.q_95.set_value(5, "Input 2")
solver.execute(plasmod.RunMode.RUN)
print_outputs(solver)
print("\nq_95 value history\n")
for hist in solver.params.q_95.history():
    print(hist)
