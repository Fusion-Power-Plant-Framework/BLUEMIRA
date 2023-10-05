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
Fusion reactivity example
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from bluemira.display import plot_defaults
from bluemira.plasma_physics.reactions import reactivity

# %% [markdown]
# # Fusion Reactivity example
#
# Let's plot the reactivity of a couple of well-known fusion reactions.

# %%
plot_defaults()

temperature = np.linspace(10e6, 100e6, 1000)  # [K]

sigma_v_DT = reactivity(temperature, "D-T")
sigma_v_DD = reactivity(temperature, "D-D")
sigma_v_DHe3 = reactivity(temperature, "D-He3")  # noqa: N816

f, ax = plt.subplots()
ax.loglog(temperature, sigma_v_DT, label="D-T")
ax.loglog(temperature, sigma_v_DD, label="D-D")
ax.loglog(temperature, sigma_v_DHe3, label="D-He3")

ax.grid(which="both")
ax.set_xlabel("T [K]")
ax.set_ylabel("$\\sigma_{v}$ [$m^{3}/s$]")
ax.legend()
plt.show()
