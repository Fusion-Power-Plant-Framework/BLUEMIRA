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
Usage of the 'spherical_harmonic_approximation' function.
"""

# %% [markdown]
# # spherical_harmonic_approximation Function
#
# This example illustrates the input and output of the
# Bluemira spherical harmonics approximation function
# (spherical_harmonic_approximation) which can be used
# in coilset current and position optimisation for spherical tokamaks.
# For an example of how spherical_harmonic_approximation is used
# please see, <TODO add link>

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.file import get_bluemira_path
from bluemira.display.auto_config import plot_defaults
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.harmonics import spherical_harmonic_approximation

plot_defaults()

# %pdb

# %%
# Data from EQDSK file
file_path = Path(
    get_bluemira_path("equilibria", subfolder="examples"), "SH_test_file.json"
)

# Plot
eq = Equilibrium.from_eqdsk(file_path.as_posix())
f, ax = plt.subplots()
eq.plot(ax=ax)
eq.coilset.plot(ax=ax)
plt.show()

# %% [markdown]
# ## Inputs
#
# ### Required
#
# - eq = Our chosen Bluemira Equilibrium
#
# ### Optional
#
# - n_points: Number of desired collocation points
# - point_type: How the collocation points are distributed
# - acceptable_fit_metric: how 'good' we require the approximation to be
# - r_t: typical length scale for spherical harmonic approximation
# - extra_info: set this to true if you wish to return additional
#               information and plot the results.

# %%
# Information needed for SH Approximation
(
    sh_coilset,
    r_t,
    coil_current_harmonic_amplitudes,
    degree,
    fit_metric_value,
    approx_total_psi,
) = spherical_harmonic_approximation(
    eq,
    n_points=50,
    point_type="random_plus_extrema",
    acceptable_fit_metric=0.03,
    plot=True,
)

# %% [markdown]
# ## Outputs
#
# spherical_harmonic_approximation outputs a dictionary of results
# that can be used in optimisation.
#
# ### Always output
#
# - "coilset", coilset to use with SH approximation
# - "r_t", typical length scale for spherical harmonic approximation
# - "harmonic_amplitudes", SH coefficients/amplitudes for required number of degrees
# - "max_degree", number of degrees required for a SH approx with the desired fit metric

# %%
print(sh_coilset)

# %%
print(r_t)

# %%
print(coil_current_harmonic_amplitudes)

# %%
print(degree)

# %% [markdown]
# ### Output on request
#
# - "fit_metric_value", fit metric achieved
# - "approx_total_psi", the total psi obtained using the SH approximation

# %%
print(fit_metric_value)

# %%
psi = approx_total_psi
levels = np.linspace(np.amin(psi), np.amax(psi), 50)
plot = plt.subplot2grid((2, 2), (0, 0), rowspan=2, colspan=1)
plot.set_title("approx_total_psi")
plot.contour(eq.grid.x, eq.grid.z, psi, levels=levels, cmap="viridis", zorder=8)
plt.show()
