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

"""
Usage of the 'spherical_harmonic_approximation' function.
"""

# %% [markdown]
# # spherical_harmonic_approximation Function
#
# This example illustrates the input and output of the
# bluemira spherical harmonics approximation function
# (spherical_harmonic_approximation) which can be used
# in coilset current and position optimisation for spherical tokamaks.

# %%
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.file import get_bluemira_path
from bluemira.display.auto_config import plot_defaults
from bluemira.display.plotter import Zorder
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.harmonics.harmonics_approx_functions import (
    PointType,
    spherical_harmonic_approximation,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_constraints import (
    SphericalHarmonicConstraint,
)
from bluemira.equilibria.optimisation.problem import (
    MinimalCurrentCOP,
)
from bluemira.equilibria.solve import (
    DudsonConvergence,
    PicardIterator,
)

plot_defaults()

# %pdb

# %%
# Data from EQDSK file
file_path = Path(
    get_bluemira_path("equilibria", subfolder="examples"), "SH_test_file.json"
)
eq = Equilibrium.from_eqdsk(file_path.as_posix(), from_cocos=3, qpsi_positive=False)
# Plot
f, ax = plt.subplots()
eq.plot(ax=ax)
eq.coilset.plot(ax=ax)
plt.show()

# %% [markdown]
# ## Inputs
#
# ### Required
#
# - eq = Our chosen bluemira equilibrium
#
# ### Optional
#
# - n_points: Number of desired collocation points
# - point_type: How the collocation points are distributed
# - grid_num: The number of points in x-direction and z-direction,
#   to use with grid point distribution
# - acceptable_fit_metric: how 'good' we require the approximation to be
# - psi_norm: 'None' will default to LCFS, otherwise choose the desired
#   normalised psi value of a closed flux surface that containe the core plasma
# - seed: Seed value to use with random point distribution
# - sig_figures: Number of significant figures for rounding during SH approximation
# - plot: Whether or not to plot the results

# %%
# Information needed for SH Approximation
(
    sh_coil_names,
    coil_current_harmonic_amplitudes,
    degree,
    fit_metric_value,
    approx_total_psi,
    r_t,
    sh_coilset_current,
) = spherical_harmonic_approximation(
    eq,
    n_points=10,
    point_type=PointType.GRID_POINTS,
    acceptable_fit_metric=0.02,
    psi_norm=0.98,
    seed=15,
    plot=True,
)

# %% [markdown]
# ## Outputs
#
# ### Results for use in optimisation
#
# - "sh_coil_names", names of the coils that can be used with SH approximation
# - "coil_current_harmonic_amplitudes", SH amplitudes for required number of degrees
# - "r_t", length scale for the approximation
#
# ### Informative outputs
#
# - "max_degree", number of degrees required for a SH approx with the desired fit metric
# - "fit_metric_value", fit metric achieved
# - "approx_total_psi", the total psi obtained using the SH approximation
# - "sh_coilset_current", the coil currents obtained using the approximation

# %%
print(sh_coil_names)
print(sh_coilset_current)

# %%
print(coil_current_harmonic_amplitudes)
print(degree)
print(fit_metric_value)
print(r_t)

# %%
psi = approx_total_psi
levels = np.linspace(np.amin(psi), np.amax(psi), 50)
plot = plt.subplot2grid((2, 2), (0, 0), rowspan=2, colspan=1)
plot.set_title("approx_total_psi")
plot.contour(
    eq.grid.x, eq.grid.z, psi, levels=levels, cmap="viridis", zorder=Zorder.PSI.value
)
plt.show()

# %% [markdown]
# ## Use in Optimisation Problem
#
# Now we will use the approximation to set up constraints for an optimisation problem.
# We use minimal current coilset optimisation problem with SH as the only constraints.
# This will try to minimise the sum of the currents squared while constraining the coil
# contribution to the core psi.

# %%
# Use results of the spherical harmonic approximation to create a set of coil constraints
sh_constraint = SphericalHarmonicConstraint(
    ref_harmonics=coil_current_harmonic_amplitudes,
    r_t=r_t,
    sh_coil_names=sh_coil_names,
)
# Make sure we only optimise with coils outside the sphere containing the core plasma by
# setting control coils using the list of appropriate coils
eq.coilset.control = list(sh_coil_names)

# %%
# Make a copy of the equilibria
sh_eq = deepcopy(eq)
# Set up a coilset optimisation problem using the spherical harmonic constraint
sh_con_len_opt = MinimalCurrentCOP(
    eq=sh_eq, coilset=sh_eq.coilset, max_currents=6.0e8, constraints=[sh_constraint]
)
# Find the optimised coilset
_ = sh_con_len_opt.optimise()

# Update plasma - one solve
sh_eq.solve()

# %%
# We should not need to solve the GS equation while optimising if the SH approximation
# is sufficiently good, but we can have a look at what happens.
sh_eq_solved = deepcopy(eq)
sh_con_len_opt = MinimalCurrentCOP(
    eq=sh_eq_solved,
    coilset=sh_eq_solved.coilset,
    max_currents=6.0e8,
    constraints=[sh_constraint],
)

# SOLVE
program = PicardIterator(
    sh_eq_solved,
    sh_con_len_opt,
    fixed_coils=True,
    convergence=DudsonConvergence(1e-2),
    relaxation=0.1,
    maxiter=100,
)
program()

# %%
# Plot the two approches
f, (ax_1, ax_2) = plt.subplots(1, 2)

sh_eq.plot(ax=ax_1)
ax_1.set_title("Coils Optimised")

sh_eq_solved.plot(ax=ax_2)
ax_2.set_title("Coils Optimised while GS solved")
plt.show()
