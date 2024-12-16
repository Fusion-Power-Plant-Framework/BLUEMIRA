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
An example of plotting the internal harmonics in toroidal coordinates about a focus
point.
"""

# %% [markdown]
# # Example of plotting the internal harmonics in toroidal coordinates

# This example shows how to plot the individual cos and sin toroidal harmonic
# contributions about a focus point.

# %% [markdown]
# ## Imports

# %%
import matplotlib.pyplot as plt
import numpy as np

from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import (  # noqa: E501
    legendre_q,
)
from bluemira.equilibria.plotting import PLOT_DEFAULTS
from bluemira.utilities.tools import cylindrical_to_toroidal

# %% [markdown]
# First we need to set up the grid in cylindrical coordinates over which we will solve,
# and define the location of the focus point.
# We will convert this to toroidal coordinates.
#
#
# To convert from cylindrical coordinates $(R, z)$ to toroidal coordinates
# $(\tau, \sigma)$ about a focus point $(R_0, z_0)$ we have the following relations:
# $$ \tau = \ln\frac{d_{1}}{d_{2}} $$
# $$ \sigma = {sign}(z - z_{0}) \arccos\frac{d_{1}^2 + d_{2}^2 - 4 R_{0}^2}{2 d_{1}
# d_{2}}  $$
# where
# $$ d_{1}^2 = (R + R_{0})^2 + (z - z_{0})^2 $$
# $$ d_{2}^2 = (R - R_{0})^2 + (z - z_{0})^2 $$
#
# The domains for the toroidal coordinates are $0 \le \tau < \infty$ and $-\pi < \sigma
# \le \pi$.

# %%
# Set up grid over which to solve
r = np.linspace(0, 6, 100)
z = np.linspace(-6, 6, 100)
R, Z = np.meshgrid(r, z)

# Define focus point
R_0 = 3.0
Z_0 = 0.0

# Convert to toroidal coordinates
tau, sigma = cylindrical_to_toroidal(R_0=R_0, z_0=Z_0, R=R, Z=Z)

# %% [markdown]
# Now we want to calculate and plot the internal harmonics in toroidal coordinates about
# the focus.
#
# We use the following equations
# $$ \psi_{sin} = R \sqrt{\cosh (\tau) - \cos (\sigma)} \textbf{Q}_{m - \frac{1}{2}}^1
# (\cosh \tau) \sin (m \sigma) $$
#
# $$ \psi_{cos} = R \sqrt{\cosh (\tau) - \cos (\sigma)} \textbf{Q}_{m - \frac{1}{2}}^1
# (\cosh \tau) \cos (m \sigma) $$
#
#
# $\textbf{Q}_{\nu}^{\mu}$ is Olver's definition of the associated Legendre function
# of the second kind. See [here](https://dlmf.nist.gov/14) or F. W. J. Olver
# (1997b) Asymptotics and Special Functions. A. K. Peters, Wellesley, MA. for more
# information.
# Here we have degree $\nu = 1$, and half integer order $\mu = m - \frac{1}{2}$.
#
# We evaluate these contributions to the harmonics about the focus and plot.
# %%
# Calculate and plot individual contributions from toroidal harmonics

nu = np.arange(0, 5)

# Setting up plots
fig_sin, axs_sin = plt.subplots(1, len(nu))
fig_sin.suptitle("sin")
fig_sin.supxlabel("R")
fig_sin.supylabel("Z")
fig_cos, axs_cos = plt.subplots(1, len(nu))
fig_cos.suptitle("cos")
fig_cos.supxlabel("R")
fig_cos.supylabel("Z")
nlevels = PLOT_DEFAULTS["psi"]["nlevels"]
cmap = PLOT_DEFAULTS["psi"]["cmap"]

for i_nu in range(len(nu)):
    foo = (
        R
        * np.sqrt(np.cosh(tau) - np.cos(sigma))
        * legendre_q(nu[i_nu] - 1 / 2, 1, np.cosh(tau))
    )
    psi_sin = foo * np.sin(nu[i_nu] * sigma)
    psi_cos = foo * np.cos(nu[i_nu] * sigma)
    axs_sin[i_nu].contour(R, Z, psi_sin, levels=nlevels, cmap=cmap)
    axs_sin[i_nu].title.set_text(f"m = {i_nu}")
    axs_cos[i_nu].contour(R, Z, psi_cos, levels=nlevels, cmap=cmap)
    axs_cos[i_nu].title.set_text(f"m = {i_nu}")

# %%
