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
Calculate solution due to a single wire as a sum of toroidal harmonics.
"""

# %% [markdown]
# # Calculating the flux solution due to a single wire as a sum of toroidal harmonics
#
# This example illustrates how the magnetic flux due to a single wire can be
# calculated by using toroidal harmonics about a focus point.

# %% [markdown]
# ## Imports

# %%
from math import factorial

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from PIL import Image

from bluemira.base.constants import MU_0
from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import (  # noqa: E501
    legendre_p,
    legendre_q,
)
from bluemira.equilibria.plotting import PLOT_DEFAULTS
from bluemira.utilities.tools import cylindrical_to_toroidal, toroidal_to_cylindrical

# %% [markdown]
# We want to calculate the psi field due to a circular wire using toroidal
# harmonics.
#
# The relevant equations for psi in toroidal coordinates, $(\tau, \sigma)$, are
#
# $$ \psi = R A $$
#
# $$ A(\tau, \sigma) = \sum_{m=0}^{\infty} \sum_{\cos, \sin} A_m^{\cos, \sin} \epsilon_m
# m! \sqrt{\frac{2}{\pi}} \Delta^{\frac{1}{2}} \textbf{Q}_{m-\frac{1}{2}}^1(\cosh \tau)_
# {\sin}^{\cos} (m \sigma)  $$
#
# $$ A_m^{\cos, \sin} = \frac{\mu_0 I_c}{2^{\frac{5}{2}}} \prod_{i=0}^{m-1} \left( 1 +
# \frac{1}{2(m-i)}\right) \frac{\sinh(\tau_c)}
# {\Delta_c^{\frac{1}{2}}} P_{m - \frac{1}{2}}^{-1}(\cosh(\tau_c))_{\sin}^{\cos}(m
# \sigma_c) $$
#
# Here, $m$ is the poloidal mode number, and we have the following
# $$ \Delta = \cosh \tau - \cos \sigma, $$
# $$ \Delta_c = \cosh \tau_c - \cos \sigma_c $$
# $$ \epsilon_0 = 1, \epsilon_{m\ge1}=2$$
# $P_{\nu}^{\mu}$ is the associated Legendre function of the first kind, of degree $\nu$
# and order $\mu$. $\textbf{Q}_{\nu}^{\mu}$ is Olver's definition of the associated
# Legendre function of the second kind. See [here](https://dlmf.nist.gov/14) or F. W. J.
# Olver
# (1997b) Asymptotics and Special Functions. A. K. Peters, Wellesley, MA. for more
# information.
#
# The following image shows the psi field we are expecting to obtain at the end of this
# example.
# %%
single_wire_image = Image.open("single_wire_output_image.png")
display(single_wire_image)
# %% [markdown]
# First we define the location in cylindrical coordinates of the focus $(R_0, z_0)$ and
# of the wire $(R_c, z_c)$, and the current in the wire, $I_c$.
#
# We then convert the location of the wire from cylindrical $(R_c, z_c)$ to toroidal
# coordinates $(\tau_c, \sigma_c)$ using the following relations:
# $$ \tau_c = \ln\frac{d_{1}}{d_{2}} $$
# $$ \sigma_c = {sign}(z - z_{0}) \arccos\frac{d_{1}^2 + d_{2}^2 - 4 R_{0}^2}{2 d_{1}
# d_{2}} $$
# where
# $$ d_{1}^2 = (R_{c} + R_{0})^2 + (z_c - z_{0})^2 $$
# $$ d_{2}^2 = (R_{c} - R_{0})^2 + (z_c - z_{0})^2 $$

#
# We need the location of the wire in toroidal coordinates. We also need to
# approximate $\tau$ at the focus instead of using coordinate transform functions
# as this would result in divide by zero errors. We use these values to create a
# grid in toroidal coordinates.

# Once we have the grid in toroidal coordinates, we can convert this to cylindrical
# coordinates for use later using the following relations:
# $$ R = R_0 \frac{\sinh\tau}{\cosh \tau - \cos \sigma}$$
# $$ z - z_0 = R_0 \frac{\sin \sigma}{\cosh \tau - \cos \sigma}$$
# %%
# Wire location
R_c = 5.0
Z_c = 1.0
I_c = 5e6

# Focus of toroidal coordinates
R_0 = 3.0
Z_0 = 0.2

# Location of wire in toroidal coordinates
tau_c, sigma_c = cylindrical_to_toroidal(R_0=R_0, z_0=Z_0, R=R_c, Z=Z_c)

# Using approximate value for d2_min to avoid infinities
# Approximating tau at the focus instead of using coordinate transform functions
# (in order to avoid divide by 0 errors)
d2_min = 0.05
tau_max = np.log(2 * R_0 / d2_min)
n_tau = 200
tau = np.linspace(tau_c, tau_max, n_tau)
n_sigma = 150
sigma = np.linspace(-np.pi, np.pi, n_sigma)

# Create grid in toroidal coordinates
tau, sigma = np.meshgrid(tau, sigma)

# Convert to cylindrical coordinates
R, Z = toroidal_to_cylindrical(R_0=R_0, z_0=Z_0, tau=tau, sigma=sigma)


# %% [markdown]
# Now we want to calculate the following coefficients
# $$ A_m = \frac{\mu_0 I_c}{2^{\frac{5}{2}}} factorial\_term \frac{\sinh(\tau_c)}
# {\Delta_c^{\frac{1}{2}}} P_{m - \frac{1}{2}}^{-1}(\cosh(\tau_c)) $$
# $$ A_m^{sin} = A_m \sin(m \sigma_c) $$
# $$ A_m^{cos} = A_m \cos(m \sigma_c) $$
# which are functions of $m$ and the coil position,
# where $$ \Delta_c = \cosh(\tau_c) - \cos(\sigma_c) $$
# and $$ factorial\_term = \prod_{i=0}^{m-1} \left( 1 + \frac{1}{2(m-i)}\right) $$
#

# %%
# Useful combinations
Delta = np.cosh(tau) - np.cos(sigma)
Deltac = np.cosh(tau_c) - np.cos(sigma_c)

# Calculate coefficients
m_max = 5
Am_cos = np.zeros(m_max + 1)
Am_sin = np.zeros(m_max + 1)

for m in range(m_max + 1):
    factorial_term = 1 if m == 0 else np.prod(1 + 0.5 / np.arange(1, m + 1))
    A_m = (
        (MU_0 * I_c / 2 ** (5 / 2))
        * factorial_term
        * (np.sinh(tau_c) / np.sqrt(Deltac))
        * legendre_p(m - 1 / 2, 1, np.cosh(tau_c))
    )
    Am_cos[m] = A_m * np.cos(m * sigma_c)
    Am_sin[m] = A_m * np.sin(m * sigma_c)

# %% [markdown]
# Now we can use the following
#
# $$ A(\tau, \sigma) = \sum_{m=0}^{\infty} A_m^{\cos} \epsilon_m m! \sqrt{\frac{2}{\pi}}
# \Delta^{\frac{1}{2}} \textbf{Q}_{m-\frac{1}{2}}^{1}(\cosh \tau) \cos(m \sigma) + A_m^
# {\sin}
# \epsilon_m m! \sqrt{\frac{2}{\pi}} \Delta^{\frac{1}{2}}
# \textbf{Q}_{m-\frac{1}{2}}^{1}(\cosh \tau) \sin(m \sigma) $$

# along with $$\psi = R A$$
# to calculate the solution and plot the psi graph. Here we have that
# $ \epsilon_0 = 1$ and $\epsilon_{m\ge 1} = 2$.

# %%
epsilon = 2 * np.ones(m_max + 1)
epsilon[0] = 1
A = np.zeros_like(R)

for m in range(m_max + 1):
    A += Am_cos[m] * epsilon[m] * factorial(m) * np.sqrt(2 / np.pi) * np.sqrt(
        Delta
    ) * legendre_q(m - 1 / 2, 1, np.cosh(tau)) * np.cos(m * sigma) + Am_sin[m] * epsilon[
        m
    ] * factorial(m) * np.sqrt(2 / np.pi) * np.sqrt(Delta) * legendre_q(
        m - 1 / 2, 1, np.cosh(tau)
    ) * np.sin(m * sigma)

psi = R * A
nlevels = PLOT_DEFAULTS["psi"]["nlevels"]
cmap = PLOT_DEFAULTS["psi"]["cmap"]
plt.contour(R, Z, psi, nlevels, cmap=cmap)
plt.xlabel("R")
plt.ylabel("Z")
plt.title("Psi")
# %%
