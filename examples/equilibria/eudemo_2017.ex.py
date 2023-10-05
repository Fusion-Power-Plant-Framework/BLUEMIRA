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
Attempt at recreating the EU-DEMO 2017 reference equilibria from a known coilset.
"""

# %% [markdown]
#
# # EU-DEMO 2017 reference breakdown and equilibrium benchmark

# %%
import contextlib
import json
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython

from bluemira.base.file import get_bluemira_path
from bluemira.base.look_and_feel import bluemira_print
from bluemira.display import plot_defaults
from bluemira.equilibria.coils import Coil, CoilSet
from bluemira.equilibria.equilibrium import Breakdown, Equilibrium
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.optimisation.constraints import (
    CoilFieldConstraints,
    CoilForceConstraints,
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
    PsiBoundaryConstraint,
)
from bluemira.equilibria.optimisation.problem import (
    BreakdownCOP,
    MinimalCurrentCOP,
    OutboardBreakdownZoneStrategy,
    UnconstrainedTikhonovCurrentGradientCOP,
)
from bluemira.equilibria.physics import calc_beta_p, calc_li3, calc_psib
from bluemira.equilibria.profiles import (
    BetaIpProfile,
    BetaLiIpProfile,
    CustomProfile,
    DoublePowerFunc,
)
from bluemira.equilibria.solve import PicardIterator

# %% [markdown]
#
# Load the reference equilibria from EFDA_D_2MUW9R

# %%
plot_defaults()

with contextlib.suppress(AttributeError):
    get_ipython().run_line_magic("matplotlib", "qt")


path = get_bluemira_path("equilibria", subfolder="examples")
name = "EUDEMO_2017_CREATE_SOF_separatrix.json"
with open(Path(path, name)) as file:
    data = json.load(file)

sof_xbdry = data["xbdry"]
sof_zbdry = data["zbdry"]

# %% [markdown]
#
# Make the same CoilSet as CREATE

# %%
x = [5.4, 14, 17.75, 17.75, 14.0, 7, 2.77, 2.77, 2.77, 2.77, 2.77]
z = [9.26, 7.9, 2.5, -2.5, -7.9, -10.5, 7.07, 4.08, -0.4, -4.88, -7.86]
dx = [0.6, 0.7, 0.5, 0.5, 0.7, 1.0, 0.4, 0.4, 0.4, 0.4, 0.4]
dz = [0.6, 0.7, 0.5, 0.5, 0.7, 1.0, 2.99 / 2, 2.99 / 2, 5.97 / 2, 2.99 / 2, 2.99 / 2]

coils = []
j = 1
for i, (xi, zi, dxi, dzi) in enumerate(zip(x, z, dx, dz)):
    if j > 6:  # noqa: PLR2004
        j = 1
    ctype = "PF" if i < 6 else "CS"  # noqa: PLR2004
    coil = Coil(
        xi,
        zi,
        current=0,
        dx=dxi,
        dz=dzi,
        ctype=ctype,
        name=f"{ctype}_{j}",
    )
    coils.append(coil)
    j += 1

coilset = CoilSet(*coils)

# Assign current density and peak field constraints
coilset.assign_material("CS", j_max=16.5e6, b_max=12.5)
coilset.assign_material("PF", j_max=12.5e6, b_max=11)
coilset.fix_sizes()
coilset.discretisation = 0.3

coilset.plot()

# %% [markdown]
#
# Define parameters

# %%
# Machine parameters
I_p = 19.07e6  # A
beta_p = 1.141
l_i = 0.8
R_0 = 8.938
Z_0 = 0.027454
B_0 = 4.8901  # ???
A = 3.1
kappa_95 = 1.65
delta_95 = 0.33
tau_flattop = 2 * 3600
v_burn = 4.220e-2  # V
c_ejima = 0.3

# Breakdown constraints (I can't quite get it with 3mT..) I've gotten close to 305 V.s,
# but only using a smaller low-field region.
# This is quite a sensitive optimisation, and is possibly a multi-modal space
# May want to think about optimising with a stochastic optimiser, and including
# a parametric location of the breakdown point...
x_zone = 9.84  # ??
z_zone = 0.0  # ??
r_zone = 2.0  # ??
b_zone_max = 0.003  # T

# Coil constraints
PF_Fz_max = 450e6  # [N]
CS_Fz_sum = 300e6  # [N]
CS_Fz_sep = 350e6  # [N]

# %% [markdown]
# Use the same grid as CREATE (but less discretised):

# %%
grid = Grid(2, 16.0, -9.0, 9.0, 100, 100)

# %% [markdown]
#
# Set up the Breakdown object

# %%
field_constraints = CoilFieldConstraints(coilset, coilset.b_max, tolerance=1e-6)
force_constraints = CoilForceConstraints(
    coilset, PF_Fz_max, CS_Fz_sum, CS_Fz_sep, tolerance=1e-6
)

max_currents = coilset.get_max_current(0.0)
breakdown = Breakdown(deepcopy(coilset), grid)

bd_opt_problem = BreakdownCOP(
    breakdown.coilset,
    breakdown,
    OutboardBreakdownZoneStrategy(R_0, A, 0.225),
    opt_algorithm="COBYLA",
    opt_conditions={"max_eval": 3000, "ftol_rel": 1e-6},
    max_currents=max_currents,
    B_stray_max=1e-3,
    B_stray_con_tol=1e-6,
    n_B_stray_points=10,
    constraints=[field_constraints, force_constraints],
)

coilset = bd_opt_problem.optimise(x0=max_currents).coilset

bluemira_print(f"Breakdown psi: {breakdown.breakdown_psi*2*np.pi:.2f} V.s")
# %% [markdown]
#
# Calculate SOF and EOF plasma boundary fluxes

# %%
psi_sof = calc_psib(breakdown.breakdown_psi * 2 * np.pi, R_0, I_p, l_i, c_ejima)
psi_eof = psi_sof - tau_flattop * v_burn

# CREATE then knocked off an extra 10 V.s for misc plasma stuff I didn't look into

psi_sof -= 10
psi_eof -= 10

# %% [markdown]
#
# Set up a parameterised profile
# Here you can use a CustomProfile, by feeding in arrays describing
# your p' and FF' flux functions which are linearly interpolated.

# Or you can use either BetaIpProfile or BetaLiIpProfile to constrain
# the plasma integrals, optimising the shape of the flux functions
# to match these.

# Comment out the relevant lines below to explore the different
# behaviour.
# %%
profiles = CustomProfile(
    np.array([86856, 86506, 84731, 80784, 74159, 64576, 52030, 36918, 20314, 4807, 0.0]),
    -np.array(
        [0.125, 0.124, 0.122, 0.116, 0.106, 0.093, 0.074, 0.053, 0.029, 0.007, 0.0]
    ),
    R_0=R_0,
    B_0=B_0,
    I_p=I_p,
)

shape = DoublePowerFunc([2, 1])
profiles = BetaIpProfile(beta_p, I_p, R_0, B_0, shape=shape)
profiles = BetaLiIpProfile(
    beta_p, l_i, I_p, R_0, B_0, shape=shape, li_min_iter=0, li_rel_tol=0.001
)


# %% [markdown]
# Solve the SOF and EOF equilibria

# %%
reference_eq = Equilibrium(
    deepcopy(coilset),
    grid,
    profiles,
)

# Make a set of magnetic constraints for the equilibria... I got lazy here,
# this is just:
#   * LCFS boundary fluxes
#   * Field null at lower X-point
#   * divertor legs are not treated, but could easily be added

sof_xbdry = np.array(sof_xbdry)[::15]
sof_zbdry = np.array(sof_zbdry)[::15]

isoflux = IsofluxConstraint(
    sof_xbdry,
    sof_zbdry,
    sof_xbdry[0],
    sof_zbdry[0],
    tolerance=1e-3,
    constraint_value=0.25,  # Difficult to choose...
)
xp_idx = np.argmin(sof_zbdry)
x_point = FieldNullConstraint(
    sof_xbdry[xp_idx],
    sof_zbdry[xp_idx],
    tolerance=1e-3,
)

ref_opt_problem = UnconstrainedTikhonovCurrentGradientCOP(
    reference_eq.coilset,
    reference_eq,
    MagneticConstraintSet([isoflux, x_point]),
    gamma=1e-7,
)

program = PicardIterator(reference_eq, ref_opt_problem, fixed_coils=True, relaxation=0.2)
program()


sof_psi_boundary = PsiBoundaryConstraint(
    sof_xbdry,
    sof_zbdry,
    psi_sof / (2 * np.pi),
    tolerance=0.5,
)

sof = deepcopy(reference_eq)

sof_opt_problem = MinimalCurrentCOP(
    sof.coilset,
    sof,
    opt_algorithm="SLSQP",
    opt_conditions={"max_eval": 1000, "ftol_rel": 1e-6},
    max_currents=max_currents,
    constraints=[sof_psi_boundary, x_point],
)

iterator = PicardIterator(
    sof, sof_opt_problem, plot=True, fixed_coils=True, relaxation=0.2
)
iterator()


eof_psi_boundary = PsiBoundaryConstraint(
    sof_xbdry,
    sof_zbdry,
    psi_eof / (2 * np.pi),
    tolerance=0.5,
)

eof = deepcopy(reference_eq)
eof_opt_problem = MinimalCurrentCOP(
    eof.coilset,
    eof,
    opt_algorithm="SLSQP",
    opt_conditions={"max_eval": 1000, "ftol_rel": 1e-6},
    max_currents=max_currents,
    constraints=[eof_psi_boundary, x_point],
)


iterator = PicardIterator(
    eof, eof_opt_problem, plot=True, relaxation=0.2, fixed_coils=True
)
iterator()

# %% [markdown]
# Plot the results

# %%
f, ax = plt.subplots(1, 3)
breakdown.plot(ax[0])
breakdown.coilset.plot(ax[0])
sof.plot(ax[1])
sof.coilset.plot(ax[1])
eof.plot(ax[2])
eof.coilset.plot(ax[2])

sof_psi = 2 * np.pi * sof.psi(*sof._x_points[0][:2])
eof_psi = 2 * np.pi * eof.psi(*eof._x_points[0][:2])
ax[1].set_title("$\\psi_{b}$ = " + f"{sof_psi:.2f} V.s")
ax[2].set_title("$\\psi_{b}$ = " + f"{eof_psi:.2f} V.s")


bluemira_print(f"SOF: beta_p: {calc_beta_p(sof):.2f} l_i: {calc_li3(sof):.2f}")
bluemira_print(f"EOF: beta_p: {calc_beta_p(eof):.2f} l_i: {calc_li3(eof):.2f}")

plt.show()
