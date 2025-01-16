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
Coilset Optimisation Problem Tutorial
"""

# %% [markdown]
# # Coilset optimisation problem tutorial
#
#
# This tutorial finds a Spherical Tokamak (ST) equilibrium in a
# double null configuration,
# using a constrained optimisation method with bound constraints on
# the maximum coil currents and some additional constraints.
#

# ## Introduction

# In this example, we will outline how to specify a `CoilsetOptimisationProblem`
# that specifies how an 'optimised' coilset state is found during the Free Boundary
# Equilibrium solve step.

# %%

import matplotlib.pyplot as plt
import numpy as np

from bluemira.equilibria.coils import Coil, CoilSet, SymmetricCircuit
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.optimisation.constraints import (
    CoilFieldConstraints,
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
)
from bluemira.equilibria.optimisation.problem import (
    TikhonovCurrentCOP,
    UnconstrainedTikhonovCurrentGradientCOP,
)
from bluemira.equilibria.profiles import CustomProfile
from bluemira.equilibria.solve import DudsonConvergence, PicardIterator

# %% [markdown]

# ## The `CoilsetOptimisationProblem` class

# The `CoilsetOptimisationProblem` class is intended to be the abstract
# base class for coilset optimisation problems across Bluemira.
#
#
# The goal of the `CoilsetOptimisationProblem` is to be able to return
# an optimised coilset, judged according to the provided objective,
# subject to the set of constraints, using a numerical search algorithm
# subject to optimiser conditions and optimiser parameters.
#
# Subclasses of `CoilsetOptimisationProblem` thus should provide an
# `optimise()` method that returns the `CoilsetOptimiserResult`,
# containing the coilset optimised according to a specific objective
# function for that subclass.
#
# ### Ingredients for a Coilset optimisation problem
# - **Coilset**: the set of coils we want to optimise
#   - the coilset 'state' will be representated by an array
#   - **coilset_state**: state vector representing degrees of
#       freedom of the coilset
# - **Objective Function**: the function we wish to minimise.
#
#   - in bluemira, `ObjectiveFunction` is the base class for objective
#   functions.
#
# - **Constraints**: optimisation constraints that need to be satisfied.
#
#   - in bluemira, `UpdateableConstraint` is the abstract base mixin class
#   that is updateable.

# - **Optimisation algorithm**: the numerical algorithm used to optimise the
#   coilset state array.
#   - Usually based on NLOpt.
#   - There are several optimisation algorithms that can be used within
#     bluemira.
#     Including gradient and non-gradient based.
#       - SLSQP
#       - COBYLA
#       - SBPLX
#
#     See the :py:class:`~bluemira.optimisation._algorithm.Algorithm`
#     enum for a reliably up-to-date list.
#   - Apart from the algorithm itself, you may also specify
#       - **optimisation conditions**: The stopping conditions for the optimiser.
#       - **optimisation parameters**: The algorithm-specific optimisation
#       parameters.
#
# # Example: Tikhonov Current COP
# `TikhonovCurrentCOP` is a `CoilsetOptimisationProblem` for coil currents
# subject to maximum current bounds with/without constraints that must be
# satisfied during the coilset optimisation.
#
# ## Parameters
#
# - **coilset**: `CoilSet` to optimise.
#
# - **eq**: `Equilibrium` object used to update magnetic field targets.
#
# - **targets**: Set of magnetic field targets to use in objective function.
#
# - **gamma**: Tikhonov regularisation parameter in units of [A⁻¹].
#
# - **opt_algorithm**, **opt_conditions** and **opt_parameters**
#
# - **max_currents**: Maximum allowed current for each independent coil
# current in coilset [A].
#
# - **constraints**: Optional list of `UpdatableConstraint` objects
#
# ## Coilset
#
# We first define the `CoilSet` to be optimised in our OptimisationProblem.
#
# We will consider the coilset to have positional symmetry about z=0, hence
# the use of SymmetricCircuit class.
# %%
coil_x = [1.55, 6.85, 6.85, 1.55, 3.2, 5.7, 5.3]
coil_z = [7.85, 4.95, 3.15, 6.1, 8.0, 7.8, 5.50]
coil_dx = [0.45, 0.5, 0.5, 0.3, 0.6, 0.5, 0.25]
coil_dz = [0.5, 0.8, 0.8, 0.8, 0.5, 0.5, 0.5]

coils = []

for i, (xi, zi, dxi, dzi) in enumerate(
    zip(coil_x, coil_z, coil_dx, coil_dz, strict=False)
):
    coil = SymmetricCircuit(
        Coil(x=xi, z=zi, dx=dxi, dz=dzi, name=f"PF_{i + 1}", ctype="PF")
    )
    coils.append(coil)
coilset = CoilSet(*coils)
coilset.b_max = np.full(coilset.n_coils(), 20)

# %% [markdown]
# ## Equilibrium
#
# We also specify an initial Equilibrium state to be used in the optimisation.
#
# For this we need a Grid and some plasma profiles
# %%

# Intialise some parameters
R_0 = 2.6
Z_0 = 0
B_t = 1.9
I_p = 16e6

r0, r1 = 0.2, 8
z0, z1 = -8, 8
nx, nz = 129, 257
grid = Grid(r0, r1, z0, z1, nx, nz)

# fmt: off
pprime = np.array([
    -850951, -844143, -782311, -714610, -659676, -615987, -572963, -540556, -509991,
    -484261, -466462, -445186, -433472, -425413, -416325, -411020, -410672, -406795,
    -398001, -389309, -378528, -364607, -346119, -330297, -312817, -293764, -267515,
    -261466, -591725, -862663,
])
ffprime = np.array([
    7.23, 5.89, 4.72, 3.78, 3.02, 2.39, 1.86, 1.43, 1.01, 0.62, 0.33, 0.06, -0.27,
    -0.61, -0.87, -1.07, -1.24, -1.18, -0.83, -0.51, -0.2, 0.08, 0.24, 0.17, 0.13,
    0.1, 0.07, 0.05, 0.15, 0.28,
])
# fmt: on

profiles = CustomProfile(pprime, ffprime, R_0=R_0, B_0=B_t, I_p=I_p)
eq = Equilibrium(coilset, grid, profiles, force_symmetry=True, vcontrol=None, psi=None)

# %% [markdown]

# ## Targets

# The `OptimisationObjective` figure of merit for `TikhonovCurrentCOP` is the
# regularised least-squares deviation of a provided Equilibrium from a set of
# magnetic field targets (eg. isoflux targets).
#
# We use a default set of isoflux targets here for this example.

# %%

x_lcfs = np.array([1.0, 1.67, 4.0, 1.73])
z_lcfs = np.array([0, 4.19, 0, -4.19])

lcfs_isoflux = IsofluxConstraint(
    x_lcfs,
    z_lcfs,
    ref_x=x_lcfs[2],
    ref_z=z_lcfs[2],
    tolerance=np.mean(eq.psi(x_lcfs, z_lcfs)) * 1e-2,
    constraint_value=0.1,
)

x_lfs = np.array([1.86, 2.24, 2.53, 2.90, 3.43, 4.28, 5.80, 6.70])
z_lfs = np.array([4.80, 5.38, 5.84, 6.24, 6.60, 6.76, 6.71, 6.71])
x_hfs = np.array([1.42, 1.06, 0.81, 0.67, 0.62, 0.62, 0.64, 0.60])
z_hfs = np.array([4.80, 5.09, 5.38, 5.72, 6.01, 6.65, 6.82, 7.34])

x_legs = np.concatenate([x_lfs, x_lfs, x_hfs, x_hfs])
z_legs = np.concatenate([z_lfs, -z_lfs, z_hfs, -z_hfs])

legs_isoflux = IsofluxConstraint(
    x_legs,
    z_legs,
    ref_x=x_lcfs[2],
    ref_z=z_lcfs[2],
    constraint_value=0.1,
    tolerance=np.mean(eq.psi(x_legs, z_legs)) * 1e-2,
)

magnetic_targets = MagneticConstraintSet([lcfs_isoflux, legs_isoflux])

# %% [markdown]

# ## Optimisation Algorithm, Conditions and Parameters
#
# We next define the Optimiser: the optimiser algorithm, conditions and
# parameters to be used. There is no one-size-fits-all approach here,
# as the best optimiser for a given problem will depend strongly on the
# optimisation objective and constraints (if any) being applied in the
# problem.
#
# `Optimiser` is currently only a wrapper for NLOpt based optimisers, and as
# such, the range of algorithms that may be chosen is determined by those
# available in the NLOpt library. Not all algorithms will work for all
# objectives - some require gradient information, for example - and
# some require additional parameters. Check the NLOpt API documentation
# for more details.
#
# Care must also be taken to ensure the termination criteria for the
# optimisation are suitable. Otherwise, the optimisation may never
# stop running if tolerances are too tight, or may stop early and
# return poorly optimised states if the maximum number of evaluations
# is too low or tolerances are too large.
#
# **Below we will directly input the optimisation algorithm, conditions and**
# **parameters when defining the `TikhonovCurrentCOP`**

# ## Iterators

# The `CoilsetOptimisationProblem` is only used to optimise the coilset state
# at fixed plasma psi;
# the Grad-Shafranov equation for the plasma is not guaranteed to still be
# satisfied
# after the coilset state is optimised.
#
# Picard iteration is therefore used to find a self-consistent solution,
# employed
# by `Iterator` objects. `PicardCoilsetIterator` specifies a scheme in which a
# Grad-Shafranov iteration is used to update the plasma psi alternate with
# Coilset
# optimisation at fixed plasma psi until the psi converges for the `Equilbrium`.
#
# ## Unconstrained Optimisation
#
# However, a poor initial `Equilibrium` (with corresponding `coilset`) will
# lead
# to difficulties during the Picard iteration used to find a self-consistent
# solution.
# It is therefore useful to perform a coarser fast pre-optimisation to try
# find a
# self-consistent state within the basin of convergence of the
# full constrained `OptimisationProblem`.

# %%

unconstrained_cop = UnconstrainedTikhonovCurrentGradientCOP(
    coilset, eq, magnetic_targets, gamma=1e-8
)
unconstrained_iterator = PicardIterator(
    eq,
    unconstrained_cop,
    fixed_coils=True,
    relaxation=0.3,
    convergence=DudsonConvergence(1e-6),
)

# %% [markdown]
#
# We have now initialised the necessary objects to perform the unconstrained
# optimisation of the `coilset` state.
#
# We first plot the initial `Equilibrium` state - as the initial coilset for
# this
# example has no current in the coils, it is predictably extremely poor!

# %%

f, ax = plt.subplots()
eq.plot(ax=ax)
unconstrained_cop.targets.plot(ax=ax)

# %% [markdown]
# Constrained optimisation of this poor initial state would be difficult, as
# local optimisers would likely struggle to find the basin of convergence of
# the desired state.
#
# We therefore perform a fast pre-optimisation to get closer to a physical
# state that satisfies our constrained optimisation problem.

# %%

unconstrained_iterator()

f, ax = plt.subplots()
eq.plot(ax=ax)
eq.coilset.plot(ax=ax, subcoil=False, label=True)
magnetic_targets.plot(ax=ax)
plt.show()

# %% [markdown]

# ## Constrained Optimisation
# We next define the list of `UpdateableConstraint` to apply.
#
# **Coil Field Constraints** are inequality constraints on the poloidal
# field at the middle of the inside edge of the coils, where the field
# is usually highest.
#
# A **Field Null Constraint** forces the poloidal field at a point to be zero.
#
# An `IsofluxConstraint` forces the flux at a set of points to be equal.
# %%
B_max = 11.5
field_constraints = CoilFieldConstraints(
    coilset=eq.coilset, B_max=B_max, tolerance=B_max * 1e-3
)

xp_idx = np.argmin(z_lcfs)
x_point = FieldNullConstraint(
    x_lcfs[xp_idx],
    z_lcfs[xp_idx],
    tolerance=1e-3,  # [T]
)

# %% [markdown]
# Now we have a better starting `Equilibrium` for our constrained optimisation
# We now initialise the TikhonovCurrentCOP and perform the optimisation:
#
# %%
opt_problem = TikhonovCurrentCOP(
    coilset=eq.coilset,
    eq=eq,
    targets=magnetic_targets,
    gamma=1e-8,
    opt_algorithm="COBYLA",
    opt_conditions={"max_eval": 400},
    opt_parameters={"initial_step": 0.01},
    max_currents=3.0e7,
    constraints=[field_constraints, x_point, lcfs_isoflux, legs_isoflux],
)
constrained_iterator = PicardIterator(
    eq,
    opt_problem,
    fixed_coils=True,
    relaxation=0.1,
    maxiter=100,
    convergence=DudsonConvergence(1e-4),
)

constrained_iterator()

f, ax = plt.subplots()
eq.plot(ax=ax)
eq.coilset.plot(ax=ax, subcoil=False, label=True)
magnetic_targets.plot(ax=ax)
plt.show()

# %% [markdown]
# You can now re-do this tutorial with other constraints of your choice.
# See `bluemira.equilibria.optimisation.constraints` for some prebuilt constraints.
