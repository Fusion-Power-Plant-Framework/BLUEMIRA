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
A quick tutorial on the optimisation of geometry in bluemira
"""

# %% [markdown]
# # Optimisation of Geometry

# %% [markdown]
# ## ⚠️ This method of defining optimisations is deprecated ⚠️
# See
# [examples/optimisation/geometry_optimisation](../optimisation/geometry_optimisation.ex.py)

# %% [markdown]
# ## Introduction
#
# We're going to set up some geometry optimisation problems and solve them with
# different optimisation algorithms.

# %%
import numpy as np

from bluemira.geometry.optimisation import GeometryOptimisationProblem, minimise_length
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.utilities.opt_problems import OptimisationConstraint, OptimisationObjective
from bluemira.utilities.optimiser import Optimiser, approx_derivative
from bluemira.utilities.tools import set_random_seed

# %% [markdown]
# Let's set up a simple GeometryOptimisationProblem, where we minimise the length of
# parameterised geometry.
#
# First, we set up the GeometryParameterisation, with some bounds on the variables.

# %%
parameterisation_1 = PrincetonD(
    {
        "x1": {"lower_bound": 2, "value": 4, "upper_bound": 6},
        "x2": {"lower_bound": 10, "value": 14, "upper_bound": 16},
        "dz": {"lower_bound": -0.5, "value": 0, "upper_bound": 0.5},
    }
)

# %% [markdown]
# Here we're minimising the length, and we can work out that the dz variable will not
# affect the optimisation, so let's just fix at some value and remove it from the problem

# %%
parameterisation_1.fix_variable("dz", value=0)

# %% [markdown]
#
# Now, we set up our optimiser. We'll start with a gradient-based optimisation algorithm

# %%
slsqp_optimiser = Optimiser(
    "SLSQP", opt_conditions={"max_eval": 100, "ftol_abs": 1e-12, "ftol_rel": 1e-12}
)

# %% [markdown]
#
# Next, we make our objective function, using in this case one of the ready-made ones.
#
# NOTE: This `minimise_length` function includes automatic numerical calculation of the
# objective function gradient, and expects a certain signature.

# %%
objective = OptimisationObjective(
    minimise_length,
    f_objective_args={"parameterisation": parameterisation_1},
)

# %% [markdown]
#
# Finally, we initialise our `GeometryOptimisationProblem` and run it.

# %%
my_problem = GeometryOptimisationProblem(parameterisation_1, slsqp_optimiser, objective)
my_problem.optimise()

# %% [markdown]
#
# Here we're minimising the length, within the bounds of our PrincetonD parameterisation,
# so we'd expect that x1 goes to its upper bound, and x2 goes to its lower bound.

# %%
print(
    f"x1: value: {parameterisation_1.variables['x1'].value}, upper_bound:"
    f" {parameterisation_1.variables['x1'].upper_bound}"
)
print(
    f"x2: value: {parameterisation_1.variables['x2'].value}, lower_bound:"
    f" {parameterisation_1.variables['x2'].lower_bound}"
)

# %% [markdown]
# Now let's do the same with an optimisation algorithm that doesn't require gradients.
# The `minimise_length` function will not calculate the gradients numerically if the
# optimisation algorithm does not require them.

# %%
parameterisation_2 = PrincetonD(
    {
        "x1": {"lower_bound": 2, "value": 4, "upper_bound": 6},
        "x2": {"lower_bound": 10, "value": 14, "upper_bound": 16},
        "dz": {"lower_bound": -0.5, "value": 0, "upper_bound": 0.5},
    }
)

cobyla_optimiser = Optimiser(
    "COBYLA",
    opt_conditions={
        "ftol_rel": 1e-3,
        "xtol_rel": 1e-12,
        "xtol_abs": 1e-12,
        "max_eval": 1000,
    },
)
objective = OptimisationObjective(
    minimise_length,
    f_objective_args={"parameterisation": parameterisation_2},
)
problem = GeometryOptimisationProblem(parameterisation_2, cobyla_optimiser, objective)
problem.optimise()

# %% [markdown]
# Again, let's check it's found the correct result:

# %%
print(
    f"x1: value: {parameterisation_2.variables['x1'].value}, upper_bound:"
    f" {parameterisation_2.variables['x1'].upper_bound}"
)
print(
    f"x2: value: {parameterisation_2.variables['x2'].value}, lower_bound:"
    f" {parameterisation_2.variables['x2'].lower_bound}"
)


# %% [markdown]
#
# Now let's include a relatively arbitrary constraint:
# We're going to minimise length again, but with a constraint that says that we don't
# want the length to be below some arbitrary value of 50.
# There are much better ways of doing this, but this is to demonstrate the use of an
# inequality constraint.


# %%
def calculate_constraint(vector, parameterisation, c_value):
    """
    Some arbitrary constraint function
    """
    parameterisation.variables.set_values_from_norm(vector)
    length = parameterisation.create_shape().length
    return np.array([c_value - length])


def my_constraint(constraint, vector, grad, parameterisation, c_value, ad_args=None):
    """
    Constraint function with numerical gradient calculation
    """
    value = calculate_constraint(vector, parameterisation, c_value)
    constraint[:] = value

    if grad.size > 0:
        grad[:] = approx_derivative(
            calculate_constraint,
            vector,
            f0=value,
            args=(parameterisation, c_value),
            **ad_args,
        )

    return constraint


c_value = 50
c_tolerance = 1e-6
constraint_function = OptimisationConstraint(
    my_constraint,
    f_constraint_args={"parameterisation": parameterisation_2, "c_value": c_value},
    tolerance=np.array([c_tolerance]),
)

# %% [markdown]
#
# Setting up the problem with a constraint is the same as before, but with an additional
# argument

# %%
parameterisation_3 = PrincetonD()
slsqp_optimiser2 = Optimiser(
    "SLSQP",
    opt_conditions={
        "ftol_rel": 1e-3,
        "xtol_rel": 1e-12,
        "xtol_abs": 1e-12,
        "max_eval": 1000,
    },
)
constraint_function = OptimisationConstraint(
    my_constraint,
    f_constraint_args={"parameterisation": parameterisation_3, "c_value": c_value},
    tolerance=np.array([c_tolerance]),
)
objective = OptimisationObjective(
    minimise_length,
    f_objective_args={"parameterisation": parameterisation_3},
)
problem = GeometryOptimisationProblem(
    parameterisation_3, slsqp_optimiser2, objective, constraints=[constraint_function]
)
problem.optimise()

# %% [markdown]
#
# Both x1 and x2 are free variables and between them they should be create a PrincetonD
# shape of length exactly 50 (as the bounds on these variables surely allow it).
# As we are minimising length, we'd expect to see a function value of 50 here (+/- the
# tolerances)... but we don't!

# %%
print(f"Theoretical optimum: {c_value-c_tolerance}")
print(f"Length with SLSQP: {parameterisation_3.create_shape().length}")
print(f"n_evals: {problem.opt.n_evals}")

# %% [markdown]
# This is because we're using numerical gradients and jacobians for our objective and
# inequality constraint functions. This can be faster than other approaches, but is less
# robust and also less likely to find the best solution.
#
# Let's try a few different optimisers, noting the different termination conditions we
# can play with and their effect.


# %%
parameterisation_4 = PrincetonD()
cobyla_optimiser2 = Optimiser(
    "COBYLA",
    opt_conditions={
        "ftol_rel": 1e-7,
        "xtol_rel": 1e-12,
        "xtol_abs": 1e-12,
        "max_eval": 1000,
    },
)
constraint_function = OptimisationConstraint(
    my_constraint,
    f_constraint_args={"parameterisation": parameterisation_4, "c_value": c_value},
    tolerance=np.array([c_tolerance]),
)
objective = OptimisationObjective(
    minimise_length,
    f_objective_args={"parameterisation": parameterisation_4},
)
problem = GeometryOptimisationProblem(
    parameterisation_4, cobyla_optimiser2, objective, constraints=[constraint_function]
)
problem.optimise()

print(f"Theoretical optimum: {c_value - c_tolerance}")
print(f"Length with COBYLA: {parameterisation_4.create_shape().length}")
print(f"n_evals: {problem.opt.n_evals}")


# %% [markdown]
#
# ISRES is a stochastic optimisation algorithm; if we want to see the same results every
# time, it's advisable to set the random seed to a known value.

# %%
set_random_seed(134365475)

parameterisation_5 = PrincetonD()
irses_optimiser = Optimiser(
    "ISRES",
    opt_conditions={
        "ftol_rel": 1e-12,
        "xtol_rel": 1e-12,
        "xtol_abs": 1e-12,
        "max_eval": 1000,
    },
)

objective = OptimisationObjective(
    minimise_length,
    f_objective_args={"parameterisation": parameterisation_5},
)
constraint_function = OptimisationConstraint(
    my_constraint,
    f_constraint_args={"parameterisation": parameterisation_5, "c_value": c_value},
    tolerance=np.array([c_tolerance]),
)
problem = GeometryOptimisationProblem(
    parameterisation_5, irses_optimiser, objective, constraints=[constraint_function]
)
problem.optimise()

print(f"Theoretical optimum: {c_value - c_tolerance}")
print(f"Length with ISRES: {parameterisation_5.create_shape().length}")
print(f"n_evals: {problem.opt.n_evals}")

# %% [markdown]
# Horses for courses folks... YMMV. Best thing you can do is specify your optimisation
# problem intelligently, using well-behaved objective and constraint functions, and smart
# bounds. Trying out different optimisers doesn't hurt. There's a trade-off between speed
# and accuracy. If you can't work out the analytical gradients, numerical gradients are a
# questionable approach, but can work well (fast) on some problems.
