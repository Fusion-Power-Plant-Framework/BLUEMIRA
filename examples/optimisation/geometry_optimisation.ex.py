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

"""Geometry Optimisation"""

# %% [markdown]
# # Geometry Optimisation
#
# In this example we will go through how to set up a simple geometry
# optimisation, including a geometric constraint.
#
# The problem to solve is, minimise the length of our wall boundary,
# in the xz-plane, whilst keeping it a minimum distance from our plasma.
#
# We will greatly simplify this problem by working with a circular
# plasma, we will use a PrincetonD for the wall shape,
# and set the minimum distance to half a meter.

# %% [markdown]
# ## Using the `optimise_geometry` Function
# Let's perform this geometry optimisation,
# utilising bluemira's `optimise_geometry` function.

# %%
from typing import List

import numpy as np

from bluemira.display import plot_2d
from bluemira.display.plotter import PlotOptions
from bluemira.geometry.optimisation import optimise_geometry
from bluemira.geometry.parameterisations import GeometryParameterisation, PrincetonD
from bluemira.geometry.tools import distance_to, make_circle
from bluemira.geometry.wire import BluemiraWire

min_distance = 0.5
plasma = make_circle(radius=2, center=(8, 0, 0.25), axis=(0, 1, 0))
# As with any optimisation, it's important to pick a reasonable initial
# parameterisation.
wall_boundary = PrincetonD(
    {
        "x1": {"value": 4, "upper_bound": 6},
        "x2": {"value": 12, "lower_bound": 10},
    }
)
print("Initial parameterisation:")
print(wall_boundary.variables)
print("Length of wall    :", wall_boundary.create_shape().length)
print("Distance to plasma:", distance_to(wall_boundary.create_shape(), plasma)[0])

plot_2d([wall_boundary.create_shape(), plasma])


# %%
def f_objective(geom: GeometryParameterisation) -> float:
    """Objective function to minimise a shape's length."""
    return geom.create_shape().length


def distance_constraint(
    geom: GeometryParameterisation, boundary: BluemiraWire, min_distance: float
) -> float:
    """
    A constraint to keep a minimum distance between two shapes.

    The constraint must be in the form f(x) <= 0, i.e., constraint
    is satisfied if f(x) <= 0.

    Since what we want is 'min_distance <= distance(A, B)', we rewrite
    this in the form 'min_distance - distance(A, B) <= 0', and return
    the left-hand side from this function.
    """
    shape = geom.create_shape()
    return min_distance - distance_to(shape, boundary)[0]


# %%
result = optimise_geometry(
    wall_boundary,
    algorithm="SLSQP",
    f_objective=f_objective,
    opt_conditions={"ftol_abs": 1e-6},
    keep_history=True,
    ineq_constraints=[
        {
            "f_constraint": lambda g: distance_constraint(g, plasma, min_distance),
            "tolerance": np.array([1e-8]),
        },
    ],
)

print("Optimised parameterisation:")
print(result.geom.variables)

boundary = result.geom.create_shape()
print("Length of wall    :", boundary.length)
print("Distance to plasma:", distance_to(boundary, plasma)[0])

plot_2d([boundary, plasma])

# %% [markdown]
# As we passed `keep_history=True` into the optimisation function, we
# can look at how the optimiser arrived at its solution.

# %%
geom = PrincetonD()
ax = plot_2d(plasma, show=False)
for i, (x, _) in enumerate(result.history):
    geom.variables.set_values_from_norm(x)
    wire = geom.create_shape()
    wire_options = {
        "alpha": 0.5 + ((i + 1) / len(result.history)) / 2,
        "color": "red",
        "linewidth": 0.1,
    }
    ax = plot_2d(wire, options=PlotOptions(wire_options=wire_options), ax=ax, show=False)
plot_2d(boundary, ax=ax, show=True)

# %% [markdown]
# ## Using the `GeomOptimisationProblem` Class
# Alternatively, we can take a class-based approach to defining this
# optimisation problem, using the `GeomOptimisationProblem` base class.

# %%

from bluemira.geometry.optimisation import GeomOptimisationProblem
from bluemira.geometry.optimisation.typing import GeomConstraintT


class ContractLengthGOP(GeomOptimisationProblem):
    """Geometry optimisation problem to minimise a shape's length."""

    def __init__(self, plasma: BluemiraWire, min_distance: float):
        self.plasma = plasma
        self.min_distance = min_distance

    def objective(self, geom: GeometryParameterisation) -> float:  # noqa: PLR6301
        """Objective function to minimise."""
        return geom.create_shape().length

    def ineq_constraints(self) -> List[GeomConstraintT]:
        """List of inequality constraints to satisfy."""
        return [
            {
                "f_constraint": lambda geom: self._distance_constraint(
                    geom, self.plasma, self.min_distance
                ),
                "tolerance": np.array([1e-8]),
            }
        ]

    @staticmethod
    def _distance_constraint(
        geom: GeometryParameterisation, boundary: BluemiraWire, min_distance: float
    ) -> float:
        """A constraint to keep a minimum distance between two shapes."""
        shape = geom.create_shape()
        return min_distance - distance_to(shape, boundary)[0]


wall_boundary = PrincetonD(
    {"x1": {"value": 4, "upper_bound": 6}, "x2": {"value": 12, "lower_bound": 10}}
)
opt_problem = ContractLengthGOP(plasma, min_distance)
result = opt_problem.optimise(
    wall_boundary, algorithm="SLSQP", opt_conditions={"ftol_abs": 1e-6}
)
print("Optimised parameterisation:")
print(result.geom.variables)

boundary = result.geom.create_shape()
print("Length of wall    :", boundary.length)
print("Distance to plasma:", distance_to(boundary, plasma)[0])
