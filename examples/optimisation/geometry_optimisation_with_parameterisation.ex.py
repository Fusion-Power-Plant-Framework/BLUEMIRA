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

"""Geometry Optimisation With a Geometry Parameterisation"""

# %% [markdown]
# # Geometry Optimisation with a New Parameterisation

# %%
from dataclasses import dataclass
from typing import Optional

from bluemira.display import plot_2d
from bluemira.display.plotter import PlotOptions
from bluemira.geometry.optimisation import optimise_geometry
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.geometry.tools import make_circle, make_polygon
from bluemira.geometry.wire import BluemiraWire
from bluemira.utilities.opt_variables import OptVariable, OptVariablesFrame, VarDictT, ov


@dataclass
class CircleOptVariables(OptVariablesFrame):
    """Optimisation variables for a circle in the xz-plane."""

    radius: OptVariable = ov("radius", 10, 1e-5, 15)
    centre_x: OptVariable = ov("centre_x", 0, -10, 10)
    centre_z: OptVariable = ov("centre_z", 0, 0, 10)


class Circle(GeometryParameterisation):
    """Geometry parameterisation for a circle in the xz-plane."""

    def __init__(self, var_dict: Optional[VarDictT] = None):
        opt_variables = CircleOptVariables()
        opt_variables.adjust_variables(var_dict, strict_bounds=False)
        super().__init__(opt_variables)

    def create_shape(self, label: str = "") -> BluemiraWire:
        """Create the circle."""
        return make_circle(
            self.variables["radius"].value,
            center=(
                self.variables["centre_x"].value,
                0,
                self.variables["centre_z"].value,
            ),
            axis=(0, 1, 0),
            label=label,
        )


zone = make_polygon({"x": [-2, -2, 3, 3], "z": [0, 1, 1, 0]}, closed=True)

# Now lets create our circle within the shape
circle = Circle(
    {"radius": {"value": 10}, "centre_x": {"value": -2}, "centre_z": {"value": 1.5}}
)

plot_2d([circle.create_shape(), zone])


def objective(geom: GeometryParameterisation) -> float:
    """Objective function to minimise the perimeter of a circle."""
    return geom.create_shape().length


result = optimise_geometry(
    geom=circle,
    f_objective=objective,
    keep_out_zones=[{"wire": zone, "n_discr": 300, "tol": 1e-12}],
    algorithm="SLSQP",
    opt_conditions={"ftol_rel": 1e-8, "max_eval": 200},
)
print(result)
print(result.geom.variables)

plot_2d([result.geom.create_shape(), zone], PlotOptions(ndiscr=500))
