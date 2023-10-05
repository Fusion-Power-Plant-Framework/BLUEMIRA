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
An example of how to generate materials
"""

# %%
from bluemira.materials import MaterialCache

# %% [markdown]
# # Material Definitions
#
# Materials play an important role in reactor design, be that via neutronics properties,
# structural properties, or thermal properties. This example gives an overview of how
# to define and use materials within bluemira.
#
# ## Defining Materials
#
# Materials will usually be defined via JSON files, such as the default definitions
# provided with bluemira (see data folder). Whe defining your own materials for
# your studies you can either take the JSON approach, or for preliminary analysis you can
# define materials using dictionaries as shown below.

# %%
material_dict = {
    "Strawberry": {
        # Pick a Material base class to use
        "material_class": "MassFractionMaterial",
        # Define the Material element mass fractions
        "elements": {"C": 0.4, "H": 0.4, "O": 0.2},
        # Define some material properties (fixed values or relations)
        "density": 1111,
        "poissons_ratio": 0.233,
        # Let's try an equation with some temperature bounds
        "youngs_modulus": {
            "value": "polynomial.Polynomial([1, 2e-5, 3e-7])(temperature_in_C)",
            "temp_min_celsius": 0.0,
            "temp_max_celsius": 100.0,
            "reference": "I made it up.",
        },
        # Let's try a linear interpolated property from some data
        "coefficient_thermal_expansion": {
            "value": (
                "interp(temperature_in_C, [20, 100, 200, 400], [10.3, 3.3, 2.2, 1.1])"
            ),
            "temp_min_celsius": 20.0,
            "temp_max_celsius": 400.0,
        },
    },
    "Cream": {
        "material_class": "MassFractionMaterial",
        "elements": {"C": 0.4, "H": 0.3, "O": 0.3},
        "density": 700,
        "poissons_ratio": 0.233,
    },
}

# Now we'll make a homogenised mixture (StrawberriesAndCream)
mixture_dict = {
    "StrawberriesAndCream": {
        # Define the mixing class
        "material_class": "HomogenisedMixture",
        # State which materials are to be used and in what volume fractions
        "materials": {"Strawberry": 0.7, "Cream": 0.3},
        # Set the temperature of the mixture
        "temperature_in_K": 290,
    },
}

# %% [markdown]
# ## The Materials Cache
#
# Bluemira materials are managed by a defining a `MaterialCache` instance. This allows
# material objects to be created from dictionaries or JSON files. Below gives an example
# of how to create materials in a cache for the dictionary definitions we have defined
# above.
#
# Note that the materials have to be loaded before the mixture is created - the cache
# handles mixing the materials that have already been created.

# %%
material_cache = MaterialCache()
material_cache.load_from_dict("Strawberry", material_dict)
material_cache.load_from_dict("Cream", material_dict)
material_cache.load_from_dict("StrawberriesAndCream", mixture_dict)

# %% [markdown]
# We can then get our materials from the cache as below. Note that the mixture
# represents the constituent materials as actual material objects, not strings.

# %%
strawberry = material_cache.get_material("Strawberry")
strawberries_and_cream = material_cache.get_material("StrawberriesAndCream")
another_strawberry = material_cache.get_material("Strawberry")
summer_time = material_cache.get_material("StrawberriesAndCream")

print(another_strawberry.elements)
print(summer_time.materials, summer_time.fracs)

# %% [markdown]
# It may be important to note that the default way to retrieve a material is through
# a cloning method - that means that you actually get back a new material that has the
# same attributes as the material in the cache, rather than the material that is stored
# in the cache. If you *really* want the material that is in the cache then you can set
# the keyword argument clone=False, but note that any changes made to that material will
# be reflected in the original object and future retrievals from the cache.

# %%
# Attributes are the same
print(another_strawberry == strawberry)
print(summer_time == strawberries_and_cream)

# But they are distinct objects
print(id(another_strawberry) == id(strawberry))
print(id(summer_time) == id(strawberries_and_cream))

# We can get the object as it is in the cache by setting clone=False
cached_strawberry = material_cache.get_material("Strawberry", clone=False)

# But beware that any changes to the cached object will affect the original object and
# any future retrievals.
print(cached_strawberry.temperature)
cached_strawberry.temperature = 273.15
print(another_strawberry == cached_strawberry)
print(material_cache.get_material("Strawberry") == another_strawberry)
