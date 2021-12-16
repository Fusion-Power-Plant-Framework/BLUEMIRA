# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                    J. Morris, D. Short
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
A geometry tutorial for users.
"""

# %% [markdown]

## Introduction

# Geometry is not plasma physics, but it isn't trivial either. Chances are most of
# your day-to-day interaction with bluemira will revolve around geometry in some form
# or another. Puns intended.

# There a few basic concepts you need to familiarise yourself with:
# * Basic objects: [`BluemiraWire`, `BluemiraFace`, `BluemiraShell`, `BluemiraSolid`]
# * Basic properties
# * Matryoshka structure
# * Geometry creation
# * Geometry modification
# * Geometry operations

## Imports

# Let's start out by importing all the basic objects, and some typical tools

# %%
from copy import deepcopy

# Basic objects
from bluemira.geometry.wire import BluemiraWire
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.shell import BluemiraShell
from bluemira.geometry.solid import BluemiraSolid

# Some useful tools
from bluemira.geometry.tools import (
    make_circle,
    make_polygon,
    make_bspline,
    revolve_shape,
    extrude_shape,
    sweep_shape,
    boolean_cut,
    boolean_fuse,
)

# Some existing parameterisations
from bluemira.geometry.parameterisations import PictureFrame, PrincetonD

# Some display functionality
from bluemira.display import show_cad, plot_2d
from bluemira.display.displayer import DisplayCADOptions

# %%[markdown]

## Make a cylinder

# There are many ways to make a cylinder, but perhaps the simplest way is as follows:
# * Make a circular Wire
# * Make a Face from that Wire
# * Extrude that Face along a vector, to make a Solid

# %%
# Note that we are going to give these geometries some labels, which
# we might use later.
circle_wire = make_circle(
    radius=5,
    center=(0, 0, 0),
    axis=(0, 0, 1),
    start_angle=0,
    end_angle=360,
    label="my_wire",
)
circle_face = BluemiraFace(circle_wire, label="my_face")
cylinder = extrude_shape(circle_face, vec=(0, 0, 10), label="my_solid")

# %%[markdown]

## Simple properties and representations

# %%
# Let's start off with some simple properties
print(f"Circle length: {circle_wire.length} m")
print(f"Circle area: {circle_face.area} m^2")
print(f"Cylinder volume: {cylinder.volume} m^3")

# You can also just print or repr these objects to get some useful info
print(cylinder)

# %%[markdown]

## Display

# %%
show_cad(cylinder, DisplayCADOptions(color="blue"))

# %%[markdown]

## Matryoshka structure

# Bluemira geometries are structured in a commonly used "Matryoska" or
# "Russian doll"-like structure.

# Solid -> Shell -> Face -> Wire

# These are accessible via the boundary attribute, so, in general, the boundary
# of a Solid is a Shell or set of Shells, and a Shell will have a set of Faces, etc.

# Let's take a little peek under the hood of our cylinder

# %%

print(f"Our cylinder is a BluemiraSolid: {isinstance(cylinder, BluemiraSolid)}")

i, j, k = 0, 0, 0  # This is just to facilitate comprehension
for i, shell in enumerate(cylinder.boundary):
    print(f"Shell: {i}.{j}.{k} is a BluemiraShell: {isinstance(shell, BluemiraShell)}")
    for j, face in enumerate(shell.boundary):
        print(f"Face: {i}.{j}.{k} is a BluemiraFace: {isinstance(face, BluemiraFace)}")
        for k, wire in enumerate(face.boundary):
            print(
                f"Wire: {i}.{j}.{k} is a BluemiraWire: {isinstance(wire, BluemiraWire)}"
            )

# OK, so a cylinder is pretty simple, but more complicated shapes
# will follow the same pattern.

# It does go deeper than this, but that is outside the intended
# user-realm.


# %%[markdown]

## Geometry creation

# Let's get familiar with some more ways of making geometries. We've
# looked at circle already, but what else is out there:
# * polygons
# * splines
# * a bit of everything

# %%

# Polygon
import numpy as np

theta = np.linspace(0, 2 * np.pi, 6)
x = 5 * np.cos(theta)
y = np.zeros(5)
z = 5 * np.sin(theta)

# TODO: transpose in API
points = np.array([x, y, z]).T
pentagon = make_polygon(points)

plot_2d(pentagon)

# %%
# Polygons are good for things with straight lines.
# Circles you've met already.
# For everything else, there's splines.

# Say you have a weird shape, that you might calculate via a equation.
# It's not a good idea to make a polygon with lots of very small sides
# for this. It's computationally expensive, and it will look ugly.


# %%[markdown]

## Modification of existing geometries

# Now we're going to look at some stuff that we can do to change
# geometries we've already made.

# %%
# TODO: Once fixed.. :'(
