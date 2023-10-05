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
A geometry tutorial for users.
"""

# %% [markdown]
# # Geometry Tutorial
# ## Introduction
#
# Geometry is not plasma physics, but it isn't trivial either. Chances are most of
# your day-to-day interaction with bluemira will revolve around geometry in some form
# or another. Puns intended.
#
# There a few basic concepts you need to familiarise yourself with:
# * Basic objects: [`BluemiraWire`, `BluemiraFace`, `BluemiraShell`, `BluemiraSolid`]
# * Basic properties
# * Matryoshka structure
# * Geometry creation
# * Geometry modification
# * Geometry operations
#
# ## Imports
#
# Let's start out by importing all the basic objects, and some typical tools

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.file import get_bluemira_path

# Some display functionality
from bluemira.display import plotter, show_cad
from bluemira.display.displayer import DisplayCADOptions

# Basic objects
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.shell import BluemiraShell
from bluemira.geometry.solid import BluemiraSolid

# Some useful tools
from bluemira.geometry.tools import (
    boolean_cut,
    boolean_fuse,
    extrude_shape,
    interpolate_bspline,
    make_circle,
    make_polygon,
    revolve_shape,
    save_cad,
    sweep_shape,
)
from bluemira.geometry.wire import BluemiraWire

# %% [markdown]
# ## Geometry creation (1-D)
#
# Let's get familiar with some ways of making 1-D geometries.
# Bluemira implements functions for the creation of:
# * polygons
# * splines
# * arcs
# * a bit of everything (check geometry.tools module for an extensive list)
#
# Any 1-D geometry is stored in a BluemiraWire object. Just as example, we can start
# considering a simple linear segmented wire with vertexes on
# (0,0,0), (1,0,0), and (1,1,0).

# %%
points1 = Coordinates({"x": [0, 1, 1], "y": [0, 0, 1], "z": [0, 0, 0]})
first_wire = make_polygon(points1, label="wire1")

# A print of the object will return some useful info
print(first_wire)

# however, each information can be accessed through the respective
# obj property, e.g.
print(f"Wire length: {first_wire.length}")

# %% [markdown]
# Concatenation of more wires is also allowed:

# %%
points2 = Coordinates({"x": [1, 2], "y": [1, 2], "z": [0, 0]})
second_wire = make_polygon(points2, label="wire2")
full_wire = BluemiraWire([first_wire, second_wire], label="full_wire")
print(full_wire)

# %% [markdown]
# In such a case, sub-wires are still accessible as separate entities and
# can be returned through a search operation on the full wire:

# %%
first_wire1 = full_wire.search("wire1")[0]
print(
    f"first_wire and first_wire1 have the same shape: {first_wire1.is_same(first_wire)}"
)

# Simple plot
wire_plotter = plotter.WirePlotter()
wire_plotter.options.view = "xy"
wire_plotter.plot_2d(full_wire)

# %% [markdown]
# More complex geometries can be created using splines, arcs, etc.

# %%
wires = [
    make_polygon([[0, 3], [0, 0], [0, 0]], label="w1"),
    make_circle(1, (3, 1, 0), 270, 360, label="c2"),
    make_polygon([[4, 4], [1, 3], [0, 0]], label="w3"),
    make_circle(1, (3, 3, 0), 0, 90, label="c4"),
    make_polygon([[3, 0], [4, 4], [0, 0]], label="w5"),
    make_polygon([[0, 0], [4, 0], [0, 0]], label="w6"),
]
closed_wire = BluemiraWire(wires, label="closed_wire")
wire_plotter.plot_2d(closed_wire)

# %% [markdown]
# In such a case, the created wire is closed. A check can be done interrogating
# the is_closed function of the wire:

# %%
print(f"wire is closed: {closed_wire.is_closed()}")

# %% [markdown]
# ## Geometry creation (2-D and 3-D)
#
# A closed planar 1-D geometry can be used as boundary to generate a 2-D face.

# %%
first_face = BluemiraFace(boundary=closed_wire, label="first_face")
print(first_face)

# %% [markdown]
# A matplotlib-style plotting of a face can be made similarly to what was done for
# a wire, i.e. using a FacePlotter

# %%
face_plotter = plotter.FacePlotter()
face_plotter.options.view = "xy"
face_plotter.plot_2d(first_face)


# %% [markdown]
# If more than one closed wire is given as boundary for a face, the first one is
# used as the external boundary and subsequent ones are considered as holes.

# %%
points = Coordinates({"x": [1, 2, 2, 1], "y": [1, 1, 2, 2]})
hole = make_polygon(points, label="hole", closed=True)
face_with_hole = BluemiraFace(boundary=[closed_wire, hole], label="face_with_hole")
print(face_with_hole)
face_plotter.plot_2d(face_with_hole)


# %% [markdown]
# Starting from 1-D or 2-D geometries, 3-D objects can be created, for example,
# by revolution or extrusion.

# %%
first_solid = extrude_shape(face_with_hole, (0, 0, 1), "first_solid")
print(first_solid)

# Note: 3-D operations generate solids that are disconnected from the primitive shape.
# For this reason, it is not possible to retrieve our initial "face_with_hole"
# interrogating "fist_solid".


# %% [markdown]
# ## 3-D Display
#
# Geometry objects can be displayed via `show_cad`, and the appearance
# of said objects customised by specifying `color` and `transparency`.

# %%
show_cad(first_solid, DisplayCADOptions(color="blue", transparency=0.1))


# %% [markdown]
# ## Matryoshka structure
#
# Bluemira geometries are structured in a commonly used "Matryoshka" or
# "Russian doll"-like structure.
#
# Solid -> Shell -> Face -> Wire
#
# These are accessible via the boundary attribute, so, in general, the boundary
# of a Solid is a Shell or set of Shells, and a Shell will have a set of Faces, etc.
#
# Let's take a little peek under the hood of our solid:

# %%
print(f"Our shape is a BluemiraSolid: {isinstance(first_solid, BluemiraSolid)}")

i, j, k = 0, 0, 0  # This is just to facilitate comprehension
for i, shell in enumerate(first_solid.boundary):
    print(f"Shell: {i}.{j}.{k} is a BluemiraShell: {isinstance(shell, BluemiraShell)}")
    for j, face in enumerate(shell.boundary):
        print(f"Face: {i}.{j}.{k} is a BluemiraFace: {isinstance(face, BluemiraFace)}")
        for k, wire in enumerate(face.boundary):
            print(
                f"Wire: {i}.{j}.{k} is a BluemiraWire: {isinstance(wire, BluemiraWire)}"
            )


# %% [markdown]
#
# ## Geometric transformations
#
# When applying a geometric transformation to a BluemiraGeo object, that operation
# is transferred also to the boundary objects (in a recursive manner). That allows
# consistency between the object shape and its boundary without recreating
# the boundary set.
#
# Just as example, we are going to apply a translation to our "face_with_hole".

# %%
# To have a reference to the initial object, we make a deepcopy of the face
face_with_hole_copy = face_with_hole.deepcopy("face_copy")

# Now we apply the translation
face_with_hole.translate((6, 1, 0))

# and plot the face before and after the transformation (the translated face
# is plotted in red)
ax = face_plotter.plot_2d(face_with_hole_copy, show=False)
face_plotter.options.face_options["color"] = "red"
face_plotter.plot_2d(face_with_hole, ax=ax, show=False)
plt.title("Translated wire")
plt.show()

# The same happens, for example, to the wire that identifies the hole.
hole_copy = face_with_hole_copy.search("hole")[0]
wire_plotter.options.wire_options["color"] = "black"
ax = wire_plotter.plot_2d(hole_copy, show=False)
wire_plotter.options.wire_options["color"] = "red"
wire_plotter.plot_2d(hole, ax=ax, show=False)
plt.title("Translated wire test")
plt.show()


# %% [markdown]
# ## Geometry creation (complex shapes)
#
# OK, let's do something more complicated now.
#
# Polygons are good for things with straight lines.
# Arcs you've met already.
# For everything else, there's splines.
#
# Say you have a weird shape, that you might calculate via a equation.
# It's not a good idea to make a polygon with lots of very small sides
# for this. It's computationally expensive, and it will look ugly.

# %%
# Spline

x = np.linspace(0, 10, 1000)
y = 0.5 * np.sin(x) + 3 * np.cos(x) ** 2
z = np.zeros(1000)

points = np.array([x, y, z])
spline = interpolate_bspline(points)
points = np.array([x, y + 3, z])
polygon = make_polygon(points)

show_cad(
    [spline, polygon], [DisplayCADOptions(color="blue"), DisplayCADOptions(color="red")]
)

# %%
# To get an idea of why polygons are bad / slow / ugly, try:
vector = (0, 0, 1)
show_cad(
    [extrude_shape(spline, vector), extrude_shape(polygon, vector)],
    [DisplayCADOptions(color="blue"), DisplayCADOptions(color="red")],
)


# %% [markdown]
# ## Additional examples
# Making 3-D shapes from 2-D shapes
#
# You can:
# * extrude a shape `extrude_shape`, as we did with our solid
# * revolve a shape `revolve_shape`
# * sweep a shape `sweep_shape`

# %%
# Make a hollow cylinder, by revolving a rectangle
points = np.array([[4, 5, 5, 4], [0, 0, 0, 0], [2, 2, 3, 3]])
rectangle = BluemiraFace(make_polygon(points, closed=True))

hollow_cylinder = revolve_shape(
    rectangle, base=(0, 0, 0), direction=(0, 0, 1), degree=360
)

show_cad(hollow_cylinder)

# %%
# Sweep a profile along a path

points = np.array([[4.5, 4.5], [0, 3], [2.5, 2.5]])
straight_line = make_polygon(points)
quarter_turn = make_circle(center=(3, 3, 2.5), axis=(0, 0, 1), radius=1.5, end_angle=90)
path = BluemiraWire([straight_line, quarter_turn])
solid = sweep_shape(rectangle.boundary[0], path)
show_cad(solid)

# %% [markdown]
# Making 3-D shapes from 3-D shapes
#
# Boolean operations often come in very useful when making CAD.
# * You can join geometries together with `boolean_fuse`
# * You can cut geometries from one another with `boolean_cut`

# %%
points = np.array(
    [
        [0, 2, 2, 0],
        [0, 0, 0, 0],
        [0, 0, 3, 3],
    ]
)

box_1 = BluemiraFace(make_polygon(points, closed=True))
box_1 = extrude_shape(box_1, (0, 2, 0))

points = np.array(
    [
        [1, 3, 3, 1],
        [0, 0, 0, 0],
        [0, 0, 2, 2],
    ]
)

box_2 = BluemiraFace(make_polygon(points, closed=True))
box_2 = extrude_shape(box_2, (0, 1, 0))

fused_boxes = boolean_fuse([box_1, box_2])

show_cad(fused_boxes)

cut_box_1 = boolean_cut(box_1, box_2)[0]

show_cad(cut_box_1)

# %% [markdown]
# ## Modification of existing geometries
#
# Now we're going to look at some stuff that we can do to change
# geometries we've already made.
# * Rotate
# * Translate
# * Scale

# %%
# Let's save a deepcopy of a shape before modifying
new_cut_box_1 = cut_box_1.deepcopy()

new_cut_box_1.rotate(base=(0, 0, 0), direction=(0, 1, 0), degree=45)
new_cut_box_1.translate((0, 3, 0))
new_cut_box_1.scale(3)
blue_red_options = [DisplayCADOptions(color="blue"), DisplayCADOptions(color="red")]
show_cad([cut_box_1, new_cut_box_1], options=blue_red_options)

# %% [markdown]
# ## Exporting geometry
#
# Many different CAD file types can be written,
# for a full list see the `CADFileType` class.

# %%
# Try saving any shape or group of shapes created above
# as a STEP assembly

my_shapes = [cut_box_1]
# Modify this file path to where you want to save the data.
my_file_path = "my_tutorial_assembly.STP"
save_cad(
    my_shapes,
    filename=Path(
        get_bluemira_path("", subfolder="generated_data"), my_file_path
    ).as_posix(),
    unit_scale="metre",
)
