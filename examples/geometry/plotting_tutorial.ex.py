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
Plotting module examples
"""

# %%
import matplotlib.pyplot as plt

import bluemira.geometry.tools
from bluemira import display
from bluemira.base.components import Component, PhysicalComponent
from bluemira.display.plotter import FacePlotter, WirePlotter
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import PrincetonD

# %% [markdown]
# # Plotting examples
# ## Setup
#
# Creation of a closed wire and respective face and discretisation points.
#
# PrincetonD parameterisation is used as example.
#
# Note: the curve is generated on the x-z plane

# %%
p = PrincetonD()
p.adjust_variable("x1", 4, lower_bound=3, upper_bound=5)
p.adjust_variable("x2", 16, lower_bound=10, upper_bound=20)
p.adjust_variable("dz", 0, lower_bound=0, upper_bound=0)
wire = p.create_shape()
face = BluemiraFace(wire)

# %% [markdown]
# ## Default plotting
#
# We can plot the list of points, as well as the BluemiraWire and BluemiraFace
# in the following way, using the display built-in function with default settings

# %%
display.plot_2d(wire)
display.plot_3d(wire)
display.plot_2d(face)
display.plot_3d(face)

# %% [markdown]
#
# In a similar way, it is possible to use specific Plotters for each entity,
# i.e. PointsPlotter, WirePlotter, and FacePlotter. For example:

# %%
plotter_2d = WirePlotter()
plotter_2d.plot_2d(wire)

# %% [markdown]
#
# ## Modifying defaults
#
# Default plot options can be obtained in form of a dictionary instancing one of the
# plotters, e.g.:

# %%
my_options = display.plotter.get_default_options()
print(my_options)

# %% [markdown]
#
# Modifying the dictionary and passing it to a plot function will display the plot
# with the new options

# %%
my_options.wire_options = {"color": "red", "linewidth": "1.5"}
display.plot_2d(wire, my_options)

# %% [markdown]
#
# Alternatively, plot options can be modified directly inside a Plotter, e.g.:

# %%
plotter_2d.options.show_points = True
plotter_2d.options.ndiscr = 15
plotter_2d.plot_2d(wire)

# %% [markdown]
#
#
# Once you get familiar with the options, you can also make your own dictionaries, and
# pass them to the plotting functions

# %%
my_options = {
    "show_points": False,
    "wire_options": {"color": "red", "linewidth": 3, "linestyle": "dashed"},
}
display.plot_2d(wire, **my_options)

# %% [markdown]
#
# Being matplotlib the default plot library, points_options, wire_options,
# and face_options are equivalent to the **kwargs passed to the functions scatter,
# plot, and fill, respectively.
#
# Discretise the wire to an array of points.

# %%
points = wire.discretize(ndiscr=10, byedges=True)

# %% [markdown]
# ## Points Plot
#
# Simple plot of the obtained points.
#
# A 2D plot with the built-in display functions

# %%
display.plot_2d(
    points, point_options={"s": 30, "facecolors": "red", "edgecolors": "black"}
)
# or with a Plotter
# pplotter = PointsPlotter(
#     point_options={"s": 30, "facecolors": "red", "edgecolors": "black"}
# )
# pplotter.plot_2d(points)

# %% [markdown]
# ## 3D Scatter Plot
#
# A plot of the same points, but in 3D this time.

# %%
display.plot_3d(
    points, point_options={"s": 30, "facecolors": "red", "edgecolors": "black"}
)

# pplotter = Plotter3D(
#      point_options={"s": 30, "facecolors": "red", "edgecolors": "black"}
# )
# pplotter.plot_3d(points)

# %% [markdown]
# ## Wire Plot
#
# A WirePlotter is used with the default setup with:
#
# - view = xz (this is the projection plane, not a section plane)
# - point size = 20
# - ndiscr = 15

# %%
wplotter = WirePlotter(view="xz")
wplotter.options.point_options["s"] = 20
wplotter.options.ndiscr = 15
wplotter.plot_2d(wire)


# %% [markdown]
# ## 3D Curve Plot
#
# A plot of the same wire, but in 3D this time.

# %%
display.plot_3d(wire, **wplotter.options.as_dict())

# %% [markdown]
# ## Wire Plot with Matplotlib Default Options
#
# In this example point_options is set to an empty dict. The default matplotlib are used.

# %%
display.plot_2d(wire, point_options={})
# The plot is immediately shown by default, so it is not possible to act on the plot

# %% [markdown]
# ## Wire plot with some modifications
#
# In this example, we choose our own matplotlib Axes onto which to plot, disable the
# automatic display of the plot (show=False), and apply a title to the plot

# %%
f, ax = plt.subplots()
wplotter.options.point_options = {}
wplotter.plot_2d(wire, ax=ax, show=False)
ax.set_title("Wire plot")
plt.show()

# %% [markdown]
# ## Face Plot
#
# A FacePlotter is used with the default setup with:
#
# - view = xz (this is the projection plane, not a section plane)
# - ndiscr = 30
# - plot title

# %%
f, ax = plt.subplots()
fplotter = FacePlotter(view="xz")
fplotter.options.ndiscr = 30
fplotter.plot_2d(face, ax=ax, show=False)
ax.set_title("Face plot without points (default)")
plt.show()

# %% [markdown]
# ## Face Plot with Points
#
# We've set the points to be deactivate by default, but we can enable them again for
# individual plotters.

# %%
f, ax = plt.subplots()
fplotter2 = FacePlotter(view="xz")
fplotter2.options.ndiscr = 30
fplotter2.options.show_points = True
fplotter2.plot_2d(face, ax=ax, show=False)
ax.set_title("Face plot with points")
plt.show()

# %% [markdown]
# ## Face Plot with only Points
#
# Only show the wire of a face

# %%
f, ax = plt.subplots()
fplotter3 = FacePlotter(view="xz")
fplotter3.options.show_wires = True
fplotter3.options.show_faces = False
fplotter3.plot_2d(face, ax=ax, show=False)
ax.set_title("Face plot its wire")
plt.show()

# %% [markdown]
# ## Make a Second Face
#
# A second geometry is created, surrounding our original face.

# %%
p2 = PrincetonD()
p2.adjust_variable("x1", 3.5, lower_bound=3, upper_bound=5)
p2.adjust_variable("x2", 17, lower_bound=10, upper_bound=20)
p2.adjust_variable("dz", 0, lower_bound=0, upper_bound=0)
wire2 = p2.create_shape()
face2 = BluemiraFace(wire2)

# %% [markdown]
# ## Combined Face Plot
#
# Face and face2 are plotted using the same FacePlotter. Since no plot options have
# been changed, the two faces will be plotted in the same way (e.g. same color).

# %%
fplotter4 = FacePlotter(view="xz")
fplotter4.options.show_points = True
fplotter4.options.face_options = {"color": "blue"}

f, ax = plt.subplots()
fplotter4.plot_2d(face, ax=ax, show=False)
fplotter4.plot_2d(face2, ax=ax, show=False)
ax.set_title("Both faces in blue")
plt.show()
print(f"fplotter2.options: {fplotter2.options}")

# %% [markdown]
# ## Combined Face Plot with Different Colours
#
# Plot both face with different colour.
#
# Note: if face is plotted before face2, face2 will be "covered" by face.

# %%
f, ax = plt.subplots()
fplotter4.options.face_options = {"color": "blue"}
fplotter4.plot_2d(face2, ax=ax, show=False)
fplotter4.options.face_options = {"color": "green"}
fplotter4.plot_2d(face, ax=ax, show=False)
ax.set_title("Both faces with different colors")
plt.show()

# %% [markdown]
# ## Face with Hole
#
# A third face is create as difference between face and face2 (a BluemiraFace object
# has been created using wire2 as outer boundary and wire as inner boundary).
#
# Note:
# - when plotting points, it can happen that markers are not centred properly as
#       described in https://github.com/matplotlib/matplotlib/issues/11836
# - face3 is created with a wire deepcopy in order to be able to modify face and face2
# (and thus wire and wire2) without modifying face3

# %%
f, ax = plt.subplots()
face3 = BluemiraFace([wire2.deepcopy(), wire.deepcopy()])
fplotter5 = FacePlotter(view="xz")
fplotter5.options.show_points = True
ax = fplotter5.plot_2d(face3, ax=ax, show=False)
ax.set_title("Face with hole - points enabled")
plt.show()

f, ax = plt.subplots()
fplotter5.options.face_options["color"] = "blue"
fplotter5.options.show_points = False
fplotter5.plot_2d(face3, ax=ax, show=False)
ax.set_title("Face with hole - points disabled - blue")
plt.show()

# %% [markdown]
# ## Perform Some Face Operations
#
# Scale and move our face,
# if you run the above face plots again you can see they will change.

# %%
bari = face.center_of_mass
face.scale(0.5)
new_bari = face.center_of_mass
diff = bari - new_bari
v = (diff[0], diff[1], diff[2])
face.translate(v)


# %% [markdown]
# ## Wires and Faces
#
# Create and plot a couple of Wires and then create and plot the corresponding Faces.

# %%
points = [[0, 0, 0], [1, 0, 0], [1, 0, 3], [0, 0, 3]]
wire = bluemira.geometry.tools.make_polygon(points, closed=True)
wire1 = wire.deepcopy()
wire1.translate((3, 0, 5))
wplotter2 = WirePlotter(view="xz")
ax = wplotter2.plot_2d(wire, show=False)
ax = wplotter2.plot_2d(wire1, ax=ax, show=False)
ax.set_title("Two wires")
plt.show()


# %% [markdown]
# ## Plots with Matplotlib Default Point Options
#
# Plot the points on a boundary of a face with matplotlib defaults.
#
# Note that, since point_options = {}, points color is automatically changed by
# matplotlib.

# %%
wface = BluemiraFace(wire)
w1face = BluemiraFace(wire1)

# %% [markdown]
# ## PhysicalComponent Plot
#
# Creates a `PhysicalComponent` and plots it in the xz plane
#
# Note that if no face colour is set, a colour from the default palette will be chosen
# by default. This will not be the same every time.

# %%
pd_phycomp = PhysicalComponent("Comp", face)
pd_phycomp.plot_options.view = "xz"
pd_phycomp.plot_options.ndiscr = 30
ax = pd_phycomp.plot_2d(show=False)
ax.set_title("test component plot")
plt.show(block=True)

# %% [markdown]
# this time plots only the wire and not the face.
#
# Note that unlike the `FacePlotter` when `show_faces = False` the wire is
# shown by default.

# %%
pd_phycomp = PhysicalComponent("Comp", face)
pd_phycomp.plot_options.view = "xz"
pd_phycomp.plot_options.show_faces = False
ax = pd_phycomp.plot_2d(show=False)
ax.set_title("test component plot wire of face")
plt.show(block=True)

# %% [markdown]
# ## Component Plot
#
# Creates a `Component` and plots it in the xz plane using matplotlib defaults.
# Here we override some defaults and make our custom set of plot options.

# %%
group = Component("Components")
my_group_options = group.plot_options.as_dict()
my_group_options["wire_options"] = {}
my_group_options["face_options"] = {"color": "red"}
c1 = PhysicalComponent("Comp1", face, parent=group)
c2 = PhysicalComponent("Comp2", wface, parent=group)
c3 = PhysicalComponent("Comp3", w1face, parent=group)
display.plot_2d(group, **my_group_options)

# %% [markdown]
# Note that, since wire_options = {}, wire color is automatically changed by matplotlib
#
# ## Component and BluemiraGeo Combined Plot
#
# Plots a component on the same axes as a BluemiraFace.

# %%
wplotter3 = WirePlotter(view="xz")
wplotter3.options.point_options["s"] = 20
wplotter3.options.ndiscr = 15
wplotter3.options.wire_options["color"] = "red"
ax = wplotter3.plot_2d(wface.boundary[0], show=False)

fplotter6 = FacePlotter(view="xz")
fplotter6.options.show_wires = True
fplotter6.options.face_options["color"] = "green"
fplotter6.options.wire_options["color"] = "black"
ax = fplotter6.plot_2d(w1face, ax=ax, show=False)

ax = pd_phycomp.plot_2d(ax=ax, show=False)
ax.set_title("test component + BluemiraGeo plot")
plt.show(block=True)

# %% [markdown]
# Show the options from our combined plot

# %%
print(f"wire plotter options: {wplotter3.options}")
print(f"face plotter options: {fplotter6.options}")
print(f"component plotter options: {pd_phycomp.plot_options}")

# %% [markdown]
# ## CAD Display
#
# BluemiraWire and BluemiraFace can be displayed as CAD using the built-in display
# function:

# %%
display.show_cad(face)

# %% [markdown]
# For what concern Components, the component function show_cad is used.
#
# Note that if no colour is set, a colour from the default palette will be chosen
# by default. This will not be the same every time.
# %%
group.show_cad()

# %% [markdown]
# We can also change the appearance of individual components inside the group.
# Colours can be specified as an R-G-B tuple, string, or hex-string.

# %%
c1.display_cad_options.modify(color=(0.1, 0.2, 0.4))
c2.display_cad_options.modify(color="g")
c3.display_cad_options.modify(color="#FF3450", transparency=0.5)

group.show_cad()
