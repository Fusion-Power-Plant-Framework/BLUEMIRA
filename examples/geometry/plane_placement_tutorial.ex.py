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
Plane placement example
"""

# %% [markdown]
# # Plane placement examples
# A plane and a placement are two CAD primitives for orientation and positioning
# of an object.
# A placement describes the origin and orientation for a given object.
# A plane is a CAD primitive surface which is defined by its placement.
#
# Necessary imports

# %%
import numpy as np

from bluemira import display
from bluemira.geometry.plane import BluemiraPlane

# %% [markdown]
# Creation of a random plane

# %%
base = np.array([20, 0, 0])
axis = np.array([0, 1, 0])
plane = BluemiraPlane(base=base, axis=axis)
print(plane)

# %% [markdown]
# The plane can be converted into a BluemiraFace both for plotting purpose or to be
# used for the generation of components or in boolean operations.

# %%
face = plane.to_face(width=50, height=50)

options = display.plotter.PlotOptions()
options.view = "xz"
display.plot_2d(face, options)

# %% [markdown]
# A placement can be created from a plane

# %%
placement = plane.to_placement()

# %% [markdown]
# On the other side, default xy, yz, and xz plane can be extracted from a placement.
# Note: xy, yz, and xz plane are referred to the placement axes. Thus, in this
# particular case, xy plane lies on the GLOBAL xz plane.

# %%
xy_plane = placement.xy_plane()
yz_plane = placement.yz_plane()
xz_plane = placement.xz_plane()

face_xy = xy_plane.to_face(width=50, height=50)
face_yz = yz_plane.to_face(width=50, height=50)
face_xz = xz_plane.to_face(width=50, height=50)

options.face_options["color"] = "blue"
ax = display.plot_3d(face_xy, options, show=False)
options.face_options["color"] = "red"
ax = display.plot_3d(face_yz, options, ax=ax, show=False)
options.face_options["color"] = "green"
ax = display.plot_3d(face_xz, options, ax=ax, show=True)
