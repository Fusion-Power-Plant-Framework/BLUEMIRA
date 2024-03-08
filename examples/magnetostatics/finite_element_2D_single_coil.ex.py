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
Application of the dolfin fem 2D magnetostatic to a single coil problem
"""

# %% [markdown]
# # 2-D FEM magnetostatic single coil
# ## Introduction
#
# In this example, we will show how to use the fem_magnetostatic_2D solver to find the
# magnetic field generated by a simple coil. The coil axis is the z-axis. Solution is
# calculated on the xz plane.
#
# ## Imports
#
# Import necessary module definitions.

# %%
from pathlib import Path

import gmsh
import matplotlib.pyplot as plt
import numpy as np
import pyvista
from dolfinx.io import XDMFFile
from dolfinx.plot import vtk_mesh
from matplotlib.axes import Axes
from mpi4py import MPI

from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.file import get_bluemira_path
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_polygon
from bluemira.geometry.wire import BluemiraWire
from bluemira.magnetostatics import greens
from bluemira.magnetostatics.biot_savart import Bz_coil_axis
from bluemira.magnetostatics.fem_utils import (
    Association,
    create_j_function,
    model_to_mesh,
    pyvista_plot_show_save,
)
from bluemira.magnetostatics.finite_element_2d import FemMagnetostatic2d
from bluemira.mesh import meshing

# %% [markdown]
# Creating coil
#
# Inner rectangular surface of singular wire and extent of coil
# %%

I_wire = 1e6  # wire's current

r_enclo = 30  # radius of enclosure
lcar_enclo = 2  # Characteristic length of enclosure
lcar_axis = lcar_enclo / 20  # axis characteristic length

rc = 5  # Outer radius wire
drc = 0.01  # Inner radius of wire
lcar_coil = 0.01  # Characteristic length of coil


poly_coil = make_polygon(
    [
        [rc - drc, rc + drc, rc + drc, rc - drc],
        [-drc, -drc, +drc, +drc],
        np.zeros(4),
    ],
    closed=True,
    label="poly_enclo",
)

poly_coil.mesh_options = {"lcar": lcar_coil, "physical_group": "poly_coil"}
coil = BluemiraFace(poly_coil)
coil.mesh_options = {"lcar": lcar_coil, "physical_group": "coil"}

poly_axis = make_polygon([np.zeros(3), [-r_enclo, 0, r_enclo], np.zeros(3)])
poly_axis.mesh_options = {"lcar": lcar_axis, "physical_group": "poly_axis"}

poly_ext = make_polygon(
    [
        [0, r_enclo, r_enclo, 0],
        [r_enclo, r_enclo, -r_enclo, -r_enclo],
        np.zeros(4),
    ],
    label="poly_ext",
)
poly_ext.mesh_options = {"lcar": lcar_enclo, "physical_group": "poly_ext"}
poly_enclo = BluemiraWire([poly_axis, poly_ext], "poly_enclo")
poly_enclo.mesh_options = {"lcar": lcar_enclo, "physical_group": "poly_enclo"}
enclosure = BluemiraFace([poly_enclo, poly_coil])
enclosure.mesh_options = {"lcar": lcar_enclo, "physical_group": "enclo"}

# %% [markdown]
# Creating external enclosure shape
#  ___
# |_  |
#  _| |
# |___|
#
# %%
r_enclo1 = 150
lcar_enclo1 = 10

poly_ext1 = make_polygon(
    [
        [0, r_enclo, r_enclo, 0, 0, r_enclo1, r_enclo1, 0],
        [r_enclo, r_enclo, -r_enclo, -r_enclo, -r_enclo1, -r_enclo1, r_enclo1, r_enclo1],
        np.zeros(8),
    ],
    label="poly_ext1",
    closed=True,
)
poly_ext1.mesh_options = {"lcar": lcar_enclo1, "physical_group": "poly_ext1"}
poly_enclo1 = BluemiraWire([poly_ext1], "poly_enclo1")
poly_enclo1.mesh_options = {"lcar": lcar_enclo1, "physical_group": "poly_enclo1"}
enclosure_ext = BluemiraFace([poly_enclo1])
enclosure_ext.mesh_options = {"lcar": lcar_enclo1, "physical_group": "enclo1"}

c_universe = Component(name="universe")
c_enclo = PhysicalComponent(name="enclosure", shape=enclosure, parent=c_universe)
c_enclo_ext = PhysicalComponent(
    name="enclosure_Ext", shape=enclosure_ext, parent=c_universe
)
c_coil = PhysicalComponent(name="coil", shape=coil, parent=c_universe)

# %% [markdown]
#
# ## Mesh
#
# Create the mesh (by default, mesh is stored in the file Mesh.msh")

# %%
gdim = 2  # Geometric dimension of the mesh

directory = get_bluemira_path("", subfolder="generated_data")
meshfiles = [Path(directory, p).as_posix() for p in ["Mesh.geo_unrolled", "Mesh.msh"]]

meshing.Mesh(meshfile=meshfiles)(c_universe, dim=gdim)

(mesh, ct, ft), labels = model_to_mesh(gmsh.model, gdim=gdim)
gmsh.write("Mesh.msh")
gmsh.finalize()

with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ft, mesh.geometry)
    xdmf.write_meshtags(ct, mesh.geometry)

with pyvista_plot_show_save("cell_tags.png") as plotter:
    grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, mesh.topology.dim))
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    grid.cell_data["Marker"] = ct.values[ct.indices < num_local_cells]
    grid.set_active_scalars("Marker")
    actor = plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()

# %%
em_solver = FemMagnetostatic2d(2)
em_solver.set_mesh(mesh, ct)

# %% [markdown]
#
# Define source term (coil current distribution) for the fem problem

# %%
coil_tag = labels["coil"][1]
jtot = create_j_function(mesh, ct, [Association(1, coil_tag, I_wire)])

# %% [markdown]
#
# solve the em problem and calculate the magnetic field B

# %%
em_solver.define_g(jtot)
em_solver.solve()

# %% [markdown]
#
# Compare the obtained B with both the theoretical value
#
# 1) Along the z axis (analytical solution)

# %%
# Comparison of the theoretical and calculated magnetic field (B).
# Note: The comparison is conducted along the z-axis, where an
# analytical expression is available. However, due to challenges
# in calculating the gradient of dPsi/dx along the axis for CG
# element, the points are translated by a value of r_offset.
r_offset = 0.25

z_points_axis = np.linspace(0, r_enclo / 2, 200)
r_points_axis = np.full(z_points_axis.shape, r_offset)
b_points = Coordinates({"x": r_points_axis, "y": z_points_axis}).xyz.T

Bz_axis = em_solver.calculate_b()(b_points)
Bz_axis = Bz_axis[:, 1]
bz_points = b_points[:, 1]
B_z_teo = Bz_coil_axis(rc, 0, bz_points, I_wire)

ax: Axes
_, ax = plt.subplots()
ax.plot(bz_points, Bz_axis, label="B_calc")
ax.plot(bz_points, B_z_teo, label="B_teo")
ax.set_xlabel("r (m)")
ax.set_ylabel("B (T)")
ax.legend()
plt.show()

_, ax = plt.subplots()
ax.plot(bz_points, Bz_axis - B_z_teo, label="B_calc - B_teo")
ax.set_xlabel("r (m)")
ax.set_ylabel("error (T)")
ax.legend()
plt.show()

# I just set an absolute tolerance for the comparison (since the magnetic field
# goes to zero, the comparison cannot be made on the basis of a relative
# tolerance). An allclose comparison was out of discussion considering the
# necessary accuracy.
np.testing.assert_allclose(Bz_axis, B_z_teo, atol=2.5e-4)

# %% [markdown]
#
# 2) Along a radial path at z_offset (solution from green function)

# %%
z_offset = 100 * drc

points_x = np.linspace(0, r_enclo, 200)
points_z = np.full(z_points_axis.shape, z_offset)

new_points = Coordinates({"x": points_x, "y": points_z}).xyz.T[1:]
Bx_fem, Bz_fem = em_solver.calculate_b()(new_points).T

g_psi, g_bx, g_bz = greens.greens_all(rc, 0, new_points[:, 0], new_points[:, 1])
g_psi *= I_wire
g_bx *= I_wire
g_bz *= I_wire

# %% [markdown]
#
# Finally plot the result comparison @ z = z_offset

# %%
_, ax = plt.subplots()
ax.plot(new_points[:, 0], Bx_fem, label="Bx_fem")
ax.plot(new_points[:, 0], g_bx, label="Green Bx")
ax.set_xlabel("r (m)")
ax.set_ylabel("Bx (T)")
ax.legend()
plt.show()

_, ax = plt.subplots()
ax.plot(new_points[:, 0], Bz_fem, label="Bz_fem")
ax.plot(new_points[:, 0], g_bz, label="Green Bz")
ax.set_xlabel("r (m)")
ax.set_ylabel("Bz (T)")
ax.legend()
plt.show()

_, ax = plt.subplots()
ax.plot(new_points[:, 0], Bx_fem - g_bx, label="Bx_calc - GreenBx")
ax.plot(new_points[:, 0], Bz_fem - g_bz, label="Bz_calc - GreenBz")
ax.legend()
ax.set_xlabel("r (m)")
ax.set_ylabel("error (T)")
plt.show()

np.testing.assert_allclose(Bx_fem, g_bx, atol=3e-4)
np.testing.assert_allclose(Bz_fem, g_bz, atol=6e-4)
