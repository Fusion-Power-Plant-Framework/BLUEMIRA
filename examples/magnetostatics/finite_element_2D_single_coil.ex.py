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


"""Example of the magnetic field generated by a circular coil"""

import gmsh
import matplotlib.pyplot as plt
import numpy as np
import pyvista
from dolfinx import geometry
from dolfinx.io import XDMFFile
from dolfinx.io.gmshio import model_to_mesh
from dolfinx.plot import create_vtk_mesh
from mpi4py import MPI

from bluemira.base.components import Component, PhysicalComponent
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_polygon
from bluemira.geometry.wire import BluemiraWire
from bluemira.magnetostatics.fem_utils import plot_scalar_field

rank = MPI.COMM_WORLD.rank

ri = 0.01  # Inner radius of copper wire
rc = 3  # Outer radius of copper wire
R = 100  # Radius of domain
I_wire = 10e6  # wire's current
gdim = 2  # Geometric dimension of the mesh
model_rank = 0
mesh_comm = MPI.COMM_WORLD

# Define geometry for wire cylinder
nwire = 20  # number of wire divisions
lwire = 0.1  # mesh characteristic length for each segment

nenclo = 20  # number of external enclosure divisions
lenclo = 1  # mesh characteristic length for each segment

# enclosure
theta_encl = np.linspace(np.pi / 2, -np.pi / 2, nenclo)
r_encl = R * np.cos(theta_encl)
z_encl = R * np.sin(theta_encl)

enclosure_points = [[0, 0, 0]]  # adding (0,0) to improve mesh quality
for ii in range(len(r_encl)):
    enclosure_point = [r_encl[ii], z_encl[ii], 0]
    enclosure_points.append(enclosure_point)

poly_enclo1 = make_polygon(enclosure_points[0:2])
poly_enclo1.mesh_options = {"lcar": 0.1, "physical_group": "poly_enclo1"}
poly_enclo2 = make_polygon(enclosure_points[1:])
poly_enclo2.mesh_options = {"lcar": 1, "physical_group": "poly_enclo2"}
poly_enclo = BluemiraWire([poly_enclo1, poly_enclo2])
poly_enclo.close("poly_enclo")


# coil
theta_coil = np.linspace(0, 2 * np.pi, nwire)
r_coil = rc + ri * np.cos(theta_coil[:-1])
z_coil = ri * np.sin(theta_coil)

coil_points = []
for ii in range(len(r_coil)):
    coil_point = [r_coil[ii], z_coil[ii], 0]
    coil_points.append(coil_point)

poly_coil = make_polygon(coil_points, closed=True)
lcar_coil = np.ones([poly_coil.vertexes.shape[1], 1]) * lwire
poly_coil.mesh_options = {"lcar": 0.01, "physical_group": "poly_coil"}

coil = BluemiraFace([poly_coil])
coil.mesh_options.physical_group = "coil"

enclosure = BluemiraFace([poly_enclo, poly_coil])
enclosure.mesh_options.physical_group = "enclo"

c_universe = Component(name="universe")

c_enclo = PhysicalComponent(name="enclosure", shape=enclosure, parent=c_universe)
c_coil = PhysicalComponent(name="coil", shape=coil, parent=c_universe)


# %% [markdown]
#
# ## Mesh
#
# Create the mesh (by default, mesh is stored in the file Mesh.msh")

# %%
from pathlib import Path

from bluemira.base.file import get_bluemira_path
from bluemira.mesh import meshing

directory = get_bluemira_path("", subfolder="generated_data")
meshfiles = [Path(directory, p).as_posix() for p in ["Mesh.geo_unrolled", "Mesh.msh"]]

meshing.Mesh(meshfile=meshfiles)(c_universe, dim=2)

mesh, ct, ft = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)
gmsh.write("Mesh.msh")
gmsh.finalize()

with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ct)

pyvista.start_xvfb()

pyvista.OFF_SCREEN = True
plotter = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(*create_vtk_mesh(mesh, mesh.topology.dim))
num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
grid.cell_data["Marker"] = ct.values[ct.indices < num_local_cells]
grid.set_active_scalars("Marker")
actor = plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()

if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    cell_tag_fig = plotter.screenshot("cell_tags.png")


from bluemira.magnetostatics.fem_utils import calculate_area, create_j_function
from bluemira.magnetostatics.finite_element_2d import FemMagnetostatic2d

coil_tag = 5

em_solver = FemMagnetostatic2d(mesh, ct, ("CG", 2))
j_wire = I_wire / calculate_area(mesh, ct, coil_tag)
j_wire = create_j_function(mesh, ct, [(j_wire, coil_tag)])
em_solver.solve(j_wire)

plotter = pyvista.Plotter()
V = em_solver._V
psi = em_solver.psi

B = em_solver.compute_B(("CG", 1))
W = B.function_space

tol = 0.001  # Avoid hitting the outside of the domain
num_points = 5001
z_min = 0
z_max = 10
z = np.linspace(z_min, z_max, num_points)
points = np.zeros((3, num_points))
points[1] = z
Psi_values = []
B_values = []

bb_tree = geometry.BoundingBoxTree(mesh, mesh.topology.dim)
cells = []
points_on_proc = []
# Find cells whose bounding-box collide with the points
cell_candidates = geometry.compute_collisions(bb_tree, points.T)
# Choose one of the cells that contains the point
colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, points.T)
for i, point in enumerate(points.T):
    if len(colliding_cells.links(i)) > 0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(i)[0])
points_on_proc = np.array(points_on_proc, dtype=np.float64)
Psi_values = psi.eval(points_on_proc, cells)
B_values = B.eval(points_on_proc, cells)

x_new = points_on_proc[:, 1]

B_z_teo = 4 * np.pi * 1e-7 * I_wire * rc**2 / (2 * np.sqrt(x_new**2 + rc**2) ** 3)

fig = plt.figure()
plt.plot(x_new, B_values[:, 1], "b--", linewidth=2, label="B")
plt.plot(x_new, B_z_teo, "g-", linewidth=2, label="B_y_teo")

plt.grid(True)
plt.xlabel("x")
plt.legend()
# If run in parallel as a python file, we save a plot per processor
plt.savefig(f"B_Az_rank{MPI.COMM_WORLD.rank:d}.png")
plt.show()

adiff = B_values[:, 1] - B_z_teo
fig = plt.figure()
plt.plot(x_new, adiff, linewidth=2, label="B absolute diff")

plt.grid(True)
plt.xlabel("x")
plt.legend()
# If run in parallel as a python file, we save a plot per processor
plt.savefig(f"B_absolute_diff{MPI.COMM_WORLD.rank:d}.png")
plt.show()

rdiff = (B_values[:, 1] - B_z_teo) / B_z_teo
fig = plt.figure()
plt.plot(x_new, rdiff, linewidth=2, label="B relative diff")

plt.grid(True)
plt.xlabel("x")
plt.legend()
# If run in parallel as a python file, we save a plot per processor
plt.savefig(f"B_relative_diff{MPI.COMM_WORLD.rank:d}.png")
plt.show()


dof_points = B.function_space.tabulate_dof_coordinates()[:, 0:2]
dofs = []

r_min = 2.8
r_max = 3.2
z_min = -0.2
z_max = 0.2

for ii in range(len(dof_points)):
    if r_min <= dof_points[ii, 0] <= r_max and z_min <= dof_points[ii, 1] <= z_max:
        dofs.append(ii)

plot_scalar_field(
    dof_points[dofs, 0],
    dof_points[dofs, 1],
    np.reshape(B.x.array, (-1, 2))[dofs, 1],
    levels=21,
    ax=None,
    tofill=True,
)
plt.title("B field near the coil")
plt.show()
