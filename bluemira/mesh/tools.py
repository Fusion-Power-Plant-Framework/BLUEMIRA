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
Converter from MSH to XDMF mesh file formats, derived from F. Loiseau,
R. Delaporte-Mathurin, and C. Weickhmann's https://github.com/floiseau/msh2xdmf

Credit: F. Loiseau, R. Delaporte-Mathurin, and C. Weickhmann
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import meshio
import numpy as np
from dolfin import Mesh, MeshValueCollection, XDMFFile
from dolfin.cpp.mesh import MeshFunctionSizet
from tabulate import tabulate

from bluemira.base.look_and_feel import bluemira_debug, bluemira_warn
from bluemira.mesh.error import MeshConversionError

__all__ = ("msh_to_xdmf", "import_mesh")


CELL_TYPE_DIM = {
    2: "line",
    3: "triangle",
    4: "tetra",
}


GMSH_PHYS = "gmsh:physical"
GMSH_BE = "gmsh:bounding_entities"

DOMAIN_SUFFIX = "domain.xdmf"
BOUNDARY_SUFFIX = "boundaries.xdmf"
LINKFILE_SUFFIX = "linkfile.json"


def msh_to_xdmf(
    mesh_name: str,
    dimensions: Union[Tuple[int], int] = (0, 2),
    directory: str = ".",
):
    """
    Convert a MSH file to an XMDF file.

    Parameters
    ----------
    mesh_name: str
        Name of the MSH file to convert to XDMF
    dimensions: Union[Tuple[int], int]
        Dimensions of the mesh (0: x, 1: y, 2: z), defaults to x-z
        (0, 1, 2) would be a 3-D mesh
    directory: str
        Directory in which the MSH file exists and where the XDMF files will be written

    Raises
    ------
    MeshConversionError:
        * If the file does not exist
        * If the dimensionality != [2, 3]
        * If no domain physical groups are found

    Notes
    -----
    Creates the following files:
        * DOMAIN_SUFFIX
        * BOUNDARY_SUFFIX
        * LINKFILE_SUFFIX
    """
    dimensions = _check_dimensions(dimensions)

    file_path = Path(directory, mesh_name)
    if not file_path.exists():
        raise MeshConversionError(f"No such file: {file_path}")

    file_prefix = mesh_name.split(".")[0]
    mesh = meshio.read(file_path.as_posix())
    _export_domain(mesh, file_prefix, directory, dimensions)
    _export_boundaries(mesh, file_prefix, directory, dimensions)
    _export_link_file(mesh, file_prefix, directory)


def import_mesh(
    file_prefix: str = "mesh", subdomains: bool = False, directory: str = "."
) -> Tuple[Mesh, MeshFunctionSizet, Optional[MeshFunctionSizet], dict]:
    """
    Import a dolfin mesh.

    Parameters
    ----------
    file_prefix:
        File prefix to use when importing a mesh (defaults to 'mesh')
    subdomains:
        Whether or not to subdomains are present (defaults to False)
    directory:
        Directory in which the MSH file and XDMF files exist

    Returns
    -------
    mesh:
        Dolfin Mesh object containing the domain
    boundaries_mf:
        Dolfin MeshFunctionSizet object containing the geometry
    subdomains_mf:
        Dolfin MeshFunctionSizet object containing the geometry
    link_dict:
        Link dictionary between MSH and XDMF objects
    """
    domain_file = Path(directory, f"{file_prefix}_{DOMAIN_SUFFIX}")
    boundary_file = Path(directory, f"{file_prefix}_{BOUNDARY_SUFFIX}")
    link_file = Path(directory, f"{file_prefix}_{LINKFILE_SUFFIX}")
    files = [domain_file, boundary_file, link_file]
    exists = [file.exists() for file in files]

    if not all(exists):
        msg = "\n".join([fn.as_posix() for fn, exist in zip(files, exists) if not exist])
        raise MeshConversionError(f"No mesh file(s) found:\n {msg}")

    mesh = Mesh()

    with XDMFFile(domain_file.as_posix()) as file:
        file.read(mesh)

    dimension = mesh.topology().dim()
    boundaries_mvc = MeshValueCollection("size_t", mesh, dim=dimension)

    with XDMFFile(boundary_file.as_posix()) as file:
        file.read(boundaries_mvc, "boundaries")

    boundaries_mf = MeshFunctionSizet(mesh, boundaries_mvc)

    if subdomains:
        subdomains_mvc = MeshValueCollection("size_t", mesh, dim=dimension)
        with XDMFFile(domain_file.as_posix()) as file:
            file.read(subdomains_mvc, "subdomains")
        subdomains_mf = MeshFunctionSizet(mesh, subdomains_mvc)
    else:
        subdomains_mf = None

    with open(link_file) as file:
        link_dict = json.load(file)

    return mesh, boundaries_mf, subdomains_mf, link_dict


def _check_dimensions(dimensions: Union[int, List[int]]) -> Tuple[int]:
    if isinstance(dimensions, int):
        dimensions = tuple(np.arange(dimensions))

    if len(dimensions) not in [2, 3]:
        raise MeshConversionError(
            f"Length of dimensions must be either 2 or 3, not: {len(dimensions)}"
        )
    for dim in dimensions:
        if dim not in [0, 1, 2]:
            raise MeshConversionError(
                f"Dimensions tuple must contain integers 0, 1, or 2, not: {dim}"
            )

    if len(dimensions) != len(set(dimensions)):
        raise MeshConversionError(
            f"Dimensions tuple cannot have repeated integers: {dimensions}"
        )
    return dimensions


def _export_domain(mesh, file_prefix, directory, dimensions):
    """
    Export the domain of a mesh to XDMF.
    """
    dimensions = _check_dimensions(dimensions)

    cell_type = CELL_TYPE_DIM[len(dimensions) + 1]
    data = _get_data(mesh, cell_type)

    if len(data) == 0:
        bluemira_warn(f"No domain physical group found in: {file_prefix}")
        return

    if GMSH_PHYS not in mesh.cell_data:
        raise MeshConversionError(f"No domain physical group found in: {file_prefix}")

    cells = _make_cellblocks(data, cell_type)

    subdomains = _get_cells(mesh, cell_type)

    cell_data = {"subdomains": [np.concatenate(subdomains)]}

    domain = _make_mesh(mesh, dimensions, cells, cell_data)

    _write_mesh(
        Path(directory, f"{file_prefix}_{DOMAIN_SUFFIX}").as_posix(),
        domain,
    )


def _export_boundaries(mesh, file_prefix, directory, dimensions):
    """
    Export the boundaries of a mesh to XDMF.
    """
    dimensions = _check_dimensions(dimensions)

    cell_type = CELL_TYPE_DIM[len(dimensions)]
    data = _get_data(mesh, cell_type)

    if len(data) == 0:
        bluemira_warn(f"No boundary physical group found in: {file_prefix}")
        return

    cells = _make_cellblocks(data, cell_type)

    boundaries = _get_cells(mesh, cell_type)

    cell_data = {"boundaries": [np.concatenate(boundaries)]}

    boundaries = _make_mesh(mesh, dimensions, cells, cell_data)

    _write_mesh(
        Path(directory, f"{file_prefix}_{BOUNDARY_SUFFIX}").as_posix(),
        boundaries,
    )


def _export_link_file(mesh, file_prefix, directory):
    """
    Export the association file between MSH and XDMF objects.
    """
    table = {}
    for key, arrays in mesh.cell_sets.items():
        for i, array in enumerate(arrays):
            if array.size != 0:
                index = i
        if key != GMSH_BE:
            value = mesh.cell_data[GMSH_PHYS][index][0]
            table[key] = int(value)

    bluemira_debug(
        tabulate(
            list(table.items()),
            headers=["GMSH label", "MeshFunction value"],
            tablefmt="simple",
        )
    )

    with open(Path(directory, f"{file_prefix}_{LINKFILE_SUFFIX}"), "w") as file:
        json.dump(table, file, indent=4)


def _get_data(mesh, cell_type):
    return [array for (typ, array) in mesh.cells if typ == cell_type]


def _make_cellblocks(data, cell_type):
    return [meshio.CellBlock(cell_type, np.concatenate(data))]


def _make_mesh(mesh, dimensions, cells, cell_data):
    return meshio.Mesh(
        mesh.points[:, list(dimensions)], cells=cells, cell_data=cell_data
    )


def _get_cells(mesh, cell_type):
    return [
        mesh.cell_data[GMSH_PHYS][i]
        for i, cell in enumerate(mesh.cells)
        if cell.type == cell_type
    ]


def _write_mesh(filename, obj):
    meshio.write(
        filename,
        obj,
        file_format="xdmf",
    )
