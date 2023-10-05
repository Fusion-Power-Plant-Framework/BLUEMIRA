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

"""Module to support the fem_fixed_boundary implementation"""

from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union

import dolfin
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from dolfin import BoundaryMesh, Mesh, Vertex
from matplotlib._tri import TriContourGenerator
from matplotlib.pyplot import Axes
from matplotlib.tri import Triangulation
from scipy.interpolate import interp1d

from bluemira.base.components import PhysicalComponent
from bluemira.base.constants import EPS
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.equilibria.flux_surfaces import ClosedFluxSurface
from bluemira.geometry.coordinates import Coordinates
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf
from bluemira.optimisation import optimise
from bluemira.utilities.tools import is_num


def plot_scalar_field(
    x: np.ndarray,
    y: np.ndarray,
    data: np.ndarray,
    levels: int = 20,
    ax: Optional[Axes] = None,
    contour: bool = True,
    tofill: bool = True,
    **kwargs,
) -> Tuple[Axes, Union[Axes, None], Union[Axes, None]]:
    """
    Plot a scalar field

    Parameters
    ----------
    x:
        x coordinate array
    z:
        z coordinate array
    data:
        value array
    levels:
        Number of contour levels to plot
    axis:
        axis onto which to plot
    contour:
        Whether or not to plot contour lines
    tofill:
        Whether or not to plot filled contours

    Returns
    -------
    Matplotlib axis on which the plot ocurred
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    defaults = {"linewidths": 2, "colors": "k"}
    contour_kwargs = {**defaults, **kwargs}

    cntr = None
    cntrf = None

    if contour:
        cntr = ax.tricontour(x, y, data, levels=levels, **contour_kwargs)

    if tofill:
        cntrf = ax.tricontourf(x, y, data, levels=levels, cmap="RdBu_r")
        fig.colorbar(cntrf, ax=ax)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal")

    return ax, cntr, cntrf


def plot_profile(
    x: np.ndarray,
    prof: np.ndarray,
    var_name: str,
    var_unit: str,
    ax: Optional[Axes] = None,
    show: bool = True,
):
    """
    Plot profile
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(x, prof)
    ax.set(xlabel="x (-)", ylabel=f"{var_name} ({var_unit})")
    ax.grid()
    if show:
        plt.show()


def get_tricontours(
    x: np.ndarray, z: np.ndarray, array: np.ndarray, value: Union[float, Iterable]
) -> List[Union[np.ndarray, None]]:
    """
    Get the contours of a value in a triangular set of points.

    Parameters
    ----------
    x:
        The x value array
    z:
        The z value array
    array:
        The value array
    value:
        The value of the desired contour in the array

    Returns
    -------
    The points of the value contour in the array. If no contour is found
    for a value, None is returned
    """
    tri = Triangulation(x, z)
    tcg = TriContourGenerator(tri.get_cpp_triangulation(), array)
    if is_num(value):
        value = [value]

    results = []
    for val in value:
        contour = tcg.create_contour(val)[0]
        if len(contour) > 0:
            results.append(contour[0])
        else:
            bluemira_warn(f"No tricontour found for {val=}")
            results.append(None)
    return results


def find_flux_surface(
    psi_norm_func: Callable[[np.ndarray], float],
    psi_norm: float,
    mesh: Optional[dolfin.Mesh] = None,
    n_points: int = 100,
) -> np.ndarray:
    """
    Find a flux surface in the psi_norm function precisely by normalised psi value.

    Parameters
    ----------
    psi_norm_func:
        Function to calculate normalised psi
    psi_norm:
        Normalised psi value for which to find the flux surface
    mesh:
        Mesh object to use to estimate the flux surface
        If None, reasonable guesses are used.
    n_points:
        Number of points along the flux surface

    Returns
    -------
    x, z coordinates of the flux surface
    """
    x_axis, z_axis = find_magnetic_axis(lambda x: -psi_norm_func(x), mesh=mesh)

    if mesh:
        search_range = mesh.hmax()
        mpoints = mesh.coordinates()
        psi_norm_array = [psi_norm_func(x) for x in mpoints]
        contour = get_tricontours(
            mpoints[:, 0], mpoints[:, 1], psi_norm_array, psi_norm
        )[0]
        d_guess = np.array([abs(np.max(contour[0, :]) - x_axis) - search_range])

        def lower_bound(x):
            return max(0.1, x - search_range)

        def upper_bound(x):
            return x + search_range

    else:
        d_guess = np.array([0.5])

        def lower_bound(_x):
            return 0.1

        def upper_bound(_x):
            return np.inf

    def psi_norm_match(x):
        return abs(psi_norm_func(x) - psi_norm)

    def theta_line(d, theta_i):
        return float(x_axis + d * np.cos(theta_i)), float(z_axis + d * np.sin(theta_i))

    def psi_line_match(d, theta):
        return psi_norm_match(theta_line(d, theta))

    theta = np.linspace(0, 2 * np.pi, n_points - 1, endpoint=False, dtype=float)
    points = np.zeros((2, n_points), dtype=float)
    distances = np.zeros(n_points)

    for i in range(len(theta)):
        result = optimise(
            f_objective=lambda d, i=i: psi_line_match(d, theta[i]),
            x0=d_guess,
            dimensions=1,
            algorithm="SLSQP",
            opt_conditions={"ftol_abs": 1e-14, "max_eval": 1000},
            bounds=(lower_bound(d_guess), upper_bound(d_guess)),
        )
        points[:, i] = theta_line(result.x, theta[i])
        distances[i] = result.x
        d_guess = result.x

    points[:, -1] = points[:, 0]

    return points


def get_mesh_boundary(mesh: dolfin.Mesh) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve the boundary of the mesh, as an ordered set of coordinates.

    Parameters
    ----------
    mesh:
        Mesh for which to retrieve the exterior boundary

    Returns
    -------
    xbdry:
        x coordinates of the boundary
    zbdry:
        z coordinates of the boundary
    """
    boundary = BoundaryMesh(mesh, "exterior")
    edges = boundary.cells()
    check_edge = np.ones(boundary.num_edges())

    index = 0
    temp_edge = edges[index]
    sorted_v = [temp_edge[0]]

    for _i in range(len(edges) - 1):
        temp_v = next(v for v in temp_edge if v not in sorted_v)
        sorted_v.append(temp_v)
        check_edge[index] = 0
        connected = np.nonzero(edges == temp_v)[0]
        index = next(e for e in connected if check_edge[e] == 1)
        temp_edge = edges[index]

    points_sorted = np.array([Vertex(boundary, v).point().array() for v in sorted_v])
    return points_sorted[:, 0], points_sorted[:, 1]


def get_flux_surfaces_from_mesh(
    mesh: dolfin.Mesh,
    psi_norm_func: Callable[[float, float], float],
    x_1d: Optional[np.ndarray] = None,
    nx: Optional[int] = None,
    ny_fs_min: int = 40,
) -> Tuple[np.ndarray, List[ClosedFluxSurface]]:
    """
    Get a list of flux surfaces from a mesh and normalised psi callable.

    Parameters
    ----------
    mesh:
        Mesh for which to extract the flux surfaces
    psi_norm_func:
        Callable for psi_norm on the mesh
    x_1d:
        Array of 1-D normalised psi_values [0..1]. If None, nx will
        define a linearly spaced vector.
    nx:
        Number of points to linearly space along [0..1]. If x_1d is
        defined, not used.
    ny_fs_min:
        Minimum number of points in a flux surface retrieved from mesh. Below
        this, flux surfaces will be discarded.

    Returns
    -------
    x_1d:
        The 1-D normalised psi_values for which flux surfaces could be
        retrieved.
    flux_surfaces:
        The list of closed flux surfaces

    Notes
    -----
    x_1d is returned, as it is not always possible to return a flux surface for
    small values of normalised psi.
    Some flux surfaces near the axis can have few points and be relatively
    distorted, causing convergence issues. Make sure to change the cut-off
    when changing the mesh discretisation.
    """
    if x_1d is None:
        if nx is None:
            raise ValueError("Please input either x_1d: np.ndarray or nx: int.")
        x_1d = np.linspace(0, 1, nx)
    elif nx is not None:
        bluemira_warn("x_1d and nx specified, discarding nx.")

    mesh_points = mesh.coordinates()
    x = mesh_points[:, 0]
    z = mesh_points[:, 1]
    psi_norm_data = np.array([psi_norm_func(p) for p in mesh_points])

    index = []
    flux_surfaces = []
    for i, xi in enumerate(x_1d):
        if np.isclose(xi, 1.0, rtol=0, atol=EPS):
            path = get_mesh_boundary(mesh)
            fs = Coordinates({"x": path[0], "z": path[1]})
            fs.close()
            flux_surfaces.append(ClosedFluxSurface(fs))
        else:
            path = get_tricontours(x, z, psi_norm_data, xi)[0]
            if path is not None and len(path.T[0]) > ny_fs_min:
                # Only capture flux surfaces with sufficient points
                fs = Coordinates({"x": path.T[0], "z": path.T[1]})
                fs.close()
                flux_surfaces.append(ClosedFluxSurface(fs))
            else:
                index.append(i)

    mask = np.ones_like(x_1d, dtype=bool)
    mask[index] = False
    return x_1d[mask], flux_surfaces


def calculate_plasma_shape_params(
    psi_norm_func: Callable[[np.ndarray], np.ndarray],
    mesh: dolfin.Mesh,
    psi_norm: float,
    plot: bool = False,
) -> Tuple[float, float, float]:
    """
    Calculate the plasma parameters (r_geo, kappa, delta) for a given magnetic
    isoflux from the mesh.

    Parameters
    ----------
    psi_norm_func:
        Function to calculate normalised psi
    mesh:
        Mesh object to use to estimate extrema prior to optimisation
    psi_norm:
        Normalised psi value for which to calculate the shape parameters
    plot:
        Whether or not to plot

    Returns
    -------
    r_geo:
        Geometric major radius of the flux surface at psi_norm
    kappa:
        Elongation of the flux surface at psi_norm
    delta:
        Triangularity of the flux surface at psi_norm
    """
    points = mesh.coordinates()
    psi_norm_array = [psi_norm_func(x) for x in points]

    contour = get_tricontours(points[:, 0], points[:, 1], psi_norm_array, psi_norm)[0]
    x, z = contour.T

    pu = contour[np.argmax(z)]
    pl = contour[np.argmin(z)]
    po = contour[np.argmax(x)]
    pi = contour[np.argmin(x)]

    if plot:
        _, ax = plt.subplots()
        dolfin.plot(mesh)
        ax.tricontour(points[:, 0], points[:, 1], psi_norm_array)
        ax.plot(x, z, color="r")
        ax.plot(*po, marker="o", color="r")
        ax.plot(*pi, marker="o", color="r")
        ax.plot(*pu, marker="o", color="r")
        ax.plot(*pl, marker="o", color="r")

        ax.set_aspect("equal")
        plt.show()

    # geometric center of a magnetic flux surface
    r_geo = 0.5 * (po[0] + pi[0])

    # elongation
    a = 0.5 * (po[0] - pi[0])
    b = 0.5 * (pu[1] - pl[1])
    kappa = 1 if a == 0 else b / a

    # triangularity
    c = r_geo - pl[0]
    d = r_geo - pu[0]
    delta = 0 if a == 0 else 0.5 * (c + d) / a

    return r_geo, kappa, delta


def find_magnetic_axis(
    psi_func: Callable[[np.ndarray], float], mesh: Optional[dolfin.Mesh] = None
) -> np.ndarray:
    """
    Find the magnetic axis in the poloidal flux map.

    Parameters
    ----------
    psi_func:
        Function to return psi at a given point
    mesh:
        Mesh object to use to estimate magnetic axis prior to optimisation
        If None, a reasonable guess is made.

    Returns
    -------
    Position vector (2) of the magnetic axis [m]
    """
    if mesh:
        points = mesh.coordinates()
        psi_array = [psi_func(x) for x in points]
        psi_max_arg = np.argmax(psi_array)

        x0 = points[psi_max_arg]
        search_range = mesh.hmax()
        lower_bounds = x0 - search_range
        upper_bounds = x0 + search_range
    else:
        x0 = np.array([0.1, 0.0])
        lower_bounds = np.array([0, -2.0])
        upper_bounds = np.array([20.0, 2.0])

    def maximise_psi(x: npt.NDArray) -> float:
        return -psi_func(x)

    x_star = optimise(
        f_objective=maximise_psi,
        x0=x0,
        dimensions=2,
        algorithm="SLSQP",
        opt_conditions={"ftol_abs": 1e-6, "max_eval": 1000},
        bounds=(lower_bounds, upper_bounds),
    )

    return x_star.x


def _interpolate_profile(
    x: np.ndarray, profile_data: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Interpolate profile data"""
    return interp1d(x, profile_data, kind="linear", fill_value="extrapolate")


def _cell_near_point(cell: dolfin.Cell, refine_point: Iterable, distance: float) -> bool:
    """
    Determine whether or not a cell is in the vicinity of a point.

    Parameters
    ----------
    cell:
        Cell to check for vicintiy to a point
    refine_point:
        Point from which to determine vicinity to a cell
    distance:
        Distance away from the midpoint of the cell to determine vicinity

    Returns
    -------
    Whether or not the cell is in the vicinity of a point
    """
    # Get the center of the cell
    # Calculate the distance between the cell center and the refinement point
    # Refine the cell if it is close to the refinement point
    return np.linalg.norm(cell.midpoint()[:] - np.array(refine_point)) < distance


def refine_mesh(
    mesh: dolfin.Mesh,
    refine_point: Iterable[float],
    distance: float,
    num_levels: int = 1,
) -> dolfin.Mesh:
    """
    Refine the mesh around a reference point.

    Parameters
    ----------
    mesh:
        Mesh to refine
    refine_point:
        Point at which to refine the mesh
    distance:
        Refinement distance from the point
    num_levels:
        Number of refinement levels

    Returns
    -------
    Refined mesh
    """
    for _ in range(num_levels):
        cell_markers = dolfin.MeshFunction("bool", mesh, mesh.topology().dim())
        cell_markers.set_all(False)
        for cell in dolfin.cells(mesh):
            if _cell_near_point(cell, refine_point, distance):
                cell_markers[cell.index()] = True
        mesh = dolfin.refine(mesh, cell_markers)

    return mesh


def create_mesh(
    plasma: PhysicalComponent,
    directory: str,
    mesh_filename: str,
    mesh_name_msh: str,
) -> Mesh:
    """
    Create mesh
    """
    meshing.Mesh(meshfile=Path(directory, mesh_name_msh).as_posix())(plasma)
    msh_to_xdmf(mesh_name_msh, dimensions=(0, 2), directory=directory)
    return import_mesh(mesh_filename, directory=directory, subdomains=True)[0]
