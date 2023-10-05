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
Couple PLASMOD to a 2-D asymmetric fixed boundary equilibrium solve

NOTE: This procedure is known to be sensitive to inputs, exercise
caution.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple, Union

if TYPE_CHECKING:
    from bluemira.codes.interface import BaseRunMode, CodesSolver
    from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
        FemGradShafranovFixedBoundary,
    )
    from bluemira.equilibria.flux_surfaces import ClosedFluxSurface
    from bluemira.geometry.parameterisations import GeometryParameterisation

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from tabulate import tabulate

from bluemira.base.components import PhysicalComponent
from bluemira.base.constants import EPS, MU_0
from bluemira.base.file import get_bluemira_path, try_get_bluemira_path
from bluemira.base.look_and_feel import bluemira_debug, bluemira_print, bluemira_warn
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.codes.plasmod import plot_default_profiles
from bluemira.equilibria.constants import DPI_GIF, PLT_PAUSE
from bluemira.equilibria.fem_fixed_boundary.utilities import (
    calculate_plasma_shape_params,
    create_mesh,
    find_magnetic_axis,
    get_flux_surfaces_from_mesh,
    refine_mesh,
)
from bluemira.geometry.constants import D_TOLERANCE
from bluemira.geometry.face import BluemiraFace
from bluemira.optimisation._tools import approx_derivative
from bluemira.utilities.plot_tools import make_gif, save_figure

__all__ = ["solve_transport_fixed_boundary"]


@dataclass
class PlasmaFixedBoundaryParams:
    """Plasma Transport Fixed Boundary parameters"""

    r_0: float
    a: float
    kappa_u: float
    kappa_l: float
    delta_u: float
    delta_l: float

    _fields = None

    def tabulate(self) -> str:
        """
        Tabulate dataclass
        """
        return tabulate(
            list(asdict(self).items()),
            headers=["name", "value"],
            tablefmt="simple",
            numalign="right",
        )

    @classmethod
    def fields(cls) -> List:
        """List of fields in the dataclass"""
        if cls._fields is None:
            cls._fields = [k.name for k in fields(cls)]
        return cls._fields


@dataclass
class TransportSolverParams(ParameterFrame):
    """Transport Solver ParameterFrame"""

    A: Parameter[float]
    R_0: Parameter[float]
    I_p: Parameter[float]
    B_0: Parameter[float]
    V_p: Parameter[float]
    v_burn: Parameter[float]
    kappa_95: Parameter[float]
    delta_95: Parameter[float]
    delta: Parameter[float]
    kappa: Parameter[float]
    q_95: Parameter[float]
    f_ni: Parameter[float]


def create_plasma_xz_cross_section(
    parameterisation: GeometryParameterisation,
    transport_params: ParameterFrame,
    params: PlasmaFixedBoundaryParams,
    kappa_95: float,
    delta_95: float,
    lcfs_options: Dict[str, Dict],
    source: str,
) -> PhysicalComponent:
    """
    Build the plasma x-z cross-section, get its volume and update transport solver
    parameters
    """
    plasma = PhysicalComponent(
        "Plasma", shape=BluemiraFace(parameterisation.create_shape())
    )
    lcfs = plasma.shape
    plasma_volume = 2 * np.pi * lcfs.center_of_mass[0] * lcfs.area

    # Update transport parameter valupes
    transport_params.kappa.set_value((params.kappa_u + params.kappa_l) / 2, source)
    transport_params.delta.set_value((params.delta_u + params.delta_l) / 2, source)

    transport_params.V_p.set_value(plasma_volume, source)
    transport_params.kappa_95.set_value(kappa_95, source)
    transport_params.delta_95.set_value(delta_95, source)

    # Set mesh options
    lcfs.boundary[0].mesh_options = lcfs_options["lcfs"]
    lcfs.mesh_options = lcfs_options["face"]

    bluemira_debug(
        "FB Params\n\n"
        f"{params.tabulate()}\n\n"
        "Transport Params\n\n"
        f"{transport_params.tabulate(keys=['name', 'value', 'unit'], tablefmt='simple')}"
    )
    return plasma


def _run_transport_solver(
    transport_solver: CodesSolver,
    transport_params: ParameterFrame,
    transport_run_mode: Union[str, BaseRunMode],
) -> Tuple[ParameterFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Run transport solver"""
    transport_solver.params.update_from_frame(transport_params)
    transp_out_params = transport_solver.execute(transport_run_mode)

    return (
        transp_out_params,
        transport_solver.get_profile("x"),
        transport_solver.get_profile("pprime"),
        transport_solver.get_profile("ffprime"),
    )


def _update_delta_kappa(
    params: PlasmaFixedBoundaryParams,
    kappa_95: float,
    kappa95_t: float,
    delta_95: float,
    delta95_t: float,
    relaxation: float,
) -> float:
    """
    Recalculate Delta and Kappa and calculate the iteration error
    """
    err_delta = abs(delta_95 - delta95_t) / delta95_t
    err_kappa = abs(kappa_95 - kappa95_t) / kappa95_t
    iter_err = max(err_delta, err_kappa)

    # calculate the new kappa_u and delta_u
    kappa_u_0 = params.kappa_u
    delta_u_0 = params.delta_u

    kappa_u = (1 - relaxation) * kappa_u_0 * (
        kappa95_t / kappa_95
    ) + relaxation * kappa_u_0
    delta_u = (1 - relaxation) * delta_u_0 * (
        delta95_t / delta_95
    ) + relaxation * delta_u_0

    bluemira_debug(
        "Previous shape parameters:\n\t"
        f"{kappa_u_0=:.3e}, {delta_u_0=:.3e}\n"
        "Recalculated shape parameters:\n\t"
        f"{kappa_u=:.3e}, {delta_u=:.3e}\n\n"
        f"Target kappa 95, actual kappa 95 = {kappa95_t:.3e}, {kappa_95:.3e}\n"
        f"Target delta 95, actual delta 95 = {delta95_t:.3e}, {delta_95:.3e}\n"
    )
    params.kappa_u = kappa_u
    params.delta_u = delta_u

    return iter_err


def solve_transport_fixed_boundary(
    parameterisation: GeometryParameterisation,
    transport_solver: CodesSolver,
    gs_solver: FemGradShafranovFixedBoundary,
    kappa95_t: float,
    delta95_t: float,
    lcar_mesh: float = 0.15,
    max_iter: int = 30,
    iter_err_max: float = 1e-5,
    max_inner_iter: int = 20,
    inner_iter_err_max: float = 1e-4,
    relaxation: float = 0.2,
    transport_run_mode: Union[str, BaseRunMode] = "run",
    mesh_filename: str = "FixedBoundaryEquilibriumMesh",
    plot: bool = False,
    debug: bool = False,
    gif: bool = False,
    refine: bool = False,
    num_levels: int = 2,
    distance: float = 1.0,
    ny_fs_min: int = 40,
):
    """
    Solve the plasma fixed boundary problem using delta95 and kappa95 as target
    values and iterating on a transport solver to have consistency with pprime
    and ffprime.

    Parameters
    ----------
    parameterisation:
        Geometry parameterisation class for the plasma
    transport_solver:
        Transport Solver to call
    gs_solver:
        Grad-Shafranov Solver instance
    kappa95_t:
        Target value for kappa at 95%
    delta95_t:
        Target value for delta at 95%
    lcar_mesh:
        Value of the characteristic length used to generate the mesh to solve the
        Grad-Shafranov problem
    max_iter:
        Maximum number of iteration between Grad-Shafranov and the transport solver
    iter_err_max:
        Convergence maximum error to stop the iteration
    max_inner_iter:
        Maximum number of inner iterations on the flux functions
    inner_iter_err_max:
        Inner convergence error on when iterating flux functions
    relaxation:
        Iteration relaxing factor
    transport_run_mode:
        Run mode for transport solver
    mesh_filename:
        filename for mesh output file
    plot:
        Whether or not to plot
    refine:
        Whether or not the mesh should be refined around the magnetic axis
    num_levels:
        Number of refinement levels
    distance:
        Maximum distance from the magnetic axis to which the refinement will be applied
    ny_fs_min:
        Minimum number of points in a flux surface extracted from the mesh.

    Returns
    -------
    equilibrium: FixedBoundaryEquilibrium
        Final fixed boundary equilibrium result from the transport <-> fixed boundary
        equilibrium solve
    """
    kappa_95 = kappa95_t
    delta_95 = delta95_t

    directory = get_bluemira_path("", subfolder="generated_data")
    mesh_name_msh = mesh_filename + ".msh"

    paramet_params = PlasmaFixedBoundaryParams(
        **{
            k: v
            for k, v in zip(
                parameterisation.variables.names, parameterisation.variables.values
            )
            if k in PlasmaFixedBoundaryParams.fields()
        }
    )

    transport_params = TransportSolverParams.from_frame(
        deepcopy(transport_solver.params)
    )

    lcfs_options = {
        "face": {"lcar": lcar_mesh, "physical_group": "plasma_face"},
        "lcfs": {"lcar": lcar_mesh, "physical_group": "lcfs"},
    }

    plot = any((plot, debug, gif))
    folder = try_get_bluemira_path("", subfolder="generated_data", allow_missing=False)
    figname = "Transport iteration "
    f, ax = None, None

    for n_iter in range(max_iter):
        transp_out_params, x, pprime, ffprime = _run_transport_solver(
            transport_solver, transport_params, transport_run_mode
        )

        f_pprime = interp1d(x, pprime, fill_value="extrapolate")
        f_ffprime = interp1d(x, ffprime, fill_value="extrapolate")

        psi_plasmod = transport_solver.get_profile("psi")
        x_psi_plasmod = np.sqrt(psi_plasmod / psi_plasmod[-1])

        q = transport_solver.get_profile("q")
        press = transport_solver.get_profile("pressure")
        q_func = interp1d(x, q, fill_value="extrapolate")
        p_func = interp1d(x, press, fill_value="extrapolate")

        if plot:
            if ax is not None:
                for axis in ax.flat:
                    axis.clear()
            f, ax = plot_default_profiles(transport_solver, show=False, f=f, ax=ax)

            f.suptitle(figname + str(n_iter))
            plt.pause(PLT_PAUSE)
            if debug or gif:
                save_figure(
                    f,
                    figname + str(n_iter),
                    save=True,
                    folder=folder,
                    dpi=DPI_GIF,
                )

        plasma = create_plasma_xz_cross_section(
            parameterisation,
            transport_params,
            paramet_params,
            kappa_95,
            delta_95,
            lcfs_options,
            f"from equilibrium iteration {n_iter}",
        )

        mesh = create_mesh(
            plasma,
            directory,
            mesh_filename,
            mesh_name_msh,
        )

        # store the created mesh as coarse mesh
        coarse_mesh = mesh

        gs_solver.set_mesh(mesh)

        points = gs_solver.mesh.coordinates()
        psi2d_0 = np.zeros(len(points))

        for n_iter_inner in range(max_inner_iter):
            gs_solver.set_profiles(
                f_pprime,
                f_ffprime,
                transp_out_params.I_p.value,
                transp_out_params.B_0.value,
                transp_out_params.R_0.value,
            )

            bluemira_print(
                "Solving fixed boundary Grad-Shafranov...[inner iteration:"
                f" {n_iter_inner}]"
            )

            equilibrium = gs_solver.solve(
                plot=plot,
                debug=debug,
                gif=gif,
                figname=f"{n_iter} Fixed boundary equilibrium iteration ",
            )

            x1d, flux_surfaces = get_flux_surfaces_from_mesh(
                mesh,
                gs_solver.psi_norm_2d,
                x_1d=x_psi_plasmod,
                ny_fs_min=ny_fs_min,
            )

            x1d, volume, _, g2, g3 = calc_metric_coefficients(
                flux_surfaces, gs_solver.grad_psi, x1d, gs_solver.psi_ax
            )
            _, _, _, pprime, _, ffprime = calc_curr_dens_profiles(
                x1d,
                p_func(x1d),
                q_func(x1d),
                g2,
                g3,
                volume,
                transp_out_params.I_p.value,
                transp_out_params.B_0.value,
                transp_out_params.R_0.value,
                gs_solver.psi_ax,
                gs_solver.psi_b,
            )

            f_pprime = interp1d(x1d, pprime, fill_value="extrapolate")
            f_ffprime = interp1d(x1d, ffprime, fill_value="extrapolate")

            psi2d = np.array([gs_solver.psi(p) for p in points])

            eps_psi2d = np.linalg.norm(psi2d - psi2d_0, ord=2) / np.linalg.norm(
                psi2d, ord=2
            )

            if eps_psi2d < inner_iter_err_max:
                break

            bluemira_print(f"Error on psi2d = {eps_psi2d} > {inner_iter_err_max}")
            psi2d_0 = psi2d
            if refine:
                magnetic_axis = find_magnetic_axis(gs_solver.psi, gs_solver.mesh)
                magnetic_axis = np.array([magnetic_axis[0], magnetic_axis[1], 0])
                mesh = refine_mesh(coarse_mesh, magnetic_axis, distance, num_levels)
                bluemira_print(f"Mesh refined on magnetic axis {magnetic_axis[:2]}")
                gs_solver.set_mesh(mesh)

        _, kappa_95, delta_95 = calculate_plasma_shape_params(
            gs_solver.psi_norm_2d,
            mesh,
            np.sqrt(0.95),
        )

        iter_err = _update_delta_kappa(
            paramet_params,
            kappa_95,
            kappa95_t,
            delta_95,
            delta95_t,
            relaxation,
        )

        bluemira_print(
            f"{transport_solver.name} <-> Fixed boundary G-S iter {n_iter} :"
            f" {iter_err:.3E}"
        )

        if iter_err <= iter_err_max:
            message = bluemira_print
            line_1 = f"successfully converged in {n_iter} iterations"
            ltgt = "<"
            break

        # update parameters
        for name, value in asdict(paramet_params).items():
            parameterisation.adjust_variable(name, value)

    else:
        # If we don't break we didn't converge
        message = bluemira_warn
        line_1 = f"did not converge within {max_iter} iterations"
        ltgt = ">"

    message(
        f"{transport_solver.name} <-> Fixed boundary G-S {line_1}:\n\t"
        f"Target kappa_95: {kappa95_t:.3f}\n\t"
        f"Actual kappa_95: {kappa_95:.3f}\n\t"
        f"Target delta_95: {delta95_t:.3f}\n\t"
        f"Actual delta_95: {delta_95:.3f}\n\t"
        f"Error: {iter_err:.3E} {ltgt} {iter_err_max:.3E}\n"
    )

    if gif:
        make_gif(folder, figname, clean=not debug)
    return equilibrium


def calc_metric_coefficients(
    flux_surfaces: List[ClosedFluxSurface],
    grad_psi_2D_func: Callable[[np.ndarray], np.ndarray],
    psi_norm_1D: np.ndarray,
    psi_ax: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate metric coefficients of a set of flux surfaces.

    Parameters
    ----------
    flux_surfaces: List[ClosedFluxSurface]
        List of closed flux surfaces on which to calculate the coefficients
    grad_psi_2D_func:
        Callable which calculates grad psi of the form f(p: Iterable[2]) = float
    psi_norm_1D:
        Array of 1-D normalised psi values
    psi_ax:
        Poloidal magnetic flux at the magnetic axis

    Returns
    -------
    psi_norm_1D:
        1-D vector of normalised psi values at which the coefficients were calculated
    volume:
        1-D volume vector
    g1:
        1-D g1 vector
    g2:
        1-D g2 vector
    g3:
        1-D g3 vector
    """
    if psi_norm_1D[0] != 0:
        # Initialise with 0 at axis
        psi_norm_1D = np.insert(psi_norm_1D, 0, 0)
    nx = psi_norm_1D.size

    g1, g2, g3, volume = np.zeros((4, nx))
    volume[1:] = [fs.volume for fs in flux_surfaces]

    volume_func = interp1d(psi_norm_1D, volume, fill_value="extrapolate")
    grad_vol_1D_array = approx_derivative(volume_func, psi_norm_1D).diagonal()

    def grad_psi_norm(x: np.ndarray) -> np.ndarray:
        return np.hypot(*grad_psi_2D_func(x))

    for i, fs in enumerate(flux_surfaces):
        points = fs.coords.xz.T
        dx = np.diff(fs.coords.x)
        dz = np.diff(fs.coords.z)
        dl = np.hypot(dx, dz)
        x_data = np.concatenate([np.array([0.0]), np.cumsum(dl)])

        psi_norm_fs = psi_norm_1D[i + 1]

        grad_psi_norm_points = np.array([grad_psi_norm(p) for p in points])
        # Scale from grad_psi_norm to get the grad_psi_norm_norm
        psi_fs = psi_ax * (1 - psi_norm_fs**2)
        factor = 1 / (2 * psi_ax * np.sqrt(1 - psi_fs / psi_ax))
        grad_psi_norm_norm_points = factor * grad_psi_norm_points

        # Poloidal field
        bp = grad_psi_norm_points / (2 * np.pi * fs.coords.x)

        grad_vol_1D_fs = grad_vol_1D_array[i + 1]
        grad_vol_norm_2 = (grad_psi_norm_norm_points * grad_vol_1D_fs) ** 2

        y0_data = 1 / bp
        y1_data = grad_vol_norm_2 * y0_data
        y3_data = 1 / (fs.coords.x**2 * bp)
        y2_data = grad_vol_norm_2 * y3_data

        denom = np.trapz(y0_data, x_data)
        g1[i + 1] = np.trapz(y1_data, x_data) / denom
        g2[i + 1] = np.trapz(y2_data, x_data) / denom
        g3[i + 1] = np.trapz(y3_data, x_data) / denom
        # NOTE: To future self, g1 is not used (right now), and the calculation could
        # be removed to speed things up.

    g2_temp = interp1d(
        psi_norm_1D[1:-1], g2[1:-1], "quadratic", fill_value="extrapolate"
    )
    g2[-1] = g2_temp(psi_norm_1D[-1])

    g3_temp = interp1d(psi_norm_1D[1:-1], g3[1:-1], fill_value="extrapolate")
    g3[0] = g3_temp(psi_norm_1D[0])
    g3[-1] = g3_temp(psi_norm_1D[-1])

    return psi_norm_1D, volume, g1, g2, g3


def calc_curr_dens_profiles(
    psi_norm_1D: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    g2: np.ndarray,
    g3: np.ndarray,
    volume: np.ndarray,
    I_p: float,
    B_0: float,
    R_0: float,
    psi_ax: float,
    psi_b: float,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate pprime and ffprime from metric coefficients, emulating behaviour
    in PLASMOD.

    Parameters
    ----------
    psi_norm_1D:
        1-D normalised psi array
    p:
        1-D pressure array
    q:
        1-D safety factor array
    g2:
        1-D g2 metric array
    g3:
        1-D g3 metric array
    volume:
        1-D volume array
    I_p:
        Plasma current [A]. If 0.0, recalculated here.
    B_0:
        Toroidal field at R_0 [T]
    R_0:
        Major radius [m]
    psi_ax:
        Poloidal magnetic flux at the magnetic axis [V.s]
    psi_b:
        Poloidal magnetic flux at the boundary [V.s]

    Returns
    -------
    I_p:
        Plasma current [A] (calculated if I_p=0)
    phi_1D:
        Toroidal magnetic flux 1-D array
    psi_1D:
        Poloidal magnetic flux 1-D array
    pprime:
        p' 1-D array
    F:
        F 1-D array
    ff_prime:
        FF' 1-D array

    Notes
    -----
    Fable et al., A stable scheme for computation of coupled transport and
    equilibrium equations in tokamaks

    https://pure.mpg.de/rest/items/item_2144754/component/file_2144753/content
    """
    psi_1D = psi_ax - psi_norm_1D**2 * (psi_ax - psi_b)

    psi_1D_0 = psi_1D
    no_iter = 50
    for _ in range(no_iter):
        # calculate pprime profile from p
        p_fun_psi1D = interp1d(psi_1D, p, "linear", fill_value="extrapolate")
        pprime = approx_derivative(p_fun_psi1D, psi_1D).diagonal()

        # Here we preserve some PLASMOD notation, for future sanity
        q3 = q / g3
        AA = g2 / q3**2 + (16 * np.pi**4) * g3  # noqa: N806
        C = -4 * np.pi**2 * MU_0 * np.gradient(p, psi_norm_1D) / AA  # noqa: N806
        dum3 = g2 / q3
        dum2 = np.gradient(dum3, psi_norm_1D)
        B = -dum2 / q3 / AA
        Fb = -R_0 * B_0 / (2 * np.pi)  # noqa: N806

        dum2 = cumulative_trapezoid(B, psi_norm_1D, initial=0)
        dum1 = np.exp(2.0 * dum2)
        dum1 = dum1 / dum1[-1]
        dum3 = cumulative_trapezoid(C / dum1, psi_norm_1D, initial=0)
        dum3 = dum3 - dum3[-1]

        y = dum1 * (dum3 + 0.5 * Fb**2)
        dum2 = g2 / q3
        dum3 = np.gradient(dum2, psi_1D)
        betahat = dum3 / q3 / AA
        chat = -4 * np.pi**2 * MU_0 * pprime / AA
        # EPS clipping here otherwise DOLFIN crashes with index error in MeshEntity.cpp
        FF = np.sqrt(2.0 * np.clip(y, EPS, None))  # noqa: N806

        ff_prime = 4 * np.pi**2 * (chat - betahat * FF**2)

        phi_1D = -cumulative_trapezoid(q, psi_1D, initial=0)

        F = 2 * np.pi * FF
        d_psi_dv = -FF / q * g3
        psi_1D = np.flip(
            cumulative_trapezoid(np.flip(d_psi_dv), np.flip(volume), initial=0)
        )

        rms_error = np.sqrt(np.mean((psi_1D - psi_1D_0) ** 2))

        psi_1D_0 = psi_1D

        if rms_error <= D_TOLERANCE:
            break
    else:
        bluemira_warn(
            "Jackpot, you've somehow found a set of inputs for which this calculation"
            f" does not converge almost immediately.{rms_error=} after"
            f" {no_iter} iterations."
        )

    if I_p == 0:
        I_p = -g2[-1] * d_psi_dv[-1] / (4 * np.pi**2 * MU_0)

    return I_p, phi_1D, psi_1D, pprime, F, ff_prime
